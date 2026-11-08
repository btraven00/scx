#!/usr/bin/env python3
"""Sample a reproducible test corpus of CELLxGENE h5ad datasets.

Two steps, so a sample can be redrawn exactly:

  cxg_sample.py snapshot catalog.json
      Save the public CELLxGENE catalogue (curation API) with its fetch time
      and SHA-256. The catalogue changes as datasets are added; the snapshot
      does not.

  cxg_sample.py sample catalog.json --seed 1 --per-year 10 --n 30 \
        [--organism "Homo sapiens"] [--assay 10x] [--max-mb 500] \
        [--min-cells N] [--max-cells N] [--title REGEX] [--year 2025] > corpus.tsv

      Filter, then take up to --per-year datasets from each publication year,
      newest year first, until --n are chosen. Within a year, datasets are
      ranked by sha256("<seed>:<dataset_version_id>"), so the same snapshot,
      seed and query give the same corpus, and a dataset added later does not
      reorder the others. The TSV header records the snapshot hash, seed and
      query.

Year is the dataset's `published_at` year (its upload to CELLxGENE). Every file
is rewritten at each schema migration, so year spreads the corpus over content
(assays, sizes, obs columns), not over writer versions.

Download a corpus with hapiq, e.g.:
  cut -f9 corpus.tsv | grep -v '^#' | tail -n +2 | \
    xargs -I{} hapiq download url {} --out data -y
"""

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request

API = "https://api.cellxgene.cziscience.com/curation/v1/datasets"
COLUMNS = [
    "year", "dataset_version_id", "dataset_id", "collection_id", "cell_count",
    "size_mb", "organism", "assay", "url", "title",
]


def snapshot(out: str) -> None:
    with urllib.request.urlopen(API, timeout=120) as r:
        body = r.read()
    datasets = json.loads(body)
    doc = {
        "source": API,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(body).hexdigest(),
        "datasets": datasets,
    }
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"{len(datasets)} datasets, sha256 {doc['sha256'][:12]}", file=sys.stderr)


def h5ad(d: dict) -> dict | None:
    return next((a for a in d["assets"] if a["filetype"] == "H5AD"), None)


def labels(d: dict, key: str) -> str:
    return ";".join(sorted({x["label"] for x in d.get(key) or []}))


def matches(d: dict, a: argparse.Namespace) -> bool:
    asset = h5ad(d)
    if d.get("tombstone") or asset is None:
        return False
    if a.max_mb is not None and asset["filesize"] > a.max_mb * 1e6:
        return False
    if a.min_cells is not None and d["cell_count"] < a.min_cells:
        return False
    if a.max_cells is not None and d["cell_count"] > a.max_cells:
        return False
    if a.organism and a.organism not in labels(d, "organism").split(";"):
        return False
    if a.assay and a.assay.lower() not in labels(d, "assay").lower():
        return False
    if a.year and d["published_at"][:4] != str(a.year):
        return False
    if a.title and not re.search(a.title, d["title"], re.IGNORECASE):
        return False
    return True


def rank(seed: int, d: dict) -> str:
    return hashlib.sha256(f"{seed}:{d['dataset_version_id']}".encode()).hexdigest()


def choose(datasets: list, a: argparse.Namespace) -> list:
    by_year: dict[str, list] = {}
    for d in filter(lambda d: matches(d, a), datasets):
        by_year.setdefault(d["published_at"][:4], []).append(d)
    chosen = []
    for year in sorted(by_year, reverse=True):
        take = sorted(by_year[year], key=lambda d: rank(a.seed, d))[: a.per_year]
        chosen += take[: a.n - len(chosen)]
        if len(chosen) == a.n:
            break
    return chosen


def sample(a: argparse.Namespace) -> None:
    with open(a.catalog) as f:
        doc = json.load(f)
    chosen = choose(doc["datasets"], a)
    query = {k: v for k, v in vars(a).items() if k not in ("cmd", "catalog", "func")}
    out = sys.stdout
    out.write(f"# catalog {doc['source']} fetched {doc['fetched_at']} sha256 {doc['sha256']}\n")
    out.write(f"# query {json.dumps(query, sort_keys=True)}\n")
    out.write("\t".join(COLUMNS) + "\n")
    for d in chosen:
        asset = h5ad(d)
        row = [
            d["published_at"][:4], d["dataset_version_id"], d["dataset_id"],
            d["collection_id"], d["cell_count"], round(asset["filesize"] / 1e6, 1),
            labels(d, "organism"), labels(d, "assay"), asset["url"],
            d["title"].replace("\t", " "),
        ]
        out.write("\t".join(map(str, row)) + "\n")
    if len(chosen) < a.n:
        print(f"only {len(chosen)} datasets match (asked for {a.n})", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="save the catalogue")
    s.add_argument("out")
    s.set_defaults(func=lambda a: snapshot(a.out))
    q = sub.add_parser("sample", help="draw a corpus from a saved catalogue")
    q.add_argument("catalog")
    q.add_argument("--seed", type=int, required=True)
    q.add_argument("--n", type=int, required=True, help="datasets in total")
    q.add_argument("--per-year", type=int, required=True, help="datasets per year, newest year first")
    q.add_argument("--organism", help='exact label, e.g. "Homo sapiens"')
    q.add_argument("--assay", help="case-insensitive substring of an assay label")
    q.add_argument("--max-mb", type=float, help="largest h5ad, in MB")
    q.add_argument("--min-cells", type=int)
    q.add_argument("--max-cells", type=int)
    q.add_argument("--title", help="regex on the dataset title")
    q.add_argument("--year", type=int, help="only datasets published that year")
    q.set_defaults(func=sample)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
