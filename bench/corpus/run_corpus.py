#!/usr/bin/env python3
"""Convert every h5ad of a corpus (cxg_sample.py TSV) to each scx output
format and check the result with that format's own reader.

The checks are omni-scformats' (github.com/btraven00/omni-scformats,
modules/): snapshot.py reads h5ad with anndata, snapshot.R reads .h5seurat
with rhdf5 + BPCells, and compare.py compares input and output snapshots
on names, every obs/var column (type, values, NA mask) and every matrix.
Zarr outputs are read with anndata.read_zarr into the same layout. scx never
judges its own output; `scx diff` is recorded beside the verdict.

Arms: h5ad, zarr (v3), h5seurat (BPCells, the default), h5seurat --dgcmatrix.

    run_corpus.py corpus.tsv data_dir out_dir --checks <omni-scformats>/modules [--scx BIN]

Run with a python that has anndata, and Rscript with rhdf5 + BPCells on PATH
(omni-scformats envs/readers.yml). Writes out_dir/results.jsonl, one record
per dataset and arm; a dataset already in it is skipped.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ARMS = {
    "h5ad": ("out.h5ad", []),
    "zarr": ("out.zarr", []),
    "h5seurat": ("out.h5seurat", []),
    "h5seurat_dgc": ("out.h5seurat", ["--dgcmatrix"]),
}


def run(cmd, timeout=3600):
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout} s", round(time.time() - t, 2)
    return r.returncode, (r.stdout + r.stderr)[-3000:], round(time.time() - t, 2)


def snap(path, out, checks):
    """Snapshot `path` with its format's reader; -> error text or None."""
    if path.suffix == ".h5seurat":
        cmd = ["Rscript", "--vanilla", str(checks / "snapshot.R"), str(path), str(out)]
    elif path.suffix == ".zarr":
        code = ("import sys, anndata; sys.path.insert(0, sys.argv[1]); import snapshot; "
                "snapshot.write(anndata.read_zarr(sys.argv[2]), sys.argv[3])")
        cmd = [sys.executable, "-c", code, str(checks), str(path), str(out)]
    else:
        cmd = [sys.executable, str(checks / "snapshot.py"), str(path), str(out)]
    rc, log, _ = run(cmd)
    return None if rc == 0 else log


def check_arm(src, arm, work, scx, checks, compare, src_snap):
    name, flags = ARMS[arm]
    out = work / arm / name
    out.parent.mkdir(parents=True)
    rec = {"arm": arm}
    rc, log, secs = run([scx, "convert", str(src), str(out), *flags])
    rec["convert_s"] = secs
    if rc != 0:
        return {**rec, "status": "convert_failed", "error": log}
    with tempfile.TemporaryDirectory(dir=work) as t:
        err = snap(out, Path(t), checks)
        if err:
            return {**rec, "status": "reader_failed", "error": err}
        props = compare.compare(compare.load(src_snap), compare.load(t))
    fails = [p for p in props if p["status"] == "fail"]
    rec.update(status="invalid" if fails else "valid", failed=fails,
               notes=[p for p in props if p["status"] == "note"])
    rc, log, _ = run([scx, "diff", "--json", str(src), str(out)])
    rec["scx_diff_exit"] = rc
    if rc in (0, 1):
        rec["scx_diff_agrees"] = (rc == 0) == (rec["status"] == "valid")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corpus")
    p.add_argument("data", type=Path, help="dir with <dataset_version_id>/<file>.h5ad")
    p.add_argument("out", type=Path)
    p.add_argument("--checks", type=Path, required=True, help="omni-scformats/modules")
    p.add_argument("--scx", default=shutil.which("scx"))
    p.add_argument("--arms", default=",".join(ARMS))
    a = p.parse_args()
    sys.path.insert(0, str(a.checks))
    import compare

    a.out.mkdir(parents=True, exist_ok=True)
    results = a.out / "results.jsonl"
    done = {json.loads(l)["dataset_version_id"] for l in results.open()} if results.exists() else set()
    rows = [l.rstrip("\n").split("\t") for l in open(a.corpus) if not l.startswith("#")]
    head, rows = rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]
    for row in rows:
        vid = row["dataset_version_id"]
        if vid in done:
            continue
        src = next((a.data / vid).glob("*.h5ad"), None)
        base = {k: row[k] for k in ("dataset_version_id", "year", "cell_count", "size_mb", "assay", "title")}
        if src is None:
            recs = [{**base, "arm": "-", "status": "missing_input"}]
        else:
            with tempfile.TemporaryDirectory(dir=a.out) as w:
                w = Path(w)
                err = snap(src, w / "in", a.checks)
                if err:
                    recs = [{**base, "arm": "-", "status": "input_reader_failed", "error": err}]
                else:
                    recs = [{**base, **check_arm(src, arm, w, a.scx, a.checks, compare, w / "in")}
                            for arm in a.arms.split(",")]
        with results.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
                print(vid[:8], r["arm"], r["status"],
                      [x["property"] for x in r.get("failed", [])][:5], flush=True)


if __name__ == "__main__":
    main()
