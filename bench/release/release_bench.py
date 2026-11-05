#!/usr/bin/env python3
"""Pre-release perf record for scx. Standard library only.

  run      fetch the reference (hapiq, sha256-pinned), time `scx convert` end to
           end, run the criterion `perf` bench, write bench/results/<label>.json
  compare  diff two result files; flags changes beyond --threshold

Run on omni before tagging (docs/releasing.md). `--scx` can point at an older
release's binary to backfill history; add `--no-criterion` then, since the
criterion suite only exists from this commit on.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))

# HLCA core, CELLxGENE dataset version 7bcad396: 584,944 cells, gzip-compressed X.
REF_URL = "https://datasets.cellxgene.cziscience.com/7bcad396-49c3-40d9-80c1-16d74e7b88bd.h5ad"
REF_SHA256 = "f07114e5f491aa304e2199665bff500871c47f185d48737efc77d4abf29b9c52"

# name -> (extra convert flags, output suffix)
TARGETS = {
    "bpcells": ([], ".h5seurat"),
    "h5seurat_dgc": (["--dgcmatrix"], ".h5seurat"),
    "h5ad": ([], ".h5ad"),
}
THREADS = "8"


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), file=sys.stderr)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch_reference(data):
    """The gzip reference via hapiq, plus an uncompressed copy (h5repack)."""
    gz = data / "hlca_core.gzip.h5ad"
    if not gz.exists():
        hapiq = os.environ.get("HAPIQ", "hapiq")
        dl = data / "hapiq"
        sh(hapiq, "download", "url", REF_URL, "--out", dl, "-y")
        [got] = glob.glob(str(dl / "**" / "*.h5ad"), recursive=True)
        if (h := sha256(got)) != REF_SHA256:
            sys.exit(f"reference sha256 mismatch: {h} != {REF_SHA256}")
        os.replace(got, gz)
    raw = data / "hlca_core.raw.h5ad"
    if not raw.exists():
        # HDF5's own tool: same values, every filter removed (~18 GB).
        sh("h5repack", "-f", "NONE", gz, f"{raw}.tmp")
        os.replace(f"{raw}.tmp", raw)
    return {"gzip": gz, "raw": raw}


def time_convert(scx, args, out):
    """(wall s, peak RSS MB) of one `scx convert`, via GNU time."""
    with tempfile.NamedTemporaryFile("r") as t:
        sh("/usr/bin/time", "-o", t.name, "-f", "%e %M", scx, "convert", *args,
           stdout=subprocess.DEVNULL)
        wall, kb = t.read().split()[-2:]
    for p in glob.glob(f"{out}*"):
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    return float(wall), int(kb) / 1024


def run_e2e(scx, inputs, work, reps):
    res = {}
    os.environ["RAYON_NUM_THREADS"] = THREADS
    for enc, src in inputs.items():
        sh("cat", src, stdout=subprocess.DEVNULL)  # warm the page cache
        for name, (flags, suffix) in TARGETS.items():
            out = work / f"out{suffix}"
            runs = [time_convert(scx, [*flags, src, out], out) for _ in range(reps)]
            key = f"e2e/{enc}/{name}"
            res[key] = {
                "wall_s": statistics.median(w for w, _ in runs),
                "peak_mb": statistics.median(m for _, m in runs),
                "runs": runs,
            }
            print(f"{key}: {res[key]['wall_s']:.2f} s, {res[key]['peak_mb']:.0f} MB", file=sys.stderr)
    return res


def run_criterion():
    sh("cargo", "bench", "-p", "scx-core", "--bench", "perf", cwd=REPO,
       env={**os.environ, "CARGO_TARGET_DIR": str(TARGET), "RAYON_NUM_THREADS": THREADS})
    res = {}
    for est in glob.glob(f"{TARGET}/criterion/**/new/estimates.json", recursive=True):
        bench = json.load(open(Path(est).parent / "benchmark.json"))
        if bench["group_id"] not in ("bp128", "write_x", "read_x"):
            continue
        e = json.load(open(est))
        res[f"criterion/{bench['full_id']}"] = {
            "median_ms": e["median"]["point_estimate"] / 1e6,
            "ci_ms": [e["median"]["confidence_interval"][k] / 1e6 for k in ("lower_bound", "upper_bound")],
        }
    return res


def machine():
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    load = os.getloadavg()[0]
    return {"host": platform.node(), "cpu": cpu, "nproc": os.cpu_count(),
            "kernel": platform.release(), "loadavg_1m_at_start": load}


def cmd_run(a):
    scx = Path(a.scx or TARGET / "release/scx").resolve()
    if not a.scx:
        sh("cargo", "build", "--release", "--bin", "scx", cwd=REPO,
           env={**os.environ, "CARGO_TARGET_DIR": str(TARGET)})
    version = subprocess.run([scx, "--version"], capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    m = machine()
    if m["loadavg_1m_at_start"] > a.max_load:
        sys.exit(f"load {m['loadavg_1m_at_start']:.1f} > --max-load {a.max_load}; numbers would be noise")

    data = Path(a.data).expanduser()
    data.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=a.work) as work:
        e2e = run_e2e(scx, fetch_reference(data), Path(work), a.reps)
    crit = {} if a.no_criterion else run_criterion()

    label = a.label or version.split()[1]
    out = REPO / "bench/results" / f"release-{label}.json"
    json.dump({"label": label, "scx_version": version, "git": sha,
               "reference": {"url": REF_URL, "sha256": REF_SHA256},
               "rayon_threads": int(THREADS), "machine": m, "elapsed_s": round(time.time() - t0),
               "results": {**e2e, **crit}}, open(out, "w"), indent=1)
    print(out)


def cmd_compare(a):
    old, new = (json.load(open(p)) for p in (a.old, a.new))
    if old["machine"]["cpu"] != new["machine"]["cpu"]:
        print(f"WARNING: different CPUs ({old['machine']['cpu']} vs {new['machine']['cpu']})")
    print(f"{'bench':45} {old['label']:>12} {new['label']:>12}   change")
    worse = False
    for k in sorted(old["results"].keys() | new["results"].keys()):
        o, n = old["results"].get(k), new["results"].get(k)
        if not (o and n):
            print(f"{k:45} {'-' if not o else '':>12} {'-' if not n else '':>12}   only in {'new' if n else 'old'}")
            continue
        for metric in ("wall_s", "peak_mb", "median_ms"):
            if metric in o:
                d = n[metric] / o[metric] - 1
                flag = ""
                if abs(d) > a.threshold:
                    flag = "  REGRESSION" if d > 0 else "  faster/leaner"
                    worse |= d > 0
                print(f"{k + ' ' + metric:45} {o[metric]:12.2f} {n[metric]:12.2f}   {d:+6.1%}{flag}")
    sys.exit(1 if worse else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--scx", help="scx binary (default: build this checkout --release)")
    r.add_argument("--data", default="~/.cache/scx-bench", help="where the reference lives (~24 GB)")
    r.add_argument("--work", default=None, help="scratch dir for outputs (default: system tmp)")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--label", help="results name (default: the scx version)")
    r.add_argument("--max-load", type=float, default=4.0, help="refuse to run above this 1-min load")
    r.add_argument("--no-criterion", action="store_true", help="end-to-end only (for old release binaries)")
    r.set_defaults(f=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    c.add_argument("--threshold", type=float, default=0.05, help="relative change to flag (default 5%%)")
    c.set_defaults(f=cmd_compare)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
