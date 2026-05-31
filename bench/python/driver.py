"""Driver for the open_stream-vs-read_h5ad memory/throughput benchmark.

Closes the bench gap noted in roadmap 0.1.5: the bounded-memory claim for
`pk.open_stream` was asserted but unmeasured. For each dataset this runs the
full-load path (`anndata.read_h5ad`) and the streaming path, each in its OWN
subprocess (runner.py) so the reported peak RSS (VmHWM) reflects only that one
scenario — peaks do not bleed across modes within a shared interpreter.

Both modes compute the same per-gene sum and emit a checksum; the driver fails
loudly if the streaming result diverges from the full-load oracle beyond rtol,
so a memory win can never be bought with a wrong answer.

Results are written to bench/results/<label>.json in the same envelope as the R
harness (label / git_sha / branch / timestamp / versions / results{}).

Run:
    pixi run -e py313 python bench/python/driver.py --label py-stream-v1
    pixi run -e py313 python bench/python/driver.py --include-large   # + hlca 5.7G
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNNER = HERE / "runner.py"
GOLDEN = REPO_ROOT / "tests" / "golden"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

# Short name -> h5ad path. `large` is opt-in (--include-large) because the
# full-load path materialises the whole matrix in RAM.
DATASETS: dict[str, Path] = {
    "pbmc3k": GOLDEN / "pbmc3k_reference.h5ad",
    "norman": GOLDEN / "norman_subset.h5ad",
}
LARGE_DATASETS: dict[str, Path] = {
    "hlca": GOLDEN / "hlca_core.h5ad",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0]}
    for pkg in ("anndata", "numpy", "picklerick"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


def _run(mode: str, path: Path, chunk_size: int) -> dict:
    """Run one scenario in a fresh subprocess; return its parsed JSON line."""
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--path",
        str(path),
        "--chunk-size",
        str(chunk_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed ({mode} {path.name}): rc={proc.returncode}\n{proc.stderr}"
        )
    # runner emits exactly one JSON line on stdout; tolerate stray warning lines.
    line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{"))
    return json.loads(line)


def _checksums_agree(load: float, stream: float, rtol: float) -> bool:
    return math.isclose(load, stream, rel_tol=rtol, abs_tol=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="py-stream", help="output label / filename stem")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        help="dataset short names (default: pbmc3k norman)",
    )
    ap.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="*",
        default=[5000],
        help="stream chunk sizes to sweep (default: 5000)",
    )
    ap.add_argument(
        "--include-large",
        action="store_true",
        help="also run the multi-GB hlca dataset (heavy full-load RSS)",
    )
    ap.add_argument("--rtol", type=float, default=1e-4, help="checksum relative tolerance")
    ap.add_argument(
        "--out", type=Path, default=None, help="output path (default: bench/results/<label>.json)"
    )
    args = ap.parse_args()

    registry = dict(DATASETS)
    if args.include_large:
        registry.update(LARGE_DATASETS)

    results: dict[str, dict] = {}
    failures: list[str] = []

    for name in args.datasets:
        path = registry.get(name) or Path(name)
        if not path.exists():
            print(f"!! skip {name}: not found ({path})", file=sys.stderr)
            continue

        print(f"== {name} ({path.name}) ==", file=sys.stderr)
        print("  load ...", file=sys.stderr, flush=True)
        load = _run("load", path, chunk_size=0)
        results[f"{name}/load"] = load

        for cs in args.chunk_sizes:
            print(f"  stream cs={cs} ...", file=sys.stderr, flush=True)
            stream = _run("stream", path, chunk_size=cs)
            ok = _checksums_agree(load["checksum"], stream["checksum"], args.rtol)
            stream["checksum_ok"] = ok
            results[f"{name}/stream-cs{cs}"] = stream
            if not ok:
                failures.append(
                    f"{name} cs={cs}: load={load['checksum']!r} stream={stream['checksum']!r}"
                )

    envelope = {
        "label": args.label,
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "versions": _versions(),
        "results": results,
    }

    out = args.out or (RESULTS_DIR / f"{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, indent=2) + "\n")

    _print_table(results)
    print(f"\nwrote {out}", file=sys.stderr)

    if failures:
        print("\nCHECKSUM MISMATCHES:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)


def _print_table(results: dict[str, dict]) -> None:
    hdr = f"{'scenario':<22}{'n_obs':>9}{'wall_s':>9}{'peak_MB':>10}{'cs':>7}{'ok':>4}"
    print(hdr)
    print("-" * len(hdr))
    for key, r in results.items():
        ok = "" if r["mode"] == "load" else ("ok" if r.get("checksum_ok") else "MISS")
        cs = "" if r["mode"] == "load" else str(r["chunk_size"])
        print(
            f"{key:<22}{r['n_obs']:>9}{r['wall_s']:>9.3f}"
            f"{r['peak_rss_mb']:>10.1f}{cs:>7}{ok:>4}"
        )


if __name__ == "__main__":
    main()
