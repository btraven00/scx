"""Single-scenario runner for the stream-vs-load memory benchmark.

Runs ONE scenario in its own process so the reported peak RSS (VmHWM from
/proc/self/status) reflects only that scenario. Both modes compute the same
per-gene sum over the count matrix and emit a checksum, so the driver can
confirm the streaming path is numerically equivalent to the full load.

Emits a single JSON line on stdout.
"""

from __future__ import annotations

import argparse
import json
import time


def _peak_rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                kb = int(line.split()[1])
                return kb / 1024.0
    return float("nan")


def run_load(path: str) -> tuple[float, int, int]:
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(path)
    n_obs, n_vars = adata.shape
    gene_sums = np.asarray(adata.X.sum(axis=0), dtype=np.float64).ravel()
    return float(gene_sums.sum()), n_obs, n_vars


def run_backed(path: str, chunk_size: int) -> tuple[float, int, int]:
    """anndata's own bounded-memory path: backed read + chunked_X.

    This is the apples-to-apples baseline for open_stream on h5ad — both read
    the matrix in row blocks off disk without materialising it. (anndata
    backed mode only supports h5ad/zarr; the non-h5ad formats open_stream
    handles have no anndata equivalent, which is the point of the comparison.)
    """
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(path, backed="r")
    n_obs, n_vars = adata.shape
    gene_sums = np.zeros(n_vars, dtype=np.float64)
    for chunk, _start, _end in adata.chunked_X(chunk_size):
        gene_sums += np.asarray(chunk.sum(axis=0), dtype=np.float64).ravel()
    return float(gene_sums.sum()), n_obs, n_vars


def run_stream(path: str, chunk_size: int) -> tuple[float, int, int]:
    import numpy as np

    import picklerick as pk

    gene_sums: np.ndarray | None = None
    n_vars = 0
    n_obs = 0
    for chunk in pk.open_stream(path, chunk_size=chunk_size):
        if gene_sums is None:
            n_vars = chunk.n_vars
            gene_sums = np.zeros(n_vars, dtype=np.float64)
        # bincount is the fast scatter-add (unlike np.add.at, the unbuffered
        # slow path): per-column weighted counts == per-gene sums. Keeping the
        # reduction cheap means wall_s reflects decode throughput, not the
        # reduction, so it is comparable to the read-dominated load path.
        gene_sums += np.bincount(
            chunk.indices, weights=chunk.data.astype(np.float64), minlength=n_vars
        )
        n_obs += chunk.nrows
    total = float(gene_sums.sum()) if gene_sums is not None else 0.0
    return total, n_obs, n_vars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["load", "stream", "backed"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--chunk-size", type=int, default=5000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.mode == "load":
        checksum, n_obs, n_vars = run_load(args.path)
    elif args.mode == "backed":
        checksum, n_obs, n_vars = run_backed(args.path, args.chunk_size)
    else:
        checksum, n_obs, n_vars = run_stream(args.path, args.chunk_size)
    wall_s = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "mode": args.mode,
                "path": args.path,
                "chunk_size": args.chunk_size,
                "n_obs": n_obs,
                "n_vars": n_vars,
                "wall_s": round(wall_s, 4),
                "peak_rss_mb": round(_peak_rss_mb(), 1),
                "checksum": checksum,
            }
        )
    )


if __name__ == "__main__":
    main()
