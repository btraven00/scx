"""Split read-cost vs reduce-cost for open_stream / anndata-backed / h5py-raw.

Goal: localise where open_stream spends time vs anndata backed on h5ad — is the
gap (a) the read/decode path or (b) the Python reduction? Process a fixed number
of chunks per variant so throughput is comparable.

  pixi run -e py313 python bench/python/profile_read_split.py <h5ad> --chunk-size 50000 --n-chunks 6

This is how the "debug build is ~3x slower" finding was made: in a debug
`maturin develop` build the open_stream read path runs ~3x slower than anndata;
with `--release` (install-picklerick-py-native-release) it is at parity, and the
only residual gap is the per-gene-sum reduction. ALWAYS build --release before
trusting these numbers.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def t(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label:<34}{dt:8.2f}s   {out}")
    return dt


def os_drain(path, cs, nmax):
    import picklerick as pk

    n = nnz = 0
    for i, ch in enumerate(pk.open_stream(path, chunk_size=cs)):
        nnz += ch.data.shape[0]  # force materialization of the bytes/array
        n += ch.nrows
        if i + 1 >= nmax:
            break
    return f"rows={n} nnz={nnz}"


def os_full(path, cs, nmax):
    import picklerick as pk

    gs = None
    n = 0
    for i, ch in enumerate(pk.open_stream(path, chunk_size=cs)):
        if gs is None:
            gs = np.zeros(ch.n_vars, dtype=np.float64)
        gs += np.bincount(ch.indices, weights=ch.data.astype(np.float64), minlength=ch.n_vars)
        n += ch.nrows
        if i + 1 >= nmax:
            break
    return f"rows={n} sum={float(gs.sum()):.0f}"


def bk_drain(path, cs, nmax):
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    n = nnz = 0
    for i, (ch, s, e) in enumerate(a.chunked_X(cs)):
        nnz += ch.nnz if hasattr(ch, "nnz") else ch.size
        n += e - s
        if i + 1 >= nmax:
            break
    return f"rows={n} nnz={nnz}"


def bk_full(path, cs, nmax):
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    gs = np.zeros(a.shape[1], dtype=np.float64)
    n = 0
    for i, (ch, s, e) in enumerate(a.chunked_X(cs)):
        gs += np.asarray(ch.sum(axis=0), dtype=np.float64).ravel()
        n += e - s
        if i + 1 >= nmax:
            break
    return f"rows={n} sum={float(gs.sum()):.0f}"


def h5py_full(path, cs, nmax):
    """Exact open_stream workload (slice + bincount) but via h5py reads."""
    import h5py

    with h5py.File(path, "r") as f:
        x = f["X"]
        indptr = x["indptr"][:].astype(np.int64)
        n_vars = int(x.attrs["shape"][1]) if "shape" in x.attrs else None
        data_ds, idx_ds = x["data"], x["indices"]
        n_obs = indptr.shape[0] - 1
        if n_vars is None:
            n_vars = int(idx_ds[:].max()) + 1
        gs = np.zeros(n_vars, dtype=np.float64)
        n = 0
        for i, rs in enumerate(range(0, n_obs, cs)):
            re = min(rs + cs, n_obs)
            a, b = int(indptr[rs]), int(indptr[re])
            data = data_ds[a:b]
            idx = idx_ds[a:b]
            gs += np.bincount(idx, weights=data.astype(np.float64), minlength=n_vars)
            n += re - rs
            if i + 1 >= nmax:
                break
    return f"rows={n} sum={float(gs.sum()):.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--n-chunks", type=int, default=6)
    args = ap.parse_args()
    cs, nm = args.chunk_size, args.n_chunks
    print(f"# {args.path}  cs={cs}  n_chunks={nm}\n")

    a = t("open_stream  drain (read only)", lambda: os_drain(args.path, cs, nm))
    b = t("open_stream  full  (+bincount)", lambda: os_full(args.path, cs, nm))
    c = t("anndata bkd  drain (read only)", lambda: bk_drain(args.path, cs, nm))
    d = t("anndata bkd  full  (+sum)", lambda: bk_full(args.path, cs, nm))
    e = t("h5py raw     full  (+bincount)", lambda: h5py_full(args.path, cs, nm))

    print("\n-- derived --")
    print(f"open_stream reduce cost (full-drain): {b - a:.2f}s")
    print(f"anndata     reduce cost (full-drain): {d - c:.2f}s")
    print(f"open_stream read   vs anndata read  : {a:.2f}s vs {c:.2f}s  ({a / c:.1f}x)")
    print(f"open_stream full   vs h5py-raw full : {b:.2f}s vs {e:.2f}s  ({b / e:.1f}x)")


if __name__ == "__main__":
    main()
