#!/usr/bin/env python3
"""Write the AnnData Zarr test fixtures: one small AnnData, three encodings.

    tests/fixtures/zarr/small.h5ad       the oracle (read by H5AdReader)
    tests/fixtures/zarr/small_v2.zarr    anndata's Zarr v2 layout (Blosc/lz4 default)
    tests/fixtures/zarr/small_v3.zarr    anndata's Zarr v3 layout (zstd default)
    tests/fixtures/zarr/small_unsorted.h5ad / small_unsorted_v3.zarr
                                         the same with unsorted CSR column indices
    tests/fixtures/zarr/varied{.h5ad,_v2.zarr,_v3.zarr}
                                         synthetic: the encodings and dtypes
                                         `small` lacks (dense int X, CSC layer and
                                         obsp, dense layer, narrow and unsigned
                                         int columns, non-string categories,
                                         int and sparse obsm, dense obsp, every
                                         uns value type)
    tests/fixtures/zarr/nox_one{.h5ad,_v3.zarr}, nox_many{.h5ad,_v3.zarr}
                                         no X: one layer, or several without
                                         counts/X (the reader's fallback rules)
    tests/fixtures/zarr/csc_x_v3.zarr     X stored as CSC, which the reader refuses

The Zarr reader's tests assert it returns exactly what H5AdReader returns for
the h5ad written from the same object, so the fixture exercises every
encoding scx reads: CSR X, a CSR layer, categorical / float / int / bool /
string obs columns, pandas nullable int, bool and string columns, obsm, varm,
obsp and uns.

Built from the pbmc3k golden file so values are realistic. Deterministic.

usage: python scripts/prepare_zarr_fixtures.py [path/to/pbmc3k_reference.h5ad]
       python scripts/prepare_zarr_fixtures.py --synthetic-only   (no golden file needed)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/fixtures/zarr"
OUT.mkdir(parents=True, exist_ok=True)


def write_small(src):
    full = ad.read_h5ad(src)
    a = full[:60, :40].copy()
    X = sp.csr_matrix(a.X, dtype=np.float32)
    X.sort_indices()

    rng = np.random.default_rng(0)
    n = a.n_obs
    obs = pd.DataFrame(index=a.obs_names.astype(str))
    obs["cluster"] = pd.Categorical(rng.choice(["B", "NK", "T"], n))
    obs["score"] = rng.normal(size=n)
    obs["n_genes"] = rng.integers(0, 500, n).astype(np.int32)
    obs["is_doublet"] = rng.random(n) < 0.1
    obs["donor"] = np.array([f"d{i % 3}" for i in range(n)], dtype=object)
    obs["nullable_int"] = pd.array([None if i % 7 == 0 else i for i in range(n)], dtype="Int32")
    obs["nullable_bool"] = pd.array([None if i % 5 == 0 else bool(i % 2) for i in range(n)], dtype="boolean")
    obs["nullable_str"] = pd.array([None if i % 4 == 0 else f"s{i}" for i in range(n)], dtype="string")

    var = pd.DataFrame(index=a.var_names.astype(str))
    var["gene_id"] = np.array([f"ENSG{i:05d}" for i in range(a.n_vars)], dtype=object)
    var["highly_variable"] = rng.random(a.n_vars) < 0.3

    b = ad.AnnData(X=X, obs=obs, var=var)
    b.layers["counts"] = sp.csr_matrix(np.round(X.toarray() * 3), dtype=np.float32)
    b.obsm["X_pca"] = rng.normal(size=(n, 5))
    b.varm["PCs"] = rng.normal(size=(a.n_vars, 5))
    nn = sp.random(n, n, density=0.05, format="csr", random_state=0, dtype=np.float32)
    nn.sort_indices()
    b.obsp["connectivities"] = nn
    b.uns["title"] = "scx zarr fixture"
    b.uns["params"] = {"n_pcs": 5, "resolution": 0.8}

    b.write_h5ad(OUT / "small.h5ad")
    for fmt in (2, 3):
        ad.settings.zarr_write_format = fmt
        b.write_zarr(OUT / f"small_v{fmt}.zarr")


    def unsorted(m):
        """Same matrix, column indices reversed within each row (valid CSR,
        has_sorted_indices False), as some writers leave them; the 10x ladder
        files are like this."""
        m = m.copy()
        for r in range(m.shape[0]):
            a, b_ = m.indptr[r], m.indptr[r + 1]
            m.indices[a:b_] = m.indices[a:b_][::-1].copy()
            m.data[a:b_] = m.data[a:b_][::-1].copy()
        m.has_sorted_indices = False
        return m


    u = b.copy()
    u.X = unsorted(b.X)
    u.layers["counts"] = unsorted(b.layers["counts"])
    u.write_h5ad(OUT / "small_unsorted.h5ad")
    ad.settings.zarr_write_format = 3
    u.write_zarr(OUT / "small_unsorted_v3.zarr")
    print(f"wrote {b.n_obs} x {b.n_vars} fixtures to {OUT} (anndata {ad.__version__})")


def write_varied():
    """Synthetic, so it needs no golden file. Every value is small and exact."""
    rng = np.random.default_rng(1)
    n, m = 30, 12
    X = rng.integers(0, 5, (n, m)).astype(np.int64)  # dense int X
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    for dt in ["int16", "int64", "uint8", "uint16", "uint32", "uint64", "float32"]:
        obs[dt] = rng.integers(0, 100, n).astype(dt)
    obs["cat_float"] = pd.Categorical(rng.choice([0.5, 1.5, 2.5], n))
    obs["cat_int"] = pd.Categorical(rng.choice([10, 20, 30], n))
    obs["cat_bool"] = pd.Categorical(rng.choice([True, False], n))
    obs["cat_ordered"] = pd.Categorical(rng.choice(["lo", "mid", "hi"], n), categories=["lo", "mid", "hi"], ordered=True)
    var = pd.DataFrame(index=[f"g{j}" for j in range(m)])
    var["n_cells"] = rng.integers(0, n, m).astype(np.uint32)

    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["csc"] = sp.csc_matrix(rng.random((n, m)) * (rng.random((n, m)) < 0.3))
    a.layers["dense_u16"] = rng.integers(0, 9, (n, m)).astype(np.uint16)
    a.obsm["int32"] = rng.integers(-5, 5, (n, 3)).astype(np.int32)
    a.obsm["int64"] = rng.integers(-5, 5, (n, 2)).astype(np.int64)
    a.obsm["float32"] = rng.random((n, 2)).astype(np.float32)
    a.varm["float32"] = rng.random((m, 2)).astype(np.float32)
    a.obsp["dist_csc"] = sp.csc_matrix(sp.random(n, n, density=0.1, random_state=1, dtype=np.float64))
    # Sparse obsm (read dense) and dense obsp; drawn last so earlier values stay put.
    a.obsm["sparse_csr"] = sp.csr_matrix(rng.random((n, 4)) * (rng.random((n, 4)) < 0.3))
    a.obsm["sparse_csc"] = sp.csc_matrix(rng.random((n, 3)) * (rng.random((n, 3)) < 0.3))
    a.obsp["dense"] = rng.random((n, n)) * (rng.random((n, n)) < 0.2)
    a.uns["flag"] = True
    a.uns["ratio"] = 0.25
    a.uns["count"] = 7
    a.uns["label"] = "varied"
    a.uns["ints"] = np.array([1, 2, 3])
    a.uns["floats"] = np.array([0.5, 1.5])
    a.uns["strings"] = np.array(["a", "b"], dtype=object)
    a.uns["bools"] = np.array([True, False])
    a.uns["grid"] = np.arange(6).reshape(2, 3)
    a.uns["nested"] = {"k": 2, "deeper": {"name": "x"}}
    # NaN (JSON has none) and matrices (kept 2-D, not flattened).
    a.uns["nan_scalar"] = np.nan
    a.uns["nan_vec"] = np.array([np.nan, 0.5, np.nan])
    a.uns["nan_mat"] = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
    write_three(a, "varied")

    nox = ad.AnnData(obs=obs[[]], var=var[[]])
    nox.layers["only"] = sp.csr_matrix(X.astype(np.float32))
    write_three(nox, "nox_one", v2=False)
    nox.layers["other"] = sp.csr_matrix(X.astype(np.float32) * 2)
    write_three(nox, "nox_many", v2=False)

    c = ad.AnnData(X=sp.csc_matrix(X.astype(np.float32)), obs=obs[[]], var=var[[]])
    ad.settings.zarr_write_format = 3
    c.write_zarr(OUT / "csc_x_v3.zarr")


def write_three(a, name, v2=True):
    a.write_h5ad(OUT / f"{name}.h5ad")
    for fmt in (2, 3) if v2 else (3,):
        ad.settings.zarr_write_format = fmt
        a.write_zarr(OUT / f"{name}_v{fmt}.zarr")


if __name__ == "__main__":
    synthetic_only = sys.argv[1:] == ["--synthetic-only"]
    if not synthetic_only:
        write_small(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/golden/pbmc3k_reference.h5ad")
    write_varied()
