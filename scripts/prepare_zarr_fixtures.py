#!/usr/bin/env python3
"""Write the AnnData Zarr test fixtures: one small AnnData, three encodings.

    tests/fixtures/zarr/small.h5ad       the oracle (read by H5AdReader)
    tests/fixtures/zarr/small_v2.zarr    anndata's Zarr v2 layout (Blosc/lz4 default)
    tests/fixtures/zarr/small_v3.zarr    anndata's Zarr v3 layout (zstd default)

The Zarr reader's tests assert it returns exactly what H5AdReader returns for
the h5ad written from the same object, so the fixture exercises every
encoding scx reads: CSR X, a CSR layer, categorical / float / int / bool /
string obs columns, pandas nullable int, bool and string columns, obsm, varm,
obsp and uns.

Built from the pbmc3k golden file so values are realistic. Deterministic.

usage: python scripts/prepare_zarr_fixtures.py [path/to/pbmc3k_reference.h5ad]
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/golden/pbmc3k_reference.h5ad"
out = ROOT / "tests/fixtures/zarr"
out.mkdir(parents=True, exist_ok=True)

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

b.write_h5ad(out / "small.h5ad")
for fmt in (2, 3):
    ad.settings.zarr_write_format = fmt
    b.write_zarr(out / f"small_v{fmt}.zarr")
print(f"wrote {b.n_obs} x {b.n_vars} fixtures to {out} (anndata {ad.__version__})")
