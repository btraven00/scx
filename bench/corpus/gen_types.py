#!/usr/bin/env python3
"""A few h5ad files from anndata's own test generator (anndata.tests.helpers.gen_adata),
as a corpus for run_corpus.py beside gen_types.R: obs/var in every dtype anndata
tests (categoricals, uint8, nullable bool/int), obsm/varm/layers/obsp/varp as
dense, sparse and DataFrame, uns with recarrays, nested dicts and awkward arrays.

    uv run --no-project --with anndata==0.13.4 --with pytest --with dask \\
        --with awkward --with zarr python gen_types.py out_dir

Writes out_dir/<id>/<id>.h5ad and out_dir/types_anndata.tsv (cxg_sample.py layout).
"""
import sys
from pathlib import Path

from importlib.metadata import version
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.tests.helpers import gen_adata

out = Path(sys.argv[1])
plain = (sp.csr_matrix, np.ndarray, pd.DataFrame)
cases = {
    "anndata-default": {},
    "anndata-dense-X": {"X_type": np.asarray},
    "anndata-csc": {"sparse_fmt": "csc"},
    # Without awkward / xarray in obsm and varm, so they don't hide the rest.
    "anndata-no-awkward": {"obsm_types": plain, "varm_types": plain},
}
rows = []
for i, (cid, kw) in enumerate(cases.items()):
    path = out / cid / f"{cid}.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    a = gen_adata((20, 30), random_state=np.random.default_rng(i), **kw)
    a.write_h5ad(path)
    mb = round(path.stat().st_size / 1e6, 3)
    rows.append(["gen", cid, "-", "-", 20, mb, "-", "generated", "-", cid])
with open(out / "types_anndata.tsv", "w") as f:
    f.write(f"# anndata {version('anndata')} gen_adata\n")
    f.write("\t".join(["year", "dataset_version_id", "dataset_id", "collection_id", "cell_count",
                       "size_mb", "organism", "assay", "url", "title"]) + "\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "written")
