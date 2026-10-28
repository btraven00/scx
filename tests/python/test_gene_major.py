"""Oracle test: `scx convert --gene-major` must write the X anndata reads as CSC.

Self-contained — builds its own fixture, so it needs no golden files, only
the scx binary and anndata.
"""

import os
import shutil
import subprocess

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

SCX = os.environ.get("SCX_BIN") or shutil.which("scx") or "target/release/scx"

pytestmark = pytest.mark.skipif(
    not os.path.exists(SCX) and shutil.which(SCX) is None,
    reason=f"scx binary not found at {SCX} — run `pixi run -e test install-scx`",
)


# A budget of 1e-7 GB (100 bytes, ~8 nonzeros) forces several passes.
@pytest.mark.parametrize("budget", ["1e-7", "4"])
def test_gene_major_matches_anndata(tmp_path, budget):
    x = sparse.random(60, 40, density=0.2, format="csr", dtype=np.float32, random_state=0)
    a = ad.AnnData(X=x)
    a.obs_names = [f"c{i}" for i in range(60)]
    a.var_names = [f"g{j}" for j in range(40)]
    src, out = tmp_path / "in.h5ad", tmp_path / "out.h5ad"
    a.write_h5ad(src)
    subprocess.run([SCX, "convert", str(src), str(out), "--gene-major", "--transpose-mem", budget],
                   check=True, capture_output=True)

    got = ad.read_h5ad(out)
    assert sparse.isspmatrix_csc(got.X) or isinstance(got.X, sparse.csc_array)
    assert got.X.has_sorted_indices
    assert (got.X != a.X).nnz == 0
    assert list(got.obs_names) == list(a.obs_names)
    assert list(got.var_names) == list(a.var_names)

    # The point of CSC: a backed gene slice reads only those genes.
    backed = ad.read_h5ad(out, backed="r")
    genes = [0, 7, 39]
    assert np.array_equal(backed.X[:, genes].toarray(), a.X[:, genes].toarray())
