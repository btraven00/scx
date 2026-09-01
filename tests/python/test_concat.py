"""Oracle test: `scx concat` must match `anndata.concat()`.

Self-contained — builds its own two-sample fixtures, so it needs no golden
files, only the scx binary and anndata.
"""

import os
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

SCX = os.environ.get("SCX_BIN") or shutil.which("scx") or "target/release/scx"

pytestmark = pytest.mark.skipif(
    not os.path.exists(SCX) and shutil.which(SCX) is None,
    reason=f"scx binary not found at {SCX} — run `pixi run -e test install-scx`",
)


def _adata(cells, genes, rows, batch):
    return ad.AnnData(
        X=sparse.csr_matrix(np.array(rows, dtype=np.float32)),
        obs=pd.DataFrame({"batch": batch, "n": np.arange(len(cells))}, index=cells),
        var=pd.DataFrame(index=genes),
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("concat")
    # Deliberately mismatched: b reorders genes, drops g0, adds g3.
    a = _adata(["c0", "c1"], ["g0", "g1", "g2"], [[1, 2, 0], [0, 0, 3]], "a")
    b = _adata(["c0"], ["g2", "g1", "g3"], [[4, 0, 5]], "b")
    # b has an obs column a lacks -> exercises the NA-fill path.
    b.obs["extra"] = [7.0]
    pa, pb = d / "a.h5ad", d / "b.h5ad"
    a.write_h5ad(pa)
    b.write_h5ad(pb)
    return a, b, str(pa), str(pb), d


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_matches_anndata_concat(inputs, join):
    a, b, pa, pb, d = inputs
    out = str(d / f"scx_{join}.h5ad")
    subprocess.run(
        [SCX, "concat", pa, pb, "-o", out,
         "--join", join, "--label", "sample", "--index-unique", "-"],
        check=True, capture_output=True,
    )
    got = ad.read_h5ad(out)
    want = ad.concat(
        {"a": a, "b": b}, join=join, label="sample", index_unique="-",
    )

    assert list(got.var_names) == list(want.var_names)
    assert list(got.obs_names) == list(want.obs_names)
    np.testing.assert_allclose(
        got.X.toarray() if sparse.issparse(got.X) else got.X,
        want.X.toarray() if sparse.issparse(want.X) else want.X,
    )
    assert list(got.obs["sample"]) == list(want.obs["sample"])
    for col in want.obs.columns:
        assert col in got.obs, f"obs column '{col}' missing from scx output"
        if want.obs[col].dtype.kind == "f":
            np.testing.assert_allclose(
                got.obs[col].to_numpy(float), want.obs[col].to_numpy(float)
            )
        else:
            assert list(got.obs[col].astype(str)) == list(want.obs[col].astype(str))
