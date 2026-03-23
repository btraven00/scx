import os
import pytest
import anndata as ad

GOLDEN = os.environ.get("SCX_GOLDEN", "tests/golden")
REF_PATH = os.path.join(GOLDEN, "pbmc3k_reference.h5ad")
SCX_PATH = os.path.join(GOLDEN, "pbmc3k_scx.h5ad")

EXPECTED_N_OBS  = 2700
EXPECTED_N_VARS = 13714
EXPECTED_NNZ    = 2282976


def _skip_if_missing(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"golden file not found: {p} — run `pixi run -e test fixtures`")


@pytest.fixture(scope="session")
def ref_adata():
    _skip_if_missing(REF_PATH)
    return ad.read_h5ad(REF_PATH)


@pytest.fixture(scope="session")
def scx_adata():
    _skip_if_missing(SCX_PATH)
    return ad.read_h5ad(SCX_PATH)
