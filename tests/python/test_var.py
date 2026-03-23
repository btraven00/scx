import numpy as np
from conftest import EXPECTED_N_VARS


def test_var_index_length(scx_adata):
    assert len(scx_adata.var) == EXPECTED_N_VARS


def test_var_columns_present(scx_adata, ref_adata):
    """Every var column in the reference must appear in SCX output."""
    missing = set(ref_adata.var.columns) - set(scx_adata.var.columns)
    assert not missing, f"var columns missing in SCX: {sorted(missing)}"


def test_var_numeric_values(scx_adata, ref_adata):
    """Numeric var columns must agree within float tolerance."""
    for col in ref_adata.var.columns:
        if ref_adata.var[col].dtype.kind in "fiu":
            np.testing.assert_allclose(
                scx_adata.var[col].to_numpy(dtype=float, na_value=0.0),
                ref_adata.var[col].to_numpy(dtype=float, na_value=0.0),
                rtol=1e-4,
                err_msg=f"var column '{col}' values differ",
            )
