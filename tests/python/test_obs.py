import numpy as np
import pytest


def test_obs_columns_present(scx_adata, ref_adata):
    """Every column in the reference must appear in SCX output."""
    missing = set(ref_adata.obs.columns) - set(scx_adata.obs.columns)
    assert not missing, f"obs columns missing in SCX: {sorted(missing)}"


def test_obs_categorical_preserved(scx_adata, ref_adata):
    """Columns stored as categorical in the reference must be categorical in SCX."""
    for col in ref_adata.obs.columns:
        if hasattr(ref_adata.obs[col].dtype, "categories"):
            assert hasattr(scx_adata.obs[col].dtype, "categories"), (
                f"obs['{col}'] should be categorical (is categorical in reference)"
            )


def test_obs_numeric_values(scx_adata, ref_adata):
    """Numeric obs columns must agree within float tolerance."""
    for col in ref_adata.obs.columns:
        if ref_adata.obs[col].dtype.kind in "fiu":
            np.testing.assert_allclose(
                scx_adata.obs[col].to_numpy(dtype=float, na_value=0.0),
                ref_adata.obs[col].to_numpy(dtype=float, na_value=0.0),
                rtol=1e-4,
                err_msg=f"obs column '{col}' values differ",
            )


def test_obs_index_length(scx_adata):
    from conftest import EXPECTED_N_OBS
    assert len(scx_adata.obs) == EXPECTED_N_OBS
