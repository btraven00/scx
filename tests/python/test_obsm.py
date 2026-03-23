import numpy as np
import pytest


OBSM_EXPECTED = {
    "X_pca":  (2700, 30),
    "X_umap": (2700, 2),
}


def test_obsm_keys_present(scx_adata, ref_adata):
    missing = set(ref_adata.obsm.keys()) - set(scx_adata.obsm.keys())
    assert not missing, f"obsm keys missing in SCX: {sorted(missing)}"


@pytest.mark.parametrize("key,expected_shape", OBSM_EXPECTED.items())
def test_obsm_shape(key, expected_shape, scx_adata):
    assert key in scx_adata.obsm, f"obsm['{key}'] missing"
    assert scx_adata.obsm[key].shape == expected_shape


@pytest.mark.parametrize("key", OBSM_EXPECTED.keys())
def test_obsm_shape_matches_reference(key, scx_adata, ref_adata):
    if key not in ref_adata.obsm:
        pytest.skip(f"reference lacks obsm['{key}']")
    assert scx_adata.obsm[key].shape == ref_adata.obsm[key].shape


@pytest.mark.parametrize("key", OBSM_EXPECTED.keys())
def test_obsm_values(key, scx_adata, ref_adata):
    """Numeric values must agree; allow sign flips on individual PCA axes."""
    if key not in ref_adata.obsm:
        pytest.skip(f"reference lacks obsm['{key}']")
    r = np.asarray(ref_adata.obsm[key], dtype=float)
    s = np.asarray(scx_adata.obsm[key], dtype=float)
    if key == "X_pca":
        # Each PCA axis can be independently sign-flipped — test column-wise
        for i in range(r.shape[1]):
            close     = np.allclose(r[:, i],  s[:, i], rtol=1e-3, atol=1e-5)
            close_neg = np.allclose(r[:, i], -s[:, i], rtol=1e-3, atol=1e-5)
            assert close or close_neg, f"PCA axis {i} values differ beyond tolerance"
    else:
        close     = np.allclose(r, s,  rtol=1e-3, atol=1e-5)
        close_neg = np.allclose(r, -s, rtol=1e-3, atol=1e-5)
        assert close or close_neg, f"obsm['{key}'] values differ beyond tolerance"
