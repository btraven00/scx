import numpy as np
import scipy.sparse as sp
from conftest import EXPECTED_NNZ


def test_nnz(scx_adata):
    assert scx_adata.X.nnz == EXPECTED_NNZ


def test_nnz_matches_reference(scx_adata, ref_adata):
    assert scx_adata.X.nnz == ref_adata.X.nnz


def test_row_sums(scx_adata, ref_adata):
    """Total counts per cell must agree."""
    r = np.asarray(ref_adata.X.sum(axis=1)).ravel()
    s = np.asarray(scx_adata.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(s, r, rtol=1e-4, err_msg="X row sums differ")


def test_col_sums(scx_adata, ref_adata):
    """Total counts per gene must agree."""
    r = np.asarray(ref_adata.X.sum(axis=0)).ravel()
    s = np.asarray(scx_adata.X.sum(axis=0)).ravel()
    np.testing.assert_allclose(s, r, rtol=1e-4, err_msg="X col sums differ")


def test_spot_values(scx_adata, ref_adata):
    """Exact value check on first 100 cells × 100 genes submatrix."""
    r = ref_adata.X[:100, :100].toarray()
    s = scx_adata.X[:100, :100].toarray()
    np.testing.assert_allclose(s, r, rtol=1e-4, err_msg="X spot values differ")


def test_x_is_sparse(scx_adata):
    assert sp.issparse(scx_adata.X), "X should be sparse"
