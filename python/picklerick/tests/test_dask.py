from __future__ import annotations

import numpy as np
import pytest
from conftest import dense
from scipy import sparse

import picklerick as pk

da = pytest.importorskip("dask.array")


def test_open_dask_is_lazy_row_blocks(norman, norman_path) -> None:
    X = pk.open_dask(norman_path, chunk_size=7)

    assert X.shape == norman.X.shape
    assert X.chunks == (
        tuple(min(7, norman.n_obs - a) for a in range(0, norman.n_obs, 7)),
        (norman.n_vars,),
    )
    assert X.dtype == norman.X.dtype
    assert isinstance(X._meta, sparse.csr_array)
    np.testing.assert_array_equal(dense(X.compute()), dense(norman.X))


def test_open_dask_blocks_slice_and_subset(norman, norman_path) -> None:
    X = pk.open_dask(norman_path, chunk_size=7)
    rows, genes = slice(5, 23), np.array([0, 3, 4, 10])
    np.testing.assert_array_equal(
        dense(X[rows][:, genes].compute()), dense(norman.X)[rows][:, genes]
    )
    # Blocks map one to one onto row ranges: block 2 is rows 14..21.
    np.testing.assert_array_equal(dense(X.blocks[2].compute()), dense(norman.X)[14:21])


def test_open_dask_rejects_gene_major_and_bad_chunks(norman, tmp_path, norman_path) -> None:
    a = norman.copy()
    a.X = sparse.csc_matrix(a.X)
    csc = tmp_path / "csc.h5ad"
    a.write_h5ad(csc)
    with pytest.raises(ValueError, match='axis="var"'):
        pk.open_dask(csc)
    with pytest.raises(ValueError, match="chunk_size"):
        pk.open_dask(norman_path, chunk_size=0)
    with pytest.raises(pk.PickleRickError):
        pk.open_dask(tmp_path / "missing.h5ad")


def test_open_dask_feeds_scanpy_covariance_pca(norman, norman_path) -> None:
    # The use this exists for: scanpy's exact block-wise PCA on dask input.
    # The reference is the same dask path over the matrix held in memory:
    # scanpy's in-memory path scales variance by 1/(n-1), its dask path by 1/n.
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    mem = da.from_array(sparse.csr_array(norman.X), chunks=(7, -1))
    ref = ad.AnnData(mem.astype(np.float64))
    lazy = ad.AnnData(pk.open_dask(norman_path, chunk_size=7).astype(np.float64))
    for a in (ref, lazy):
        sc.pp.pca(a, n_comps=5, svd_solver="covariance_eigh")
    np.testing.assert_allclose(
        np.asarray(lazy.uns["pca"]["variance"]), np.asarray(ref.uns["pca"]["variance"]), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.abs(np.asarray(lazy.varm["PCs"])), np.abs(np.asarray(ref.varm["PCs"])), atol=1e-10
    )
