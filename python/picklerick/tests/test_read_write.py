from __future__ import annotations

import anndata as ad
import numpy as np
from conftest import dense
from scipy import sparse

import picklerick as pk


def _toy() -> ad.AnnData:
    x = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32))
    adata = ad.AnnData(
        X=x,
        obs={"cell_type": ["a", "b", "a"], "score": [0.1, 0.2, 0.3]},
        var={"gene_symbol": ["g1", "g2", "g3"]},
    )
    adata.obs_names = ["c1", "c2", "c3"]
    adata.var_names = ["g1", "g2", "g3"]
    adata.obsm["X_pca"] = np.eye(3, 2, dtype=np.float32)
    return adata


def test_read_h5ad_matches_anndata(norman, norman_path) -> None:
    got = pk.read(norman_path)
    np.testing.assert_array_equal(dense(got.X), dense(norman.X))
    assert list(got.obs.columns) == list(norman.obs.columns)


def test_read_bpcells_dir(bpcells_csr_path) -> None:
    got = pk.read(bpcells_csr_path)
    assert got.shape == (4, 5)
    assert got.X.nnz == 8


def test_write_h5seurat_roundtrip(tmp_path) -> None:
    toy = _toy()
    out = pk.write_h5seurat(toy, tmp_path / "toy.h5seurat")
    back = pk.read(out)

    assert pk.inspect(out)["format"] == "H5Seurat"
    np.testing.assert_array_equal(dense(back.X), dense(toy.X))
    assert list(back.obs_names) == list(toy.obs_names)
    assert list(back.var_names) == list(toy.var_names)
    assert back.obs["cell_type"].astype(str).tolist() == ["a", "b", "a"]
    np.testing.assert_allclose(back.obs["score"], toy.obs["score"])
    assert "X_pca" in back.obsm


def test_read_zarr_matches_anndata(zarr_path) -> None:
    got = pk.read(zarr_path)
    ref = ad.read_zarr(zarr_path)
    np.testing.assert_array_equal(dense(got.X), dense(ref.X))
    assert list(got.obs_names) == list(ref.obs_names)
    np.testing.assert_allclose(got.obsm["X_pca"], ref.obsm["X_pca"])
