from __future__ import annotations

import anndata as ad

import picklerick as pk


def test_read_h5seurat_returns_anndata_with_expected_shape(
    h5seurat_path,
    require_fixtures,
    require_scx,
    expected_n_obs,
    expected_n_vars,
) -> None:
    adata = pk.read_h5seurat(h5seurat_path)

    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (expected_n_obs, expected_n_vars)


def test_read_h5seurat_x_has_expected_nnz(
    h5seurat_path,
    require_fixtures,
    require_scx,
    expected_nnz,
) -> None:
    adata = pk.read_h5seurat(h5seurat_path)

    assert adata.X is not None
    assert adata.X.nnz == expected_nnz


def test_read_h5seurat_preserves_obs_columns(
    h5seurat_path,
    require_fixtures,
    require_scx,
) -> None:
    adata = pk.read_h5seurat(h5seurat_path)

    assert adata.obs.shape[1] > 0


def test_read_h5seurat_preserves_obsm_keys(
    h5seurat_path,
    require_fixtures,
    require_scx,
) -> None:
    adata = pk.read_h5seurat(h5seurat_path)

    assert "X_pca" in adata.obsm
    assert "X_umap" in adata.obsm


def test_read_h5ad_returns_anndata(
    h5ad_ref_path,
    require_fixtures,
    expected_n_obs,
    expected_n_vars,
) -> None:
    adata = pk.read_h5ad(h5ad_ref_path)

    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (expected_n_obs, expected_n_vars)


def test_read_dataset_dispatches_on_extension(
    h5seurat_path,
    h5ad_ref_path,
    require_fixtures,
    require_scx,
    expected_n_obs,
    expected_n_vars,
) -> None:
    adata_seurat = pk.read_dataset(h5seurat_path)
    adata_h5ad = pk.read_dataset(h5ad_ref_path)

    assert adata_seurat.shape == (expected_n_obs, expected_n_vars)
    assert adata_h5ad.shape == (expected_n_obs, expected_n_vars)


def test_read_alias_matches_read_dataset(
    h5seurat_path,
    require_fixtures,
    require_scx,
) -> None:
    adata_alias = pk.read(h5seurat_path)
    adata_explicit = pk.read_dataset(h5seurat_path)

    assert adata_alias.shape == adata_explicit.shape
    assert list(adata_alias.obs_names) == list(adata_explicit.obs_names)
    assert list(adata_alias.var_names) == list(adata_explicit.var_names)


def test_read_h5seurat_matches_reference_h5ad(
    h5seurat_path,
    h5ad_ref_path,
    require_fixtures,
    require_scx,
) -> None:
    scx_adata = pk.read_h5seurat(h5seurat_path)
    ref_adata = pk.read_h5ad(h5ad_ref_path)

    assert scx_adata.shape == ref_adata.shape
    assert scx_adata.X.nnz == ref_adata.X.nnz
    assert list(scx_adata.obs_names) == list(ref_adata.obs_names)
    assert list(scx_adata.var_names) == list(ref_adata.var_names)