from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest

import picklerick as pk


def test_convert_h5ad_preserves_shape_and_obs_columns(
    require_fixtures: None,
    require_scx: str,
    h5ad_ref_path: Path,
    tmp_path: Path,
    expected_n_obs: int,
    expected_n_vars: int,
) -> None:
    output = tmp_path / "roundtrip.h5ad"

    result = pk.convert(h5ad_ref_path, output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 1000

    adata = ad.read_h5ad(output)
    assert adata.shape == (expected_n_obs, expected_n_vars)
    assert "nCount_RNA" in adata.obs.columns
    assert "nFeature_RNA" in adata.obs.columns


def test_convert_h5ad_preserves_obsm_embeddings(
    require_fixtures: None,
    require_scx: str,
    h5ad_ref_path: Path,
    tmp_path: Path,
    expected_n_obs: int,
) -> None:
    output = tmp_path / "obsm.h5ad"

    pk.convert(h5ad_ref_path, output)

    adata = ad.read_h5ad(output)
    assert "X_pca" in adata.obsm
    assert "X_umap" in adata.obsm
    assert adata.obsm["X_pca"].shape == (expected_n_obs, 30)
    assert adata.obsm["X_umap"].shape == (expected_n_obs, 2)


def test_convert_h5ad_dtype_f64_produces_larger_file(
    require_fixtures: None,
    require_scx: str,
    h5ad_ref_path: Path,
    tmp_path: Path,
) -> None:
    out_f32 = tmp_path / "out_f32.h5ad"
    out_f64 = tmp_path / "out_f64.h5ad"

    pk.convert(h5ad_ref_path, out_f32, dtype="f32")
    pk.convert(h5ad_ref_path, out_f64, dtype="f64")

    assert out_f32.exists()
    assert out_f64.exists()
    assert out_f64.stat().st_size > out_f32.stat().st_size


def test_convert_h5seurat_preserves_shape(
    require_fixtures: None,
    require_scx: str,
    h5seurat_path: Path,
    tmp_path: Path,
    expected_n_obs: int,
    expected_n_vars: int,
) -> None:
    output = tmp_path / "converted_from_h5seurat.h5ad"

    pk.convert(h5seurat_path, output)

    adata = ad.read_h5ad(output)
    assert adata.shape == (expected_n_obs, expected_n_vars)


def test_different_chunk_sizes_produce_equivalent_outputs(
    require_fixtures: None,
    require_scx: str,
    h5ad_ref_path: Path,
    tmp_path: Path,
) -> None:
    out_big = tmp_path / "big_chunk.h5ad"
    out_small = tmp_path / "small_chunk.h5ad"

    pk.convert(h5ad_ref_path, out_big, chunk_size=5000)
    pk.convert(h5ad_ref_path, out_small, chunk_size=100)

    ad_big = ad.read_h5ad(out_big)
    ad_small = ad.read_h5ad(out_small)

    assert ad_big.shape == ad_small.shape
    assert list(ad_big.obs.columns) == list(ad_small.obs.columns)
    assert list(ad_big.obsm.keys()) == list(ad_small.obsm.keys())


def test_convert_invalid_dtype_raises_error(
    require_fixtures: None,
    require_scx: str,
    h5ad_ref_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid_dtype.h5ad"

    with pytest.raises(pk.ScxCommandError):
        pk.convert(h5ad_ref_path, output, dtype="not_a_dtype")