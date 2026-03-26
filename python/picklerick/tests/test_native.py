from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest

import picklerick as pk
from picklerick import _native


pytestmark = pytest.mark.skipif(
    not _native.native_available(),
    reason="native backend not available",
)


def test_native_backend_reports_available() -> None:
    assert _native.native_available() is True
    assert pk.native_available() is True


def test_convert_via_native_smoke(
    h5ad_ref_path: Path,
    require_fixtures: None,
    tmp_path: Path,
    expected_n_obs: int,
    expected_n_vars: int,
) -> None:
    output = tmp_path / "native_convert.h5ad"

    used_native = _native.convert_via_native(
        h5ad_ref_path,
        output,
        chunk_size=5000,
        dtype="f32",
        assay="RNA",
        layer="counts",
    )

    assert used_native is True
    assert output.exists()

    adata = ad.read_h5ad(output)
    assert adata.shape == (expected_n_obs, expected_n_vars)


def test_write_h5seurat_via_native_smoke(
    h5ad_ref_path: Path,
    require_fixtures: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "native_write.h5seurat"

    used_native = _native.write_h5seurat_via_native(
        h5ad_ref_path,
        output,
        chunk_size=5000,
        assay="RNA",
    )

    assert used_native is True
    assert output.exists()
    assert output.stat().st_size > 0


def test_public_api_uses_native_when_available(
    h5seurat_path: Path,
    require_fixtures: None,
    expected_n_obs: int,
    expected_n_vars: int,
) -> None:
    adata = pk.read_h5seurat(h5seurat_path)

    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (expected_n_obs, expected_n_vars)