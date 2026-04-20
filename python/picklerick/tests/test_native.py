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


def test_open_stream_h5ad(
    h5ad_ref_path: Path,
    require_fixtures: None,
    expected_n_obs: int,
    expected_n_vars: int,
    expected_nnz: int,
) -> None:
    import numpy as np

    chunks = list(pk.open_stream(h5ad_ref_path, chunk_size=1000))

    assert len(chunks) > 0
    # Row offsets must be contiguous
    assert chunks[0].row_offset == 0
    for i, chunk in enumerate(chunks):
        assert chunk.n_vars == expected_n_vars
        assert chunk.indptr.dtype == np.uint64
        assert chunk.indices.dtype == np.uint32
        assert len(chunk.indptr) == chunk.nrows + 1
        assert chunk.indptr[0] == 0

    total_obs = sum(c.nrows for c in chunks)
    total_nnz = sum(len(c.indices) for c in chunks)
    assert total_obs == expected_n_obs
    assert total_nnz == expected_nnz


def test_open_stream_h5seurat(
    h5seurat_path: Path,
    require_fixtures: None,
    expected_n_obs: int,
    expected_n_vars: int,
) -> None:
    chunks = list(pk.open_stream(h5seurat_path, chunk_size=500))

    total_obs = sum(c.nrows for c in chunks)
    assert total_obs == expected_n_obs
    assert all(c.n_vars == expected_n_vars for c in chunks)


def test_open_stream_chunk_row_offsets(
    h5ad_ref_path: Path,
    require_fixtures: None,
) -> None:
    offset = 0
    for chunk in pk.open_stream(h5ad_ref_path, chunk_size=500):
        assert chunk.row_offset == offset
        offset += chunk.nrows


def test_open_stream_early_stop(
    h5ad_ref_path: Path,
    require_fixtures: None,
) -> None:
    # Dropping the iterator mid-stream must not deadlock or raise.
    stream = pk.open_stream(h5ad_ref_path, chunk_size=500)
    first = next(iter(stream))
    assert first.nrows > 0
    del stream  # background thread should drain gracefully