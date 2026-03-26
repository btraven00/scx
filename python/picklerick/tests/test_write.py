from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

import picklerick as pk
from picklerick import ScxNotFoundError


def _toy_adata() -> ad.AnnData:
    x = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
                [4.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
    )

    obs = {
        "cell_type": ["a", "b", "c"],
        "score": [0.1, 0.2, 0.3],
    }
    var = {
        "gene_symbol": ["g1", "g2", "g3"],
    }

    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.obsm["X_pca"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    return adata


def test_write_h5ad_creates_file(tmp_path: Path) -> None:
    adata = _toy_adata()
    out = tmp_path / "toy.h5ad"

    result = pk.write_h5ad(adata, out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert Path(result) == out


def test_write_h5ad_roundtrip_shape(tmp_path: Path) -> None:
    adata = _toy_adata()
    out = tmp_path / "toy_roundtrip.h5ad"

    pk.write_h5ad(adata, out)
    reread = ad.read_h5ad(out)

    assert reread.shape == adata.shape
    assert list(reread.obs_names) == list(adata.obs_names)
    assert list(reread.var_names) == list(adata.var_names)
    assert "X_pca" in reread.obsm


def test_write_h5seurat_creates_file(tmp_path: Path, require_scx: str) -> None:
    adata = _toy_adata()
    out = tmp_path / "toy.h5seurat"

    result = pk.write_h5seurat(adata, out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert Path(result) == out


def test_write_h5seurat_roundtrip_via_read_h5seurat(
    tmp_path: Path,
    require_scx: str,
) -> None:
    adata = _toy_adata()
    out = tmp_path / "toy_roundtrip.h5seurat"

    pk.write_h5seurat(adata, out)
    reread = pk.read_h5seurat(out)

    assert reread.shape == adata.shape
    assert reread.n_obs == adata.n_obs
    assert reread.n_vars == adata.n_vars


def test_write_h5seurat_respects_assay_argument(
    tmp_path: Path,
    require_scx: str,
) -> None:
    adata = _toy_adata()
    out = tmp_path / "toy_assay.h5seurat"

    pk.write_h5seurat(adata, out, assay="RNA")

    assert out.exists()
    assert out.stat().st_size > 0


def test_write_h5seurat_requires_scx_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = _toy_adata()
    out = tmp_path / "missing_scx.h5seurat"

    monkeypatch.delenv("SCX_BIN", raising=False)
    monkeypatch.setenv("PATH", "")

    with pytest.raises(ScxNotFoundError):
        pk.write_h5seurat(adata, out)
