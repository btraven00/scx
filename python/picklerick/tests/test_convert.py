from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from conftest import dense, golden

import picklerick as pk


def test_h5ad_roundtrip_is_exact(norman, norman_path, tmp_path) -> None:
    out = pk.convert(norman_path, tmp_path / "out.h5ad")
    back = ad.read_h5ad(out)

    np.testing.assert_array_equal(dense(back.X), dense(norman.X))
    np.testing.assert_array_equal(dense(back.layers["counts"]), norman.layers["counts"])
    assert list(back.obs_names) == list(norman.obs_names)
    assert list(back.var_names) == list(norman.var_names)
    for col in norman.obs.columns:
        assert back.obs[col].astype(str).tolist() == norman.obs[col].astype(str).tolist(), col


def test_h5seurat_output_is_h5seurat(norman_path, tmp_path) -> None:
    # Regression: convert() used to write H5AD whatever the extension.
    out = pk.convert(norman_path, tmp_path / "out.h5seurat")
    assert pk.inspect(out)["format"] == "H5Seurat"


def test_h5seurat_roundtrip_keeps_counts_and_data(norman, norman_path, tmp_path) -> None:
    # Source has a `counts` layer, so X goes to the Seurat `data` slot and
    # the layer to `counts` (same rule as `scx convert --x-slot auto`).
    seurat = pk.convert(norman_path, tmp_path / "out.h5seurat")
    back = pk.read(seurat, layer="counts")

    np.testing.assert_array_equal(dense(back.X), norman.layers["counts"])
    np.testing.assert_array_equal(dense(back.layers["data"]), dense(norman.X))
    assert list(back.obs_names) == list(norman.obs_names)
    assert set(norman.obs.columns) <= set(back.obs.columns)


def test_chunk_size_does_not_change_output(norman, norman_path, tmp_path) -> None:
    small = ad.read_h5ad(pk.convert(norman_path, tmp_path / "small.h5ad", chunk_size=7))
    np.testing.assert_array_equal(dense(small.X), dense(norman.X))


def test_dtype_f64(norman_path, tmp_path) -> None:
    back = ad.read_h5ad(pk.convert(norman_path, tmp_path / "f64.h5ad", dtype="f64"))
    assert back.X.dtype == np.float64


def test_creates_parent_directories(norman_path, tmp_path) -> None:
    out = pk.convert(norman_path, tmp_path / "a" / "b" / "out.h5ad")
    assert out.exists()


def test_rejects_unknown_dtype(norman_path, tmp_path) -> None:
    with pytest.raises(ValueError, match="dtype"):
        pk.convert(norman_path, tmp_path / "out.h5ad", dtype="f16")


def test_rejects_unsupported_output_extension(norman_path, tmp_path) -> None:
    with pytest.raises(pk.PickleRickError, match="unsupported output extension"):
        pk.convert(norman_path, tmp_path / "out.loom")


def test_missing_input_raises(tmp_path) -> None:
    with pytest.raises(pk.PickleRickError):
        pk.convert(tmp_path / "nope.h5ad", tmp_path / "out.h5ad")


def test_pbmc3k_h5seurat_matches_reference(tmp_path) -> None:
    ref = ad.read_h5ad(golden("pbmc3k_reference.h5ad"))
    got = ad.read_h5ad(pk.convert(golden("pbmc3k.h5seurat"), tmp_path / "out.h5ad"))

    assert got.shape == ref.shape
    assert got.X.nnz == ref.X.nnz
    assert list(got.obs_names) == list(ref.obs_names)
    assert list(got.var_names) == list(ref.var_names)
    assert {"X_pca", "X_umap"} <= set(got.obsm)


def test_x_slot_layer_leaves_x_absent(norman, norman_path, tmp_path) -> None:
    # The matrix lands in layers/raw, X is absent; the fixture's own `counts`
    # layer is untouched because its name differs.
    back = ad.read_h5ad(pk.convert(norman_path, tmp_path / "slot.h5ad", x_slot="layers/raw"))
    assert back.X is None
    np.testing.assert_array_equal(dense(back.layers["raw"]), dense(norman.X))
    np.testing.assert_array_equal(dense(back.layers["counts"]), norman.layers["counts"])

    # Same name as a source layer: the source layer is dropped, not written twice.
    back = ad.read_h5ad(pk.convert(norman_path, tmp_path / "clash.h5ad", x_slot="layers/counts"))
    assert back.X is None
    np.testing.assert_array_equal(dense(back.layers["counts"]), dense(norman.X))

    with pytest.raises(Exception, match="layers/<name>"):
        pk.convert(norman_path, tmp_path / "bad.h5ad", x_slot="counts")


def test_zarr_output_reads_back_with_anndata(norman, norman_path, tmp_path) -> None:
    # Zarr v3 stores need zarr >= 3, which exists for Python >= 3.11 only.
    pytest.importorskip("zarr", minversion="3")

    out = pk.convert(norman_path, tmp_path / "out.zarr")
    assert out.is_dir()
    back = ad.read_zarr(out)
    assert back.shape == norman.shape
    np.testing.assert_array_equal(dense(back.X), dense(norman.X))
    assert list(back.obs_names) == list(norman.obs_names)
    assert list(back.var_names) == list(norman.var_names)
    for col in norman.obs.columns:
        assert back.obs[col].astype(str).tolist() == norman.obs[col].astype(str).tolist(), col
    assert set(back.layers) == set(norman.layers)
    np.testing.assert_array_equal(dense(back.layers["counts"]), dense(norman.layers["counts"]))
    # And through picklerick's own reader.
    assert pk.inspect(out)["format"].startswith("Zarr")
    np.testing.assert_array_equal(dense(pk.read(out).X), dense(norman.X))
