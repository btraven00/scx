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


def test_gene_major_writes_x_by_gene(norman, norman_path, tmp_path) -> None:
    # A tiny budget forces many passes over the input.
    out = pk.convert(norman_path, tmp_path / "by_gene.h5ad", gene_major=True, transpose_mem=1e-5)
    assert pk.inspect(out)["x_axis"] == "var"
    back = ad.read_h5ad(out)
    assert back.X.format == "csc"
    np.testing.assert_array_equal(dense(back.X), dense(norman.X))
    # Streamed by gene, the blocks put back together are X.
    blocks = [c.to_scipy().toarray() for c in pk.open_stream(out, axis="var", chunk_size=7)]
    np.testing.assert_array_equal(np.hstack(blocks), dense(norman.X))


def test_gene_major_needs_h5ad_output_and_a_positive_budget(norman_path, tmp_path) -> None:
    with pytest.raises(Exception, match="gene_major only applies to .h5ad"):
        pk.convert(norman_path, tmp_path / "x.h5seurat", gene_major=True)
    with pytest.raises(ValueError, match="transpose_mem"):
        pk.convert(norman_path, tmp_path / "x.h5ad", gene_major=True, transpose_mem=0)


def _where_source(tmp_path):
    """A small AnnData with every cell-axis slot, written to h5ad."""
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n, m = 12, 7
    X = sp.random(n, m, density=0.4, format="csr", random_state=1, dtype=np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["B", "T", None, "B", "NK", "T", "T", "B", "NK", "B", "T", None]
            ),
            "donor": pd.Categorical(["d0", "d1"] * 6),
            "ok": [True, False] * 3 + [False, True] * 3,
            "batch": pd.array([1, 2, None, 2, 1, None, 1, 2, 2, 1, None, 1], dtype="Int32"),
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(m)]))
    a.layers["counts"] = sp.csr_matrix(np.round(X.toarray() * 10).astype(np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, 3))
    a.obsp["connectivities"] = sp.random(n, n, density=0.3, format="csr", random_state=2)
    a.raw = ad.AnnData(X=a.layers["counts"].copy(), obs=a.obs[[]], var=a.var)
    path = tmp_path / "src.h5ad"
    a.write_h5ad(path)
    return ad.read_h5ad(path), path


@pytest.mark.parametrize(
    "where",
    [
        {"cell_type": ["B", "T"]},  # several values OR; NA never matches
        {"cell_type": "B", "donor": "d1"},  # columns AND
        {"ok": True},  # bool, as text
        {"batch": 2},  # nullable int, NA never matches
    ],
)
@pytest.mark.parametrize("ext", ["h5ad", "zarr"])
def test_convert_where_equals_anndata_subset(tmp_path, where, ext) -> None:
    if ext == "zarr":
        pytest.importorskip("zarr", minversion="3")  # scx writes Zarr v3
    src, path = _where_source(tmp_path)
    mask = np.ones(src.n_obs, dtype=bool)
    for col, vals in where.items():
        vals = vals if isinstance(vals, list) else [vals]
        mask &= src.obs[col].astype(str).isin([str(v) for v in vals]).to_numpy()
        mask &= src.obs[col].notna().to_numpy()
    want = src[mask]
    out = pk.convert(path, tmp_path / f"sub.{ext}", where=where, chunk_size=2)
    got = ad.read_h5ad(out) if ext == "h5ad" else ad.read_zarr(out)

    assert got.n_obs == mask.sum() > 0
    assert list(got.obs_names) == list(want.obs_names)
    assert list(got.var_names) == list(want.var_names)
    np.testing.assert_array_equal(dense(got.X), dense(want.X))
    np.testing.assert_array_equal(dense(got.layers["counts"]), dense(want.layers["counts"]))
    np.testing.assert_allclose(got.obsm["X_pca"], want.obsm["X_pca"])
    # obsp keeps the matching cells on both axes, renumbered.
    np.testing.assert_allclose(
        dense(got.obsp["connectivities"]), dense(want.obsp["connectivities"])
    )
    for col in want.obs.columns:
        assert got.obs[col].astype(str).tolist() == want.obs[col].astype(str).tolist(), col


def test_convert_where_raw_goes_to_seurat_counts(tmp_path) -> None:
    src, path = _where_source(tmp_path)
    mask = src.obs["cell_type"].isin(["B"]).to_numpy()
    out = pk.convert(path, tmp_path / "sub.h5seurat", where={"cell_type": "B"})
    counts = pk.read(out, layer="counts")
    assert list(counts.obs_names) == list(src.obs_names[mask])
    np.testing.assert_array_equal(dense(counts.X), dense(src.raw.X[mask]))


def test_convert_where_no_match_and_missing_column(tmp_path) -> None:
    _, path = _where_source(tmp_path)
    empty = ad.read_h5ad(pk.convert(path, tmp_path / "e.h5ad", where={"cell_type": "nope"}))
    assert empty.shape == (0, 7)
    with pytest.raises(pk.PickleRickError, match="celltype"):
        pk.convert(path, tmp_path / "x.h5ad", where={"celltype": "B"})
