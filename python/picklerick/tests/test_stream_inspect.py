from __future__ import annotations

import numpy as np
import pytest
from conftest import dense
from scipy import sparse

import picklerick as pk


def test_stream_reassembles_x(norman, norman_path) -> None:
    chunks = list(pk.open_stream(norman_path, chunk_size=7))

    assert [c.row_offset for c in chunks] == list(range(0, norman.n_obs, 7))
    for c in chunks:
        assert isinstance(c, pk.MatrixChunk)
        assert c.indptr.dtype == np.uint64
        assert c.indices.dtype == np.uint32
        assert c.data.dtype == np.dtype(c.dtype)
        assert len(c.indptr) == c.nrows + 1
    x = sparse.vstack(
        [sparse.csr_matrix((c.data, c.indices, c.indptr), shape=(c.nrows, c.n_vars)) for c in chunks]
    )
    np.testing.assert_array_equal(x.toarray(), dense(norman.X))


def test_to_scipy_shares_data_and_indices(norman, norman_path) -> None:
    chunks = list(pk.open_stream(norman_path, chunk_size=7))
    blocks = [c.to_scipy() for c in chunks]
    for c, m in zip(chunks, blocks):
        assert isinstance(m, sparse.csr_array)
        assert m.shape == (c.nrows, c.n_vars)
        assert m.indices.dtype == np.int32
        assert np.shares_memory(m.data, c.data)
        assert np.shares_memory(m.indices, c.indices)
    np.testing.assert_array_equal(sparse.vstack(blocks).toarray(), dense(norman.X))


def test_stream_can_be_abandoned(norman_path) -> None:
    stream = pk.open_stream(norman_path, chunk_size=1)
    assert next(stream).nrows == 1
    del stream  # the reader thread must stop, not block on a full channel


def test_stream_bpcells_dir_values(bpcells_csr_path) -> None:
    # Pure-Rust BP-128 decode path; synth_packed_uint_csr is this 4x5 matrix.
    expected = np.array(
        [[1, 0, 3, 0, 0], [0, 2, 0, 4, 0], [5, 0, 0, 0, 6], [0, 0, 7, 0, 8]],
        dtype=np.uint32,
    )
    chunks = list(pk.open_stream(bpcells_csr_path, chunk_size=2))
    assert all(c.dtype == "uint32" for c in chunks)
    x = sparse.vstack(
        [sparse.csr_matrix((c.data, c.indices, c.indptr), shape=(c.nrows, c.n_vars)) for c in chunks]
    )
    np.testing.assert_array_equal(x.toarray(), expected)


def test_stream_missing_file_raises(tmp_path) -> None:
    with pytest.raises(pk.PickleRickError):
        pk.open_stream(tmp_path / "nope.h5ad")


def test_inspect_h5ad(norman, norman_path) -> None:
    info = pk.inspect(norman_path)

    assert info["format"] == "H5AD"
    assert (info["n_obs"], info["n_vars"]) == norman.shape
    assert info["obs_cols"] == list(norman.obs.columns)
    assert info["obs_dtypes"][info["obs_cols"].index("condition")] == "categorical"
    assert info["x_stats"]["nnz"] == np.count_nonzero(dense(norman.X))
    assert [layer["name"] for layer in info["layers"]] == ["counts"]


def test_inspect_names(norman, norman_path, bpcells_csr_path) -> None:
    assert "obs_names" not in pk.inspect(norman_path)
    info = pk.inspect(norman_path, names=True)
    assert info["obs_names"] == list(norman.obs_names)
    assert info["var_names"] == list(norman.var_names)
    info = pk.inspect(bpcells_csr_path, names=True)
    assert (len(info["obs_names"]), len(info["var_names"])) == (4, 5)


def test_inspect_bpcells_dir(bpcells_csr_path) -> None:
    info = pk.inspect(bpcells_csr_path)
    assert info["format"] == "BPCells"
    assert (info["n_obs"], info["n_vars"]) == (4, 5)


def test_version() -> None:
    assert pk.__version__.startswith("0.")


def _stack(chunks):
    blocks = [c.to_scipy() for c in chunks]
    return sparse.vstack(blocks).toarray() if blocks else np.empty((0, 0))


def test_stream_where_reads_only_matching_rows(norman, norman_path) -> None:
    mask = (norman.obs["split"] == "train").to_numpy()
    chunks = list(pk.open_stream(norman_path, chunk_size=7, where={"split": "train"}))
    assert sum(c.nrows for c in chunks) == int(mask.sum())
    # Each chunk's row_offset is its first row in the full matrix.
    assert all(mask[c.row_offset : c.row_offset + c.nrows].all() for c in chunks)
    np.testing.assert_array_equal(_stack(chunks), dense(norman.X)[mask])

    # Values are matched as text: the int column `control` against str(1).
    both = (norman.obs["split"] == "train") & (norman.obs["control"] == 1)
    chunks = list(pk.open_stream(norman_path, where={"split": "train", "control": 1}))
    assert both.sum() > 0
    assert sum(c.nrows for c in chunks) == int(both.sum())
    np.testing.assert_array_equal(_stack(chunks), dense(norman.X)[both.to_numpy()])

    assert list(pk.open_stream(norman_path, where={"split": "nope"})) == []
    with pytest.raises(pk.PickleRickError, match="obs column"):
        list(pk.open_stream(norman_path, where={"no_such_column": "x"}))


def test_stream_rows_accepts_mask_indices_and_ranges(norman, norman_path) -> None:
    x = dense(norman.X)
    mask = np.zeros(norman.n_obs, dtype=bool)
    mask[[3, 4, 5, 10, 40, 41]] = True
    for rows in (mask, [41, 3, 5, 4, 10, 40, 40], [(3, 6), (10, 11), (40, 42)]):
        chunks = list(pk.open_stream(norman_path, chunk_size=2, rows=rows))
        assert [c.row_offset for c in chunks] == [3, 5, 10, 40]
        np.testing.assert_array_equal(_stack(chunks), x[mask])
    # rows and where intersect.
    train = (norman.obs["split"] == "train").to_numpy()
    chunks = list(pk.open_stream(norman_path, rows=mask, where={"split": "train"}))
    np.testing.assert_array_equal(_stack(chunks), x[mask & train])
    with pytest.raises(pk.PickleRickError, match="beyond"):
        list(pk.open_stream(norman_path, rows=[(0, norman.n_obs + 1)]))
