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


def test_inspect_bpcells_dir(bpcells_csr_path) -> None:
    info = pk.inspect(bpcells_csr_path)
    assert info["format"] == "BPCells"
    assert (info["n_obs"], info["n_vars"]) == (4, 5)


def test_version() -> None:
    assert pk.__version__.startswith("0.")
