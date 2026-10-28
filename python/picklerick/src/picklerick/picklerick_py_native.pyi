# Type stub for the compiled engine module (rust/src/lib.rs). The docstrings
# repeat the Rust doc comments so that type checkers and the docs site see
# them; tests/test_stub.py fails when the two drift apart.
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.sparse import csr_array

class PickleRickError(Exception):
    """Raised when the SCX engine cannot read, convert or write a dataset."""

class MatrixChunk:
    """A chunk of consecutive rows of the matrix, as CSR numpy arrays.

    The arrays take ownership of the buffers the reader decoded into, so
    there is no per-chunk copy. They are writable and outlive the chunk.
    """

    @property
    def row_offset(self) -> int:
        """Index of the first row of this chunk in the full matrix."""
    @property
    def nrows(self) -> int:
        """Number of rows in this chunk."""
    @property
    def n_vars(self) -> int:
        """Number of columns (features) of the full matrix."""
    @property
    def dtype(self) -> str:
        """NumPy dtype name of `data`, e.g. ``"float32"``."""
    @property
    def indptr(self) -> np.ndarray:
        """``(nrows + 1,)`` uint64 row pointers."""
    @property
    def indices(self) -> np.ndarray:
        """``(nnz,)`` uint32 column indices."""
    @property
    def data(self) -> np.ndarray:
        """``(nnz,)`` values."""
    def to_scipy(self) -> csr_array:
        """The chunk as a ``scipy.sparse.csr_array`` that shares ``data`` and
        ``indices`` with this chunk. scipy wants signed indices of one dtype:
        ``indices`` is viewed as int32 (same bits, never negative) and
        ``indptr`` is cast to int32, a copy of only nrows + 1 entries. Given an
        int64 indptr, scipy would upcast, and so copy, ``indices``.
        """

class MatrixStream(Iterator[MatrixChunk]):
    """Iterator over `MatrixChunk`s, filled by a background reader thread."""

    def __iter__(self) -> MatrixStream: ...
    def __next__(self) -> MatrixChunk: ...

def scx_convert(
    input: str,
    output: str,
    chunk_size: int,
    dtype: str,
    assay: str,
    layer: str,
    x_slot: str = "X",
) -> None: ...
def scx_inspect(input: str, with_names: bool = False) -> dict[str, Any]: ...
def scx_open_stream(
    path: str,
    chunk_size: int,
    assay: str,
    layer: str,
    rows: list[tuple[int, int]] | None = None,
    where: list[tuple[str, list[str]]] | None = None,
) -> MatrixStream:
    """`rows` are (start, stop) half-open row ranges; `where` maps obs columns to
    the accepted values, all of which must hold. Both are turned into row
    ranges; `rows` and `where` together take their intersection.
    """
