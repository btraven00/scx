# Type stub for the compiled engine module (rust/src/lib.rs). The docstrings
# repeat the Rust doc comments so that type checkers and the docs site see
# them; tests/test_stub.py fails when the two drift apart.
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.sparse import csc_array, csr_array

class PickleRickError(Exception):
    """Raised when the SCX engine cannot read, convert or write a dataset."""

class MatrixChunk:
    """A block of the matrix, as compressed numpy arrays: consecutive rows
    (``axis == "obs"``, CSR, every column) or consecutive genes
    (``axis == "var"``, CSC, every row). The block is rows
    ``row_offset:row_offset + nrows`` by columns
    ``var_offset:var_offset + ncols``.

    The arrays take ownership of the buffers the reader decoded into, so
    there is no per-chunk copy. They are writable and outlive the chunk.
    """

    @property
    def axis(self) -> str:
        """``"obs"`` for a block of rows (CSR), ``"var"`` for a block of genes (CSC)."""

    @property
    def row_offset(self) -> int:
        """Index of the first row of this chunk in the full matrix."""
    @property
    def nrows(self) -> int:
        """Number of rows in this chunk."""
    @property
    def var_offset(self) -> int:
        """Index of the first column of this chunk in the full matrix."""
    @property
    def ncols(self) -> int:
        """Number of columns in this chunk."""
    @property
    def n_vars(self) -> int:
        """Number of columns (features) of the full matrix."""
    @property
    def dtype(self) -> str:
        """NumPy dtype name of `data`, e.g. ``"float32"``."""
    @property
    def indptr(self) -> np.ndarray:
        """uint64 pointers: ``(nrows + 1,)`` per row for ``"obs"``,
        ``(ncols + 1,)`` per gene for ``"var"``.
        """
    @property
    def indices(self) -> np.ndarray:
        """``(nnz,)`` uint32 column indices (``"obs"``) or row indices (``"var"``)."""
    @property
    def data(self) -> np.ndarray:
        """``(nnz,)`` values."""
    def to_scipy(self) -> csr_array | csc_array:
        """The chunk as a ``scipy.sparse.csr_array`` (``"obs"``) or
        ``csc_array`` (``"var"``) of shape ``(nrows, ncols)`` that shares
        ``data`` and ``indices`` with this chunk. scipy wants signed indices
        of one dtype: ``indices`` is viewed as int32 (same bits, never
        negative) and ``indptr`` is cast to int32, a copy of only its few
        entries. Given an int64 indptr, scipy would upcast, and so copy,
        ``indices``.
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
    transpose_mem: int | None = None,
) -> None: ...
def scx_inspect(input: str, with_names: bool = False) -> dict[str, Any]: ...
def scx_open_stream(
    path: str,
    chunk_size: int,
    assay: str,
    layer: str,
    rows: list[tuple[int, int]] | None = None,
    where: list[tuple[str, list[str]]] | None = None,
    by_gene: bool = False,
    genes: list[tuple[int, int]] | None = None,
) -> MatrixStream:
    """`rows` are (start, stop) half-open row ranges; `where` maps obs columns to
    the accepted values, all of which must hold. Both are turned into row
    ranges; `rows` and `where` together take their intersection. `by_gene`
    streams by gene instead, over the (start, stop) ranges in `genes`, or
    every gene when that is `None`.
    """
