from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Generator

import numpy as np

from ._io import read_h5ad_file, write_h5ad_file
from ._native import (
    convert_via_native,
    inspect_via_native,
    native_available,
    open_stream_via_native,
    write_h5seurat_via_native,
)
from ._util import ensure_parent_directory, is_h5ad_path, normalize_path

if TYPE_CHECKING:
    import anndata as ad


Pathish = str | os.PathLike[str]


def read_h5ad(path: Pathish) -> "ad.AnnData":
    """
    Read an H5AD file into an AnnData object.
    """
    return read_h5ad_file(normalize_path(path))


def read_h5seurat(
    path: Pathish,
    assay: str = "RNA",
    layer: str = "counts",
    chunk_size: int = 5000,
    dtype: str = "f32",
) -> "ad.AnnData":
    """
    Read an H5Seurat file into an AnnData object.

    The input is converted to a temporary H5AD through the optional native
    backend when available, otherwise through the SCX CLI, and then loaded
    with :mod:`anndata`.
    """
    input_path = normalize_path(path)

    with TemporaryDirectory(prefix="picklerick-") as tmpdir:
        tmp_h5ad = Path(tmpdir) / "read_h5seurat_tmp.h5ad"
        convert(
            input=input_path,
            output=tmp_h5ad,
            chunk_size=chunk_size,
            dtype=dtype,
            assay=assay,
            layer=layer,
        )
        return read_h5ad_file(tmp_h5ad)


def read_dataset(
    path: Pathish,
    assay: str = "RNA",
    layer: str = "counts",
    chunk_size: int = 5000,
    dtype: str = "f32",
) -> "ad.AnnData":
    """
    Read a supported dataset into an AnnData object.

    H5AD inputs are read directly. Other inputs are routed through the
    H5Seurat conversion path.
    """
    input_path = normalize_path(path)
    if is_h5ad_path(input_path):
        return read_h5ad(input_path)

    return read_h5seurat(
        input_path,
        assay=assay,
        layer=layer,
        chunk_size=chunk_size,
        dtype=dtype,
    )


def read(
    path: Pathish,
    assay: str = "RNA",
    layer: str = "counts",
    chunk_size: int = 5000,
    dtype: str = "f32",
) -> "ad.AnnData":
    """
    Alias for :func:`read_dataset`.
    """
    return read_dataset(
        path,
        assay=assay,
        layer=layer,
        chunk_size=chunk_size,
        dtype=dtype,
    )


def write_h5ad(
    adata: "ad.AnnData",
    path: Pathish,
    compression: str = "gzip",
):
    """
    Write an AnnData object to H5AD.
    """
    output_path = ensure_parent_directory(path)
    write_h5ad_file(adata, output_path, compression=compression)
    return output_path


def write_h5seurat(
    adata: "ad.AnnData",
    path: Pathish,
    assay: str = "RNA",
    chunk_size: int = 5000,
):
    """
    Write an AnnData object to H5Seurat.
    """
    if not native_available():
        raise RuntimeError(
            "write_h5seurat() requires the native backend. "
            "Install with: pip install picklerick[native]"
        )

    output_path = ensure_parent_directory(path)

    with TemporaryDirectory(prefix="picklerick-") as tmpdir:
        tmp_h5ad = Path(tmpdir) / "write_h5seurat_tmp.h5ad"
        write_h5ad_file(adata, tmp_h5ad, compression="gzip")
        write_h5seurat_via_native(
            input_h5ad=tmp_h5ad,
            output_h5seurat=output_path,
            chunk_size=chunk_size,
            assay=assay,
        )

    return output_path


_VALID_DTYPES = {"f32", "f64", "i32", "u32"}


def convert(
    input: Pathish,
    output: Pathish,
    chunk_size: int = 5000,
    dtype: str = "f32",
    assay: str = "RNA",
    layer: str = "counts",
):
    """
    Convert a supported single-cell dataset to another format.
    """
    if dtype not in _VALID_DTYPES:
        from ._exceptions import ScxCommandError

        raise ScxCommandError(
            f"unknown dtype '{dtype}': use one of {sorted(_VALID_DTYPES)}"
        )

    input_path = normalize_path(input)
    output_path = ensure_parent_directory(output)

    used_native = convert_via_native(
        input_path=input_path,
        output_path=output_path,
        chunk_size=chunk_size,
        dtype=dtype,
        assay=assay,
        layer=layer,
    )
    if not used_native:
        raise RuntimeError(
            "convert() requires the native backend. "
            "Install with: pip install picklerick[native]"
        )

    return output_path


def inspect(
    path: Pathish,
    chunk_size: int = 5000,
) -> dict:
    """
    Inspect a single-cell file and return metadata without loading any data.

    Reads only shape, column names, embedding keys, and layer names. The count
    matrix is never loaded. Memory usage is minimal regardless of dataset size.

    Parameters
    ----------
    path:
        Path to the file (``.h5seurat``, ``.h5ad``, BPCells directory, or ``.h5``).
    chunk_size:
        Internal chunk size used for metadata reads. Default ``5000``.

    Returns
    -------
    dict
        Keys: ``format``, ``n_obs``, ``n_vars``, ``obs_cols``, ``obs_dtypes``,
        ``var_cols``, ``var_dtypes``, ``obsm_keys``, ``layers``, ``uns_keys``,
        ``obsp_keys``, ``varm_keys``.

    Raises
    ------
    RuntimeError
        If the native backend is not installed. Install ``picklerick`` with
        the native extras: ``pip install picklerick[native]``.
    """
    result = inspect_via_native(normalize_path(path), chunk_size=chunk_size)
    if result is None:
        raise RuntimeError(
            "inspect() requires the native backend. "
            "Install with: pip install picklerick[native]"
        )
    return result


_STREAM_DTYPE_MAP: dict[str, type] = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "uint32": np.uint32,
}


class MatrixChunk:
    """
    A single chunk of rows from a streaming matrix read.

    Attributes
    ----------
    row_offset : int
        Index of the first row in this chunk within the full matrix.
    nrows : int
        Number of rows in this chunk.
    n_vars : int
        Total number of features (columns) in the matrix.
    dtype : str
        NumPy dtype string for the ``data`` array (e.g. ``"float32"``).
    indptr : numpy.ndarray
        Shape ``(nrows+1,)``, dtype ``uint64``. CSR row-pointer array.
    indices : numpy.ndarray
        Shape ``(nnz,)``, dtype ``uint32``. Column indices.
    data : numpy.ndarray
        Shape ``(nnz,)``, dtype matches ``self.dtype``. Non-zero values.

    Notes
    -----
    The arrays wrap an immutable ``bytes`` buffer via ``numpy.frombuffer``,
    so they are **read-only**. Copy (``arr.copy()``) before mutating. The
    chunk is copied once out of Rust into the ``bytes`` buffer; the numpy
    wrapping itself is copy-free.
    """

    def __init__(self, native: object) -> None:
        self._native = native
        self.row_offset: int = native.row_offset
        self.nrows: int = native.nrows
        self.n_vars: int = native.n_vars
        self.dtype: str = native.dtype

    @cached_property
    def indptr(self) -> np.ndarray:
        return np.frombuffer(self._native.indptr_bytes, dtype=np.uint64)

    @cached_property
    def indices(self) -> np.ndarray:
        return np.frombuffer(self._native.indices_bytes, dtype=np.uint32)

    @cached_property
    def data(self) -> np.ndarray:
        return np.frombuffer(
            self._native.data_bytes, dtype=_STREAM_DTYPE_MAP[self.dtype]
        )


def open_stream(
    path: Pathish,
    chunk_size: int = 5000,
    assay: str = "RNA",
    layer: str = "counts",
) -> Generator[MatrixChunk, None, None]:
    """
    Stream the count matrix of a single-cell file as row-chunks.

    Yields :class:`MatrixChunk` objects in row order. The matrix is never
    fully materialised — peak RSS stays at ``O(chunk_size * n_vars)``.

    Parameters
    ----------
    path:
        Path to the file (``.h5seurat``, ``.h5ad``, BPCells directory, or ``.h5``).
    chunk_size:
        Number of cells per chunk. Default ``5000``.
    assay:
        Seurat assay name (ignored for H5AD inputs). Default ``"RNA"``.
    layer:
        Seurat layer to read (ignored for H5AD inputs). Default ``"counts"``.

    Yields
    ------
    MatrixChunk
        Each chunk exposes ``row_offset``, ``nrows``, ``n_vars``, ``dtype``,
        and read-only numpy arrays ``indptr``, ``indices``, ``data``.

    Raises
    ------
    RuntimeError
        If the native backend is not installed.

    Examples
    --------
    >>> for chunk in pk.open_stream("atlas.h5seurat", chunk_size=5000):
    ...     X = scipy.sparse.csr_matrix(
    ...         (chunk.data, chunk.indices, chunk.indptr),
    ...         shape=(chunk.nrows, chunk.n_vars),
    ...     )
    """
    native_stream = open_stream_via_native(
        normalize_path(path),
        chunk_size=chunk_size,
        assay=assay,
        layer=layer,
    )
    if native_stream is None:
        raise RuntimeError(
            "open_stream() requires the native backend. "
            "Install with: pip install picklerick[native]"
        )
    for native_chunk in native_stream:
        yield MatrixChunk(native_chunk)


__all__ = [
    "MatrixChunk",
    "convert",
    "inspect",
    "open_stream",
    "read",
    "read_dataset",
    "read_h5ad",
    "read_h5seurat",
    "write_h5ad",
    "write_h5seurat",
]