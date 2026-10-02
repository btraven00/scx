from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad

from . import picklerick_py_native as _native
from .picklerick_py_native import MatrixChunk

Pathish = str | os.PathLike[str]

_DTYPES = ("f32", "f64", "i32", "u32")


def _path(p: Pathish) -> str:
    return str(Path(p).expanduser())


def convert(
    input: Pathish,
    output: Pathish,
    *,
    dtype: str = "f32",
    assay: str = "RNA",
    layer: str = "counts",
    chunk_size: int = 5000,
) -> Path:
    """Convert a single-cell dataset to ``.h5ad`` or ``.h5seurat``.

    The input format is detected from the file (H5AD, H5Seurat, BPCells,
    10x HDF5, MatrixMarket, AnnData Zarr). The output format is chosen by
    the extension of ``output``. The matrix is streamed in chunks of
    ``chunk_size`` cells, so peak memory does not grow with the cell count.

    Parameters
    ----------
    input, output
        Source and destination paths.
    dtype
        Value type of the written matrix: ``"f32"``, ``"f64"``, ``"i32"``
        or ``"u32"``.
    assay, layer
        Seurat assay and layer to read from H5Seurat input; ``assay`` is
        also the assay name written to H5Seurat output.
    chunk_size
        Cells per streamed chunk.

    Returns
    -------
    Path
        ``output``.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}: use one of {_DTYPES}")
    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    _native.scx_convert(_path(input), str(out), chunk_size, dtype, assay, layer)
    return out


def read(
    path: Pathish,
    *,
    assay: str = "RNA",
    layer: str = "counts",
    dtype: str = "f32",
    chunk_size: int = 5000,
) -> ad.AnnData:
    """Read any supported dataset into memory as an AnnData object.

    H5AD files are read with :func:`anndata.read_h5ad` directly and the
    other arguments are ignored. Other formats are first converted to a
    temporary H5AD (see :func:`convert`).
    """
    if Path(path).suffix.lower() == ".h5ad":
        return ad.read_h5ad(_path(path))
    with TemporaryDirectory(prefix="picklerick-") as tmp:
        h5ad = convert(
            path,
            Path(tmp) / "read.h5ad",
            dtype=dtype,
            assay=assay,
            layer=layer,
            chunk_size=chunk_size,
        )
        return ad.read_h5ad(h5ad)


def write_h5seurat(
    adata: ad.AnnData,
    path: Pathish,
    *,
    assay: str = "RNA",
    chunk_size: int = 5000,
) -> Path:
    """Write an AnnData object to H5Seurat (SeuratDisk layout).

    The object is written to a temporary H5AD first and then converted.
    """
    with TemporaryDirectory(prefix="picklerick-") as tmp:
        h5ad = Path(tmp) / "write.h5ad"
        adata.write_h5ad(h5ad)
        return convert(h5ad, path, assay=assay, chunk_size=chunk_size)


def inspect(path: Pathish) -> dict:
    """Describe a dataset without reading its matrix.

    Returns a dict with ``format``, ``n_obs``, ``n_vars``, ``obs_cols``,
    ``obs_dtypes``, ``var_cols``, ``var_dtypes``, ``obsm_keys``,
    ``varm_keys``, ``uns_keys``, ``layers`` and ``obsp`` (lists of dicts
    with name, shape and per-row nnz quartiles), and ``x_stats`` when X is
    stored as CSR.
    """
    return _native.scx_inspect(_path(path))


def open_stream(
    path: Pathish,
    *,
    chunk_size: int = 5000,
    assay: str = "RNA",
    layer: str = "counts",
) -> Iterator[MatrixChunk]:
    """Iterate over the matrix X in chunks of ``chunk_size`` rows.

    Decoding runs on a background thread a few chunks ahead of the
    consumer, so peak memory is bounded by the chunk size, not the file.

    Examples
    --------
    >>> for chunk in pk.open_stream("atlas.h5ad", chunk_size=5000):
    ...     X = scipy.sparse.csr_matrix(
    ...         (chunk.data, chunk.indices, chunk.indptr),
    ...         shape=(chunk.nrows, chunk.n_vars),
    ...     )
    """
    return _native.scx_open_stream(_path(path), chunk_size, assay, layer)
