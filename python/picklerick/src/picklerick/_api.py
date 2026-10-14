from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np

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
    x_slot: str = "X",
    chunk_size: int = 5000,
) -> Path:
    """Convert a single-cell dataset to ``.h5ad``, ``.h5seurat`` or ``.zarr``.

    The input format is detected from the file (H5AD, H5Seurat, BPCells,
    10x HDF5, MatrixMarket, AnnData Zarr). The output format is chosen by
    the extension of ``output``; ``.zarr`` is an AnnData Zarr v3 store. The matrix is streamed in chunks of
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
    x_slot
        Where the matrix goes in ``.h5ad`` output: ``"X"`` (default) or
        ``"layers/<name>"``, which writes it there and leaves ``X`` absent.
        A source layer of that name is dropped. Applies to ``.h5ad`` and
        ``.zarr`` output; ignored for ``.h5seurat``.
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
    _native.scx_convert(_path(input), str(out), chunk_size, dtype, assay, layer, x_slot)
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


def inspect(path: Pathish, *, names: bool = False) -> dict:
    """Describe a dataset without reading its matrix.

    Returns a dict with ``format``, ``n_obs``, ``n_vars``, ``obs_cols``,
    ``obs_dtypes``, ``var_cols``, ``var_dtypes``, ``obsm_keys``,
    ``varm_keys``, ``uns_keys``, ``layers`` and ``obsp`` (lists of dicts
    with name, shape and per-row nnz quartiles), and ``x_stats`` when X is
    stored as CSR. With ``names=True`` it also has ``obs_names`` and
    ``var_names`` (lists of str), e.g. to key per-cell results from
    :func:`open_stream` without opening the file with anndata.
    """
    return _native.scx_inspect(_path(path), names)


Rows = Sequence[tuple[int, int]] | Sequence[int] | np.ndarray


def open_stream(
    path: Pathish,
    *,
    chunk_size: int = 5000,
    assay: str = "RNA",
    layer: str = "counts",
    rows: Rows | None = None,
    where: Mapping[str, object] | None = None,
) -> Iterator[MatrixChunk]:
    """Iterate over the matrix X in chunks of ``chunk_size`` rows.

    Decoding runs on a background thread a few chunks ahead of the
    consumer, so peak memory is bounded by the chunk size, not the file.

    ``rows`` and ``where`` select rows; with both, a row must satisfy both.
    On h5ad and Zarr only the selected rows are read (the row pointer is
    loaded, then just their slices of ``data``/``indices``), so filtering
    costs what the matching rows cost. Chunks never span a gap in the
    selection, and ``chunk.row_offset`` is each chunk's first row in the
    full matrix.

    Parameters
    ----------
    rows
        A boolean mask over all rows, a sequence of row indices, or a
        sequence of ``(start, stop)`` half-open ranges.
    where
        ``{column: value}`` or ``{column: [values]}`` over ``obs``: a row
        is kept when every column's value is one of the given ones. Values
        are compared as text (``str(v)``, booleans as ``True``/``False``);
        NA never matches.

    Examples
    --------
    >>> for chunk in pk.open_stream("atlas.h5ad", chunk_size=5000):
    ...     X = chunk.to_scipy()  # csr_array sharing chunk.data / chunk.indices
    >>> a549 = pk.open_stream("emt.h5ad", where={"CellLine": "A549", "Time": ["3d", "7d"]})
    """
    filters = None
    if where is not None:
        filters = [
            (col, [str(v) for v in (vals if isinstance(vals, (list, tuple, set)) else [vals])])
            for col, vals in where.items()
        ]
    return _native.scx_open_stream(
        _path(path), chunk_size, assay, layer, _ranges(rows), filters
    )



def _ranges(rows: Rows | None) -> list[tuple[int, int]] | None:
    """``rows`` as sorted, merged ``(start, stop)`` ranges."""
    if rows is None:
        return None
    arr = np.asarray(rows)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(int(a), int(b)) for a, b in arr]
    if arr.dtype == bool:
        idx = np.flatnonzero(arr)
    else:
        idx = np.unique(arr.astype(np.int64))
        if idx.size and idx[0] < 0:
            raise ValueError("row indices must be non-negative")
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate(([idx[0]], idx[breaks]))
    stops = np.concatenate((idx[breaks - 1] + 1, [idx[-1] + 1]))
    return [(int(a), int(b)) for a, b in zip(starts, stops)]
