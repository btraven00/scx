from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from ._cli import convert_via_scx
from ._io import read_h5ad_file, write_h5ad_file
from ._native import convert_via_native, write_h5seurat_via_native
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

    This first writes a temporary H5AD and then converts it through the
    optional native backend when available, otherwise through the SCX CLI.
    """
    output_path = ensure_parent_directory(path)

    with TemporaryDirectory(prefix="picklerick-") as tmpdir:
        tmp_h5ad = Path(tmpdir) / "write_h5seurat_tmp.h5ad"
        write_h5ad_file(adata, tmp_h5ad, compression="gzip")

        used_native = write_h5seurat_via_native(
            input_h5ad=tmp_h5ad,
            output_h5seurat=output_path,
            chunk_size=chunk_size,
            assay=assay,
        )
        if not used_native:
            convert_via_scx(
                input_path=tmp_h5ad,
                output_path=output_path,
                chunk_size=chunk_size,
                dtype="f32",
                assay=assay,
                layer="counts",
            )

    return output_path


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
        convert_via_scx(
            input_path=input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            dtype=dtype,
            assay=assay,
            layer=layer,
        )

    return output_path


__all__ = [
    "convert",
    "read",
    "read_dataset",
    "read_h5ad",
    "read_h5seurat",
    "write_h5ad",
    "write_h5seurat",
]