"""Thin Python bindings for the SCX interoperability engine."""

from ._api import (
    MatrixChunk,
    convert,
    inspect,
    open_stream,
    read,
    read_dataset,
    read_h5ad,
    read_h5seurat,
    write_h5ad,
    write_h5seurat,
)
from ._exceptions import PickleRickError, ScxCommandError, ScxNotFoundError
from ._native import native_available

__all__ = [
    "MatrixChunk",
    "PickleRickError",
    "ScxCommandError",
    "ScxNotFoundError",
    "convert",
    "inspect",
    "native_available",
    "open_stream",
    "read",
    "read_dataset",
    "read_h5ad",
    "read_h5seurat",
    "write_h5ad",
    "write_h5seurat",
]