"""Thin Python bindings for the SCX interoperability engine."""

from ._api import (
    convert,
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
    "PickleRickError",
    "ScxCommandError",
    "ScxNotFoundError",
    "convert",
    "native_available",
    "read",
    "read_dataset",
    "read_h5ad",
    "read_h5seurat",
    "write_h5ad",
    "write_h5seurat",
]