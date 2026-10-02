"""Python bindings for the SCX single-cell format conversion engine."""

from importlib.metadata import version

from ._api import MatrixChunk, convert, inspect, open_stream, read, write_h5seurat
from .picklerick_py_native import PickleRickError

__version__ = version("scx-picklerick")

__all__ = [
    "MatrixChunk",
    "PickleRickError",
    "__version__",
    "convert",
    "inspect",
    "open_stream",
    "read",
    "write_h5seurat",
]
