from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


_NATIVE_MODULE_NAME = "picklerick_py_native"


def _load_native_module() -> Any | None:
    """
    Attempt to import the optional native extension module.

    Returns
    -------
    module or None
        The imported native module if available, otherwise ``None``.
    """
    try:
        return import_module(_NATIVE_MODULE_NAME)
    except ImportError:
        return None


_NATIVE = _load_native_module()


def native_available() -> bool:
    """
    Return ``True`` when the optional native backend is importable.
    """
    return _NATIVE is not None


def convert_via_native(
    input_path: str | Path,
    output_path: str | Path,
    *,
    chunk_size: int = 5000,
    dtype: str = "f32",
    assay: str = "RNA",
    layer: str = "counts",
) -> bool:
    """
    Attempt to run conversion through the optional native backend.

    Parameters
    ----------
    input_path:
        Input dataset path.
    output_path:
        Output dataset path.
    chunk_size:
        Number of cells per streaming chunk.
    dtype:
        Output numeric dtype.
    assay:
        Seurat assay name.
    layer:
        Seurat layer name.

    Returns
    -------
    bool
        ``True`` if the native backend handled the request, ``False`` if the
        native backend is unavailable.

    Notes
    -----
    Any exception raised by the native module is allowed to propagate to the
    caller. This helper only handles the "module not installed" case.
    """
    if _NATIVE is None:
        return False

    _NATIVE.scx_convert_native(
        str(Path(input_path).expanduser()),
        str(Path(output_path).expanduser()),
        int(chunk_size),
        str(dtype),
        str(assay),
        str(layer),
    )
    return True


def write_h5seurat_via_native(
    input_h5ad: str | Path,
    output_h5seurat: str | Path,
    *,
    chunk_size: int = 5000,
    assay: str = "RNA",
) -> bool:
    """
    Attempt to write H5Seurat through the optional native backend.

    Parameters
    ----------
    input_h5ad:
        Input temporary H5AD path.
    output_h5seurat:
        Output H5Seurat path.
    chunk_size:
        Number of cells per streaming chunk.
    assay:
        Seurat assay name to write.

    Returns
    -------
    bool
        ``True`` if the native backend handled the request, ``False`` if the
        native backend is unavailable.
    """
    if _NATIVE is None:
        return False

    _NATIVE.scx_write_h5seurat_native(
        str(Path(input_h5ad).expanduser()),
        str(Path(output_h5seurat).expanduser()),
        int(chunk_size),
        str(assay),
    )
    return True


__all__ = [
    "convert_via_native",
    "native_available",
    "write_h5seurat_via_native",
]