from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad


def read_h5ad_file(path: str | Path) -> ad.AnnData:
    """Read an H5AD file into an AnnData object."""
    return ad.read_h5ad(str(Path(path).expanduser()))


def write_h5ad_file(
    adata: ad.AnnData,
    path: str | Path,
    *,
    compression: str = "gzip",
    **kwargs: Any,
) -> Path:
    """Write an AnnData object to H5AD and return the output path."""
    out = Path(path).expanduser()
    adata.write_h5ad(str(out), compression=compression, **kwargs)
    return out