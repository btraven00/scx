from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

Pathish = str | os.PathLike[str]


def to_path(path: Pathish) -> Path:
    """Normalize a user-supplied path-like value to a ``Path``."""
    return Path(path).expanduser()


def normalize_path(path: Pathish) -> Path:
    """Backward-compatible alias for path normalization."""
    return to_path(path)


def file_ext(path: Pathish) -> str:
    """Return the lowercase file extension without the leading dot."""
    return to_path(path).suffix.lower().lstrip(".")


def is_h5ad_path(path: Pathish) -> bool:
    """Return ``True`` when the path looks like an H5AD file."""
    return file_ext(path) == "h5ad"


def ensure_parent_directory(path: Pathish) -> Path:
    """Ensure the parent directory for ``path`` exists and return the path."""
    p = to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def temporary_h5ad_path(
    prefix: str = "picklerick-",
    name: str = "tmp.h5ad",
) -> Iterator[Path]:
    """Yield a temporary ``.h5ad`` path inside a private temporary directory."""
    with TemporaryDirectory(prefix=prefix) as tmpdir:
        yield Path(tmpdir) / name