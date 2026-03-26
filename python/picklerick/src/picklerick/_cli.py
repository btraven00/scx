from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from ._exceptions import PickleRickError, ScxCommandError, ScxNotFoundError


def find_scx_binary() -> str:
    """
    Locate the SCX CLI binary.

    Resolution order:
    1. ``SCX_BIN`` environment variable
    2. ``scx`` on ``PATH``
    """
    env_bin = os.environ.get("SCX_BIN", "").strip()
    if env_bin:
        candidate = Path(env_bin).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise ScxNotFoundError(
            f"SCX binary specified by SCX_BIN was not found: {candidate}"
        )

    which = shutil.which("scx")
    if which:
        return which

    raise ScxNotFoundError(
        "Could not find the SCX CLI binary. Set SCX_BIN or put `scx` on PATH."
    )


def run_scx(
    args: Sequence[str],
    *,
    binary: str | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run the SCX CLI with the given arguments.

    Parameters
    ----------
    args:
        Arguments passed after the ``scx`` executable.
    binary:
        Optional explicit path to the ``scx`` binary.
    cwd:
        Optional working directory.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed subprocess result.

    Raises
    ------
    ScxNotFoundError
        If the SCX executable cannot be found.
    ScxCommandError
        If the command exits with a non-zero status.
    """
    exe = binary or find_scx_binary()
    command = [exe, *map(str, args)]

    try:
        result = subprocess.run(
            command,
            cwd=str(Path(cwd).expanduser()) if cwd is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ScxNotFoundError(f"SCX binary not found: {exe}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "no output captured"
        raise ScxCommandError(
            f"SCX command failed with exit code {result.returncode}: {detail}",
            command=command,
            stderr=result.stderr or "",
        )

    return result


def convert_via_scx(
    input_path: str | Path,
    output_path: str | Path,
    *,
    chunk_size: int = 5000,
    dtype: str = "f32",
    assay: str = "RNA",
    layer: str = "counts",
    binary: str | None = None,
) -> None:
    """
    Run ``scx convert`` with picklerick-compatible defaults.
    """
    run_scx(
        [
            "convert",
            str(Path(input_path).expanduser()),
            str(Path(output_path).expanduser()),
            "--chunk-size",
            str(int(chunk_size)),
            "--dtype",
            str(dtype),
            "--assay",
            str(assay),
            "--layer",
            str(layer),
        ],
        binary=binary,
    )


convert_file = convert_via_scx

__all__ = [
    "PickleRickError",
    "ScxCommandError",
    "ScxNotFoundError",
    "convert_file",
    "convert_via_scx",
    "find_scx_binary",
    "run_scx",
]