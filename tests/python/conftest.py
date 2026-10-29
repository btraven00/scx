import os
import shutil
import subprocess
import tomllib

import pytest
import anndata as ad

GOLDEN = os.environ.get("SCX_GOLDEN", "tests/golden")
REF_PATH = os.path.join(GOLDEN, "pbmc3k_reference.h5ad")
SCX_PATH = os.path.join(GOLDEN, "pbmc3k_scx.h5ad")

EXPECTED_N_OBS  = 2700
EXPECTED_N_VARS = 13714
EXPECTED_NNZ    = 2282976


def _skip_if_missing(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"golden file not found: {p} — run `pixi run -e test fixtures`")


@pytest.fixture(scope="session")
def ref_adata():
    _skip_if_missing(REF_PATH)
    return ad.read_h5ad(REF_PATH)


@pytest.fixture(scope="session")
def scx_adata():
    _skip_if_missing(SCX_PATH)
    return ad.read_h5ad(SCX_PATH)


# The checkout's own build beats a stale `scx` on PATH: SCX_BIN, else the
# newer of target/{release,debug}, else PATH.
def _find_scx():
    if os.environ.get("SCX_BIN"):
        return os.environ["SCX_BIN"]
    built = [p for p in ("target/release/scx", "target/debug/scx") if os.path.exists(p)]
    return max(built, key=os.path.getmtime) if built else shutil.which("scx")


@pytest.fixture(scope="session")
def scx():
    """Path to an scx binary whose version matches this checkout."""
    path = _find_scx()
    if path is None:
        pytest.skip("scx binary not found — run `cargo build` or set SCX_BIN")
    with open("crates/scx-cli/Cargo.toml", "rb") as f:
        want = tomllib.load(f)["package"]["version"]
    got = subprocess.run([path, "--version"], capture_output=True, text=True)
    # `scx 0.5.0 (-net +zarr)`; pre-0.5 builds reject --version outright.
    have = got.stdout.split()[1] if got.returncode == 0 else "unknown"
    if have != want:
        pytest.fail(f"{path} is scx {have}, checkout is {want} — rebuild or set SCX_BIN")
    return path
