from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

EXPECTED_N_OBS = 2700
EXPECTED_N_VARS = 13714
EXPECTED_NNZ = 2282976


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _golden_root() -> Path:
    env = os.getenv("SCX_GOLDEN")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "tests" / "golden"


def _scx_bin() -> str | None:
    env = os.getenv("SCX_BIN")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return str(p)

    found = shutil.which("scx")
    if found:
        return found

    candidate = _repo_root() / "target" / "release" / "scx"
    if candidate.exists():
        return str(candidate)

    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_scx: mark test as requiring the scx CLI binary",
    )
    config.addinivalue_line(
        "markers",
        "requires_fixtures: mark test as requiring golden input fixtures",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture(scope="session")
def golden_root() -> Path:
    return _golden_root()


@pytest.fixture(scope="session")
def h5seurat_path(golden_root: Path) -> Path:
    return golden_root / "pbmc3k.h5seurat"


@pytest.fixture(scope="session")
def h5ad_ref_path(golden_root: Path) -> Path:
    return golden_root / "pbmc3k_reference.h5ad"


@pytest.fixture(scope="session")
def expected_n_obs() -> int:
    return EXPECTED_N_OBS


@pytest.fixture(scope="session")
def expected_n_vars() -> int:
    return EXPECTED_N_VARS


@pytest.fixture(scope="session")
def expected_nnz() -> int:
    return EXPECTED_NNZ


@pytest.fixture(scope="session")
def scx_bin() -> str | None:
    return _scx_bin()


@pytest.fixture
def require_fixtures(h5seurat_path: Path, h5ad_ref_path: Path) -> None:
    missing = [str(p) for p in (h5seurat_path, h5ad_ref_path) if not p.exists()]
    if missing:
        joined = ", ".join(missing)
        pytest.skip(f"golden fixture not found: {joined} — run `pixi run -e test fixtures`")


@pytest.fixture
def require_scx(scx_bin: str | None) -> str:
    if not scx_bin:
        pytest.skip("scx not available — set SCX_BIN, put scx on PATH, or build target/release/scx")
    return scx_bin