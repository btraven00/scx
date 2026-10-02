from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]

# Tracked in git: these tests always run.
NORMAN = REPO / "tests" / "fixtures" / "norman_subset.h5ad"
BPCELLS_CSR = REPO / "tests" / "golden" / "bpcells" / "synth_packed_uint_csr"
ZARR = REPO / "tests" / "fixtures" / "zarr" / "small_v2.zarr"

# Generated locally (`pixi run -e test fixtures`); tests using them skip without.
GOLDEN = Path(os.getenv("SCX_GOLDEN", REPO / "tests" / "golden"))


def dense(m) -> np.ndarray:
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def golden(name: str) -> Path:
    path = GOLDEN / name
    if not path.exists():
        pytest.skip(f"golden fixture missing: {path} (run `pixi run -e test fixtures`)")
    return path


@pytest.fixture(scope="session")
def norman_path() -> Path:
    return NORMAN


@pytest.fixture(scope="session")
def norman() -> ad.AnnData:
    return ad.read_h5ad(NORMAN)


@pytest.fixture(scope="session")
def bpcells_csr_path() -> Path:
    return BPCELLS_CSR


@pytest.fixture(scope="session")
def zarr_path() -> Path:
    return ZARR
