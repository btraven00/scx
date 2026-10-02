"""Assert picklerick's Rust read path reproduces the oracle-extracted digests.

The golden JSON in tests/golden/properties/ was produced by anndata (see
scripts/extract_golden_properties.py). Here we read the SAME fixture through
picklerick's native streaming decode (open_stream — the Rust path, incl. the
future parallel-inflate decode) and assert the canonical X digest matches
bit-for-bit. This is the differential-correctness gate: a single wrong value
changes the hash.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import picklerick as pk

from conftest import golden
from golden_props import digest_matrix, matrix_from_stream_chunks

PROP_DIR = Path(__file__).resolve().parents[3] / "tests" / "golden" / "properties"

# fixture stem -> path relative to golden root
H5AD_CASES = {
    "pbmc3k_reference": "pbmc3k_reference.h5ad",
    "norman_subset": "norman_subset.h5ad",
    "hlca_core": "hlca_core.h5ad",
}


def _load_props(stem: str) -> dict | None:
    p = PROP_DIR / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


@pytest.mark.parametrize("stem,rel", list(H5AD_CASES.items()))
def test_open_stream_matches_oracle_x_digest(stem: str, rel: str) -> None:
    # The digest materialises the whole matrix (~40 GB RSS for HLCA).
    if stem == "hlca_core" and not os.getenv("SCX_TEST_LARGE"):
        pytest.skip("HLCA needs ~40 GB RAM; set SCX_TEST_LARGE=1")
    props = _load_props(stem)
    if props is None:
        pytest.skip(f"no golden properties for {stem} (run extract_golden_properties.py)")
    path = golden(rel)

    chunks = list(pk.open_stream(path, chunk_size=5000))
    n_vars = props["n_vars"]
    assert chunks, "stream produced no chunks"
    assert chunks[0].n_vars == n_vars

    x = matrix_from_stream_chunks(chunks, n_vars)
    got = digest_matrix(x)

    assert got["shape"] == props["X"]["shape"], "X shape differs from oracle"
    assert got["nnz"] == props["X"]["nnz"], "X nnz differs from oracle"
    assert got["digest"] == props["X"]["digest"], (
        f"X content digest differs from anndata oracle for {stem}: "
        f"oracle row0 spot={props['X']['row0_spot']} got={got['row0_spot']}"
    )
