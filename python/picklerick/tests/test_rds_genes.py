"""Gene blocks straight from a SpatialExperiment .rds (no gene-major copy).

visiumhd_spe.rds: 5000 bins x 30 genes of a 10x Visium HD SpatialExperiment
(scripts/make_spe_fixtures.R); visiumhd_spe.json holds what R read back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

import picklerick as pk

from conftest import REPO

TINY = REPO / "tests" / "fixtures" / "tiny"
RDS = TINY / "visiumhd_spe.rds"


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((TINY / "visiumhd_spe.json").read_text())


def test_rds_streams_by_cell_and_by_gene(expected) -> None:
    by_cell = sparse.vstack([c.to_scipy() for c in pk.open_stream(RDS, chunk_size=777)])
    assert by_cell.shape == (5000, 30)
    np.testing.assert_array_equal(by_cell.sum(axis=0), expected["gene_sums"])
    np.testing.assert_array_equal(by_cell.sum(axis=1), expected["cell_sums"])

    chunks = list(pk.open_stream(RDS, axis="var", chunk_size=8))
    assert [(c.var_offset, c.ncols) for c in chunks] == [(0, 8), (8, 8), (16, 8), (24, 6)]
    by_gene = sparse.hstack([c.to_scipy() for c in chunks])
    assert (by_gene != by_cell).nnz == 0


def test_rds_gene_ranges(expected) -> None:
    full = sparse.vstack([c.to_scipy() for c in pk.open_stream(RDS)]).toarray()
    chunks = list(pk.open_stream(RDS, axis="var", genes=[(20, 25), (3, 9)], chunk_size=4))
    assert [(c.var_offset, c.ncols) for c in chunks] == [(3, 4), (7, 2), (20, 4), (24, 1)]
    got = sparse.hstack([c.to_scipy() for c in chunks]).toarray()
    np.testing.assert_array_equal(got, full[:, list(range(3, 9)) + list(range(20, 25))])


def test_rds_inspect_reads_the_spatial_experiment(expected) -> None:
    info = pk.inspect(RDS, names=True)
    assert (info["n_obs"], info["n_vars"]) == (5000, 30)
    assert info["x_axis"] == "obs"
    assert info["var_names"] == expected["var_names"]
    assert "spatial" in info["obsm_keys"]
