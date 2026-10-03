#!/usr/bin/env python3
"""Write tests/fixtures/tiny/: one 6-cell x 5-gene dataset in every input format.

    expected.json           the dataset itself; tests compare readers against it
    tiny.h5ad               anndata: CSR X, int counts layer, every obs encoding
                            (categorical with NA, nullable int/bool, string, ...),
                            obsm, varm, obsp, nested uns with arrays and bools
    tiny_dense.h5ad         X stored dense, nothing else
    tiny_10x.h5             Cell Ranger v3 HDF5 (counts)
    tiny_mtx/               Cell Ranger v3 MatrixMarket directory (counts, gzipped)
    tiny_v4.h5seurat        SeuratDisk v3/v4 layout: dgCMatrix counts + data
    tiny_v5_bpcells.h5seurat
                            Seurat v5 layout; this script writes names and
                            metadata, scripts/make_tiny_fixtures_bpcells.R adds
                            the BPCells-encoded counts matrix

Row 1 has no counts and gene 2 is expressed in one cell only, so empty rows
and sparse columns are always exercised.

The H5Seurat files are written by hand (h5py) following SeuratDisk's
SaveH5Seurat layout, as in prepare_h5seurat_test.R: SeuratDisk itself no
longer installs alongside Seurat 5.

Deterministic: rerunning produces the same data.

usage: pixi run -e py make-tiny-fixtures
"""

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/fixtures/tiny"

COUNTS = np.array(
    [
        [1, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 4],
        [5, 0, 6, 0, 0],
        [0, 0, 0, 0, 7],
        [8, 9, 0, 0, 0],
    ],
    dtype=np.int32,
)
X = (COUNTS * 0.5).astype(np.float32)  # stands in for normalised data
CELLS = [f"cell{i}" for i in range(6)]
GENES = [f"gene{j}" for j in range(5)]
SYMBOLS = ["CD3E", "MS4A1", "NKG7", "LYZ", "GNLY"]

CELL_TYPE = ["B", "T", None, "B", "NK", "T"]  # None = NA
CELL_TYPE_LEVELS = ["B", "NK", "T"]
BATCH = [1, None, 2, 2, 1, None]  # nullable int
FLAG = [True, None, False, True, None, False]  # nullable bool
SCORE = [0.5, 1.5, float("nan"), -2.0, 0.0, 3.25]
IS_DOUBLET = [False, False, True, False, False, False]
DONOR = ["d0", "d1", "d0", "d1", "d0", "d1"]  # plain strings
N_COUNTS = COUNTS.sum(axis=1).tolist()

PCA = np.arange(12, dtype=np.float32).reshape(6, 2) / 10
PCS = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
CONN = sp.csr_matrix(
    ([1.0, 0.5, 0.5, 1.0], ([0, 2, 3, 5], [3, 5, 0, 2])), shape=(6, 6), dtype=np.float32
)


def write_expected():
    nan_to_none = lambda xs: [None if isinstance(x, float) and np.isnan(x) else x for x in xs]
    expected = {
        "n_obs": 6,
        "n_vars": 5,
        "obs_names": CELLS,
        "var_names": GENES,
        "X": X.tolist(),
        "counts": COUNTS.tolist(),
        "obs": {
            "cell_type": {"type": "categorical", "levels": CELL_TYPE_LEVELS, "values": CELL_TYPE},
            "batch": {"type": "int", "values": BATCH},
            "flag": {"type": "bool", "values": FLAG},
            "score": {"type": "float", "values": nan_to_none(SCORE)},
            "is_doublet": {"type": "bool", "values": IS_DOUBLET},
            "donor": {"type": "string", "values": DONOR},
            "n_counts": {"type": "int", "values": N_COUNTS},
        },
        "var": {"gene_symbol": {"type": "string", "values": SYMBOLS}},
        "obsm": {"X_pca": PCA.tolist()},
        "varm": {"PCs": PCS.tolist()},
        "obsp": {"connectivities": CONN.toarray().tolist()},
        "uns": {
            "title": "tiny",
            "n_pcs": 2,
            "params": {"resolution": 0.5, "use_raw": False},
            "colors": ["red", "blue"],
            "weights": [0.25, 0.75],
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")


def write_h5ad():
    obs = pd.DataFrame(index=CELLS)
    obs["cell_type"] = pd.Categorical(CELL_TYPE, categories=CELL_TYPE_LEVELS)
    obs["batch"] = pd.array(BATCH, dtype="Int32")
    obs["flag"] = pd.array(FLAG, dtype="boolean")
    obs["score"] = np.array(SCORE)
    obs["is_doublet"] = np.array(IS_DOUBLET)
    obs["donor"] = np.array(DONOR, dtype=object)
    obs["n_counts"] = np.array(N_COUNTS, dtype=np.int32)
    var = pd.DataFrame(index=GENES)
    var["gene_symbol"] = np.array(SYMBOLS, dtype=object)

    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    a.layers["counts"] = sp.csr_matrix(COUNTS)
    a.obsm["X_pca"] = PCA
    a.varm["PCs"] = PCS
    a.obsp["connectivities"] = CONN
    a.uns["title"] = "tiny"
    a.uns["n_pcs"] = 2
    a.uns["params"] = {"resolution": 0.5, "use_raw": False}
    a.uns["colors"] = np.array(["red", "blue"], dtype=object)
    a.uns["weights"] = np.array([0.25, 0.75])
    a.write_h5ad(OUT / "tiny.h5ad")

    ad.AnnData(X=X, obs=pd.DataFrame(index=CELLS), var=pd.DataFrame(index=GENES)).write_h5ad(
        OUT / "tiny_dense.h5ad"
    )


def write_10x():
    # Cell Ranger stores features x barcodes CSC, i.e. CSR over cells.
    m = sp.csr_matrix(COUNTS)
    with h5py.File(OUT / "tiny_10x.h5", "w") as f:
        g = f.create_group("matrix")
        g["barcodes"] = np.array(CELLS, dtype="S")
        g["data"] = m.data.astype(np.int32)
        g["indices"] = m.indices.astype(np.int64)
        g["indptr"] = m.indptr.astype(np.int64)
        g["shape"] = np.array([len(GENES), len(CELLS)], dtype=np.int32)
        feat = g.create_group("features")
        feat["id"] = np.array(GENES, dtype="S")
        feat["name"] = np.array(SYMBOLS, dtype="S")
        feat["feature_type"] = np.array(["Gene Expression"] * len(GENES), dtype="S")
        feat["genome"] = np.array(["GRCh38"] * len(GENES), dtype="S")
        feat["_all_tag_keys"] = np.array([b"genome"])


def gz_write(path: Path, data: bytes):
    # mtime=0 keeps the output byte-identical across runs.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(data)


def write_mtx():
    d = OUT / "tiny_mtx"
    d.mkdir(exist_ok=True)
    tmp = d / "matrix.mtx"
    # genes x cells, entries ordered by cell as Cell Ranger writes them
    # (scx streams MatrixMarket by cell and needs that order).
    cells, genes = np.nonzero(COUNTS)
    coo = sp.coo_matrix((COUNTS[cells, genes], (genes, cells)), shape=COUNTS.T.shape)
    scipy.io.mmwrite(tmp, coo, field="integer")
    gz_write(d / "matrix.mtx.gz", tmp.read_bytes())
    tmp.unlink()
    gz_write(d / "barcodes.tsv.gz", "".join(f"{c}\n" for c in CELLS).encode())
    gz_write(
        d / "features.tsv.gz",
        "".join(f"{g}\t{s}\tGene Expression\n" for g, s in zip(GENES, SYMBOLS)).encode(),
    )


def write_seurat_common(f: h5py.File, version: str):
    """Names, metadata and attributes shared by both H5Seurat layouts."""
    str_dt = h5py.string_dtype()
    f.attrs["active.assay"] = "RNA"
    f.attrs["version"] = version
    f.attrs["project"] = "tiny"
    f.create_dataset("cell.names", data=CELLS, dtype=str_dt)
    rna = f.create_group("assays/RNA")
    rna.attrs["key"] = "rna_"
    rna.create_dataset("features", data=GENES, dtype=str_dt)

    meta = f.create_group("meta.data")
    meta.attrs["_class"] = "data.frame"
    cols = ["cell_type", "batch", "flag", "score", "is_doublet", "donor", "n_counts"]
    meta.attrs["colnames"] = cols
    na_int = np.iinfo(np.int32).min  # R's NA_integer_
    ct = meta.create_group("cell_type")  # factor: 1-based codes, NA = NA_integer_
    ct["values"] = np.array(
        [na_int if v is None else CELL_TYPE_LEVELS.index(v) + 1 for v in CELL_TYPE], dtype=np.int32
    )
    ct.create_dataset("levels", data=CELL_TYPE_LEVELS, dtype=str_dt)
    meta["batch"] = np.array([na_int if v is None else v for v in BATCH], dtype=np.int32)
    # SeuratDisk logicals: 0 = FALSE, 1 = TRUE, 2 = NA, listed in the `logicals` attr.
    meta["flag"] = np.array([2 if v is None else int(v) for v in FLAG], dtype=np.int32)
    meta["is_doublet"] = np.array(IS_DOUBLET, dtype=np.int32)
    meta.attrs["logicals"] = ["flag", "is_doublet"]
    meta["score"] = np.array(SCORE)
    donor = meta.create_group("donor")  # character columns are saved as factors
    levels = sorted(set(DONOR))
    donor["values"] = np.array([levels.index(v) + 1 for v in DONOR], dtype=np.int32)
    donor.create_dataset("levels", data=levels, dtype=str_dt)
    meta["n_counts"] = np.array(N_COUNTS, dtype=np.float64)
    return rna


def write_dgc(grp: h5py.Group, dense: np.ndarray, dtype):
    # dgCMatrix is genes x cells CSC == cells x genes CSR.
    m = sp.csr_matrix(dense)
    grp["data"] = m.data.astype(dtype)
    grp["indices"] = m.indices.astype(np.int32)
    grp["indptr"] = m.indptr.astype(np.int32)
    grp.attrs["dims"] = np.array([len(GENES), len(CELLS)], dtype=np.int32)


def write_h5seurat():
    with h5py.File(OUT / "tiny_v4.h5seurat", "w") as f:
        rna = write_seurat_common(f, "4.4.0")
        write_dgc(rna.create_group("counts"), COUNTS, np.float64)
        write_dgc(rna.create_group("data"), X, np.float64)
        pca = f.create_group("reductions/pca")
        # R matrices are column-major: an (ncells, k) matrix lands as (k, ncells).
        pca["cell.embeddings"] = PCA.T.astype(np.float64)
        pca.attrs["key"] = "PC_"
        pca.attrs["active.assay"] = "RNA"

    with h5py.File(OUT / "tiny_v5_bpcells.h5seurat", "w") as f:
        write_seurat_common(f, "5.0.0")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_expected()
    write_h5ad()
    write_10x()
    write_mtx()
    write_h5seurat()
    print(f"wrote {OUT}; now run make-tiny-fixtures-bpcells")


if __name__ == "__main__":
    main()
