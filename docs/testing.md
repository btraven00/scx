# SCX Testing Plan

## Overview

Three layers of tests:

1. **Rust unit + integration tests** (`cargo test`) — format-specific logic,
   sparse matrix arithmetic, IR correctness. No external dependencies.
2. **Python pytest suite** (`pixi run -e test verify-python`) — loads
   SCX-produced files with `anndata` and compares against a reference produced
   by zellkonverter. The Python ecosystem is the primary consumer of H5AD.
3. **R testthat suite** (`pixi run -e test verify-r`) — loads SCX-produced files
   with `anndataR` and validates slot-by-slot against the reference H5AD.

All three run in the `pixi test` environment. The full gate is
`pixi run -e test roundtrip`, which chains: fixtures → convert → pytest → testthat.

---

## Environment setup

```
pixi run -e test fixtures       # generate golden files
pixi run -e test roundtrip      # full gate (includes build)
```

The `test` pixi environment (defined in `pixi.toml`) provides:
- R ≥ 4.4, Seurat ≥ 5, SeuratDisk, anndataR, testthat, hdf5r, Matrix
- Python ≥ 3.14, anndata ≥ 0.12, h5py, numpy, scipy, pytest
- The `hdf5` system library is inherited from the default environment

The default environment has no R or Python — just `hdf5` and `pkg-config` for
building the Rust crate.

---

## Golden fixtures

All fixtures live in `tests/golden/`. They are **not committed to git** (binary
HDF5 files; listed in `.gitignore`). Regenerate with `pixi run -e test fixtures`.

| File | How generated | Purpose |
|------|--------------|---------|
| `pbmc3k.h5` | `scripts/prepare_test_data.R` | SCX internal H5 schema; used by Rust unit tests |
| `pbmc3k.h5seurat` | `scripts/prepare_h5seurat_test.R` | SeuratDisk H5Seurat; primary read-side golden |
| `pbmc3k_reference.h5ad` | `scripts/prepare_h5ad_reference.R` | zellkonverter output; the oracle for all comparisons |
| `pbmc3k_scx.h5ad` | `pixi run -e test convert-h5seurat-to-h5ad` | SCX output under test |

### `scripts/prepare_h5ad_reference.R` (to create)

Reads `pbmc3k.h5seurat` with `SeuratDisk::LoadH5Seurat`, converts with
`zellkonverter::writeH5AD`, saves as `tests/golden/pbmc3k_reference.h5ad`.
This is the trusted reference for all downstream comparisons.

---

## Rust tests (`cargo test`)

### Existing coverage (keep, do not regress)

- `sparse.rs` — `csc_to_csr` identity, rectangular, slice
- `h5seurat.rs` — shape, obs, var, obsm, stream coverage and nnz count
- `h5ad.rs` — round-trip through SCX-H5 → H5AD, structural validation

### Additions needed (as features are implemented)

| Module | Test name | What it checks |
|--------|-----------|---------------|
| `h5ad.rs` (reader) | `test_read_shape` | n_obs, n_vars from pbmc3k_reference.h5ad |
| `h5ad.rs` (reader) | `test_read_obs` | obs index length, column names, categorical dtype |
| `h5ad.rs` (reader) | `test_read_var` | var index length, column names |
| `h5ad.rs` (reader) | `test_read_obsm` | obsm keys present, shape correct |
| `h5ad.rs` (reader) | `test_stream_coverage` | total cells and nnz across all chunks |
| `h5ad.rs` (reader) | `test_roundtrip_h5ad` | H5AD → IR → H5AD, structural equivalence via hdf5-rs |
| `h5seurat.rs` (writer) | `test_write_shape` | dims attr correct |
| `h5seurat.rs` (writer) | `test_write_obs` | meta.data present, factor groups correct |
| `h5seurat.rs` (writer) | `test_write_obsm` | reductions present, shape correct |
| `h5seurat.rs` (writer) | `test_write_roundtrip` | H5Seurat → IR → H5Seurat → IR, shape + nnz match |
| `sparse.rs` | `test_csr_to_csc_identity` | inverse of existing csc_to_csr test |
| `sparse.rs` | `test_csr_to_csc_rectangular` | 2×3 case |
| `sparse.rs` | `test_csc_csr_roundtrip` | CSC → CSR → CSC identity |

These are pure Rust tests using the golden fixtures. They do not require the
`test` pixi environment.

---

## Python pytest suite (`tests/python/`)

### Structure

```
tests/python/
├── conftest.py          # fixtures: paths, loaded AnnData objects
├── test_shape.py        # shape and index
├── test_obs.py          # obs column presence, dtype class, categorical levels
├── test_var.py          # var column presence and dtype
├── test_matrix.py       # X nnz, row sums, col sums, spot-check values
├── test_obsm.py         # obsm keys, shapes, numeric values (with sign-flip tolerance)
└── test_roundtrip.py    # H5AD → SCX → H5AD → compare (once H5AD reader exists)
```

### `conftest.py`

```python
import pytest, anndata as ad, os

GOLDEN = os.environ.get("SCX_GOLDEN", "tests/golden")

@pytest.fixture(scope="session")
def scx_adata():
    return ad.read_h5ad(f"{GOLDEN}/pbmc3k_scx.h5ad")

@pytest.fixture(scope="session")
def ref_adata():
    return ad.read_h5ad(f"{GOLDEN}/pbmc3k_reference.h5ad")
```

### `test_shape.py`

```python
def test_shape(scx_adata, ref_adata):
    assert scx_adata.shape == ref_adata.shape

def test_obs_names(scx_adata, ref_adata):
    assert list(scx_adata.obs_names) == list(ref_adata.obs_names)

def test_var_names(scx_adata, ref_adata):
    assert list(scx_adata.var_names) == list(ref_adata.var_names)
```

### `test_obs.py`

```python
def test_obs_columns_present(scx_adata, ref_adata):
    missing = set(ref_adata.obs.columns) - set(scx_adata.obs.columns)
    assert not missing, f"obs columns missing: {missing}"

def test_obs_categorical_columns(scx_adata, ref_adata):
    """Columns that are categorical in the reference must be categorical in SCX."""
    for col in ref_adata.obs.columns:
        if hasattr(ref_adata.obs[col].dtype, "categories"):
            assert hasattr(scx_adata.obs[col].dtype, "categories"), \
                f"obs['{col}'] should be categorical"

def test_obs_numeric_columns(scx_adata, ref_adata):
    """Numeric columns must agree within tolerance."""
    import numpy as np
    for col in ref_adata.obs.columns:
        if ref_adata.obs[col].dtype.kind in "fiu":
            np.testing.assert_allclose(
                scx_adata.obs[col].values,
                ref_adata.obs[col].values,
                rtol=1e-4,
                err_msg=f"obs column '{col}' values differ",
            )
```

### `test_matrix.py`

```python
import numpy as np, scipy.sparse as sp

def test_nnz(scx_adata, ref_adata):
    assert scx_adata.X.nnz == ref_adata.X.nnz

def test_row_sums(scx_adata, ref_adata):
    r = np.asarray(ref_adata.X.sum(axis=1)).ravel()
    s = np.asarray(scx_adata.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(s, r, rtol=1e-4)

def test_col_sums(scx_adata, ref_adata):
    r = np.asarray(ref_adata.X.sum(axis=0)).ravel()
    s = np.asarray(scx_adata.X.sum(axis=0)).ravel()
    np.testing.assert_allclose(s, r, rtol=1e-4)

def test_spot_values(scx_adata, ref_adata):
    """Exact value check on first 100 cells × 100 genes submatrix."""
    r = ref_adata.X[:100, :100].toarray()
    s = scx_adata.X[:100, :100].toarray()
    np.testing.assert_allclose(s, r, rtol=1e-4)
```

### `test_obsm.py`

```python
import numpy as np

def test_obsm_keys_present(scx_adata, ref_adata):
    missing = set(ref_adata.obsm.keys()) - set(scx_adata.obsm.keys())
    assert not missing, f"obsm keys missing: {missing}"

@pytest.mark.parametrize("key", ["X_pca", "X_umap"])
def test_obsm_shape(key, scx_adata, ref_adata):
    assert scx_adata.obsm[key].shape == ref_adata.obsm[key].shape

@pytest.mark.parametrize("key", ["X_pca", "X_umap"])
def test_obsm_values(key, scx_adata, ref_adata):
    """Numeric values must agree; allow sign flips on PCA axes."""
    r = np.asarray(ref_adata.obsm[key])
    s = np.asarray(scx_adata.obsm[key])
    close     = np.allclose(r, s,  rtol=1e-4, atol=1e-6)
    close_neg = np.allclose(r, -s, rtol=1e-4, atol=1e-6)
    assert close or close_neg, f"obsm['{key}'] values differ"
```

### `test_roundtrip.py` (stub — enabled once H5AD reader exists)

```python
import pytest, anndata as ad, subprocess, tempfile, pathlib

@pytest.mark.skipif(not H5AD_READER_IMPLEMENTED, reason="H5AD reader not yet implemented")
def test_h5ad_roundtrip(scx_adata, tmp_path):
    """H5AD → SCX convert → H5AD → load → compare."""
    out = tmp_path / "rt.h5ad"
    subprocess.run(["scx", "convert", SCX_H5AD, str(out)], check=True)
    rt = ad.read_h5ad(out)
    assert rt.shape == scx_adata.shape
    assert rt.X.nnz == scx_adata.X.nnz
```

---

## R testthat suite (`tests/r/`)

### Structure

```
tests/r/
├── helper-fixtures.R    # shared: load SCE objects once per session
├── test-shape.R         # dim, colnames, rownames
├── test-coldataR         # colData columns, types, factor levels
├── test-matrix.R        # nnz, row/col sums, spot-check values
├── test-reduceddims.R   # reducedDimNames, shapes, numeric values
└── test-roundtrip.R     # stub: H5AD → SCX → H5Seurat → SCE compare
```

### `helper-fixtures.R`

```r
suppressPackageStartupMessages({
  library(anndataR)
  library(Matrix)
})

GOLDEN     <- Sys.getenv("SCX_GOLDEN", unset = "../golden")
REF_PATH   <- file.path(GOLDEN, "pbmc3k_reference.h5ad")
SCX_PATH   <- file.path(GOLDEN, "pbmc3k_scx.h5ad")

EXPECTED_N_OBS  <- 2700L
EXPECTED_N_VARS <- 13714L
EXPECTED_NNZ    <- 2282976L

skip_if_no_golden <- function(...) {
  paths <- c(...)
  for (p in paths) {
    if (!file.exists(p)) {
      testthat::skip(paste0("golden file not found: ", p,
                            " — run `pixi run -e test fixtures`"))
    }
  }
}

.scx_ad <- NULL
.ref_ad <- NULL

scx_ad <- function() {
  if (is.null(.scx_ad)) { skip_if_no_golden(SCX_PATH); .scx_ad <<- read_h5ad(SCX_PATH) }
  .scx_ad
}

ref_ad <- function() {
  if (is.null(.ref_ad)) { skip_if_no_golden(REF_PATH); .ref_ad <<- read_h5ad(REF_PATH) }
  .ref_ad
}
```

### `test-shape.R`

```r
test_that("n_obs matches expected", {
  skip_if_no_golden(SCX_PATH)
  expect_equal(nrow(scx_ad()$obs), EXPECTED_N_OBS)
})

test_that("shape matches reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(nrow(scx_ad()$obs), nrow(ref_ad()$obs))
  expect_equal(nrow(scx_ad()$var), nrow(ref_ad()$var))
})

test_that("obs names match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_identical(scx_ad()$obs_names, ref_ad()$obs_names)
})

test_that("var names match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_identical(scx_ad()$var_names, ref_ad()$var_names)
})
```

### `test-coldata.R`

```r
test_that("all reference obs columns present", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  missing <- setdiff(colnames(ref_ad()$obs), colnames(scx_ad()$obs))
  expect_length(missing, 0)
})

test_that("factor columns are factors", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  ref_factors <- names(which(sapply(ref_ad()$obs, is.factor)))
  for (col in ref_factors) {
    expect_true(is.factor(scx_ad()$obs[[col]]),
                info = paste("expected factor:", col))
  }
})

test_that("numeric columns agree within tolerance", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  ref_nums <- names(which(sapply(ref_ad()$obs, is.numeric)))
  for (col in ref_nums) {
    expect_equal(scx_ad()$obs[[col]], ref_ad()$obs[[col]],
                 tolerance = 1e-4, info = col)
  }
})
```

### `test-matrix.R`

```r
test_that("X nnz matches expected", {
  skip_if_no_golden(SCX_PATH)
  expect_equal(Matrix::nnzero(scx_ad()$X), EXPECTED_NNZ)
})

test_that("X nnz matches reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(Matrix::nnzero(scx_ad()$X), Matrix::nnzero(ref_ad()$X))
})

test_that("X row sums match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(Matrix::rowSums(scx_ad()$X), Matrix::rowSums(ref_ad()$X),
               tolerance = 1e-4)
})

test_that("X col sums match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(Matrix::colSums(scx_ad()$X), Matrix::colSums(ref_ad()$X),
               tolerance = 1e-4)
})

test_that("spot check: first 100 obs x 100 vars exact", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  s <- as.matrix(scx_ad()$X[1:100, 1:100])
  r <- as.matrix(ref_ad()$X[1:100, 1:100])
  expect_equal(s, r, tolerance = 1e-4)
})
```

### `test-reduceddims.R`

```r
test_that("all reference obsm keys present", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  missing <- setdiff(names(ref_ad()$obsm), names(scx_ad()$obsm))
  expect_length(missing, 0)
})

test_that("X_pca shape correct", {
  skip_if_no_golden(SCX_PATH)
  pca <- scx_ad()$obsm[["X_pca"]]
  expect_equal(nrow(pca), EXPECTED_N_OBS)
  expect_equal(ncol(pca), 30L)
})

test_that("X_pca shapes match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(dim(scx_ad()$obsm[["X_pca"]]), dim(ref_ad()$obsm[["X_pca"]]))
})

test_that("X_pca values agree per-axis (sign-flip tolerant)", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  s <- scx_ad()$obsm[["X_pca"]]
  r <- ref_ad()$obsm[["X_pca"]]
  for (i in seq_len(min(ncol(s), ncol(r), 10L))) {
    aligned <- isTRUE(all.equal(s[, i],  r[, i], tolerance = 1e-3)) ||
               isTRUE(all.equal(s[, i], -r[, i], tolerance = 1e-3))
    expect_true(aligned, info = paste("X_pca axis", i))
  }
})
```

---

## CI integration

Add a GitHub Actions workflow (`.github/workflows/integration.yml`) that:

1. Installs pixi
2. Runs `pixi run -e test fixtures` (generates golden files from scratch)
3. Runs `pixi run test` (Rust tests, default env)
4. Runs `pixi run -e test convert-h5seurat-to-h5ad`
5. Runs `pixi run -e test verify-python` (pytest)
6. Runs `pixi run -e test verify-r` (testthat)

The golden fixtures are not cached between CI runs — they are regenerated each
time. This prevents bit-rot from stale reference files.

---

## What's needed before execution

The following files must be created before `pixi run -e test roundtrip` works:

| File | Status | Notes |
|------|--------|-------|
| `scripts/prepare_h5ad_reference.R` | **to create** | Calls zellkonverter on pbmc3k.h5seurat → reference.h5ad |
| `tests/python/conftest.py` | **to create** | Session-scoped fixtures |
| `tests/python/test_shape.py` | **to create** | |
| `tests/python/test_obs.py` | **to create** | |
| `tests/python/test_var.py` | **to create** | |
| `tests/python/test_matrix.py` | **to create** | |
| `tests/python/test_obsm.py` | **to create** | |
| `tests/r/helper-fixtures.R` | **to create** | |
| `tests/r/test-shape.R` | **to create** | |
| `tests/r/test-coldata.R` | **to create** | |
| `tests/r/test-matrix.R` | **to create** | |
| `tests/r/test-reduceddims.R` | **to create** | |

The `test_roundtrip.py` and `test-roundtrip.R` stubs are created but skipped
until the H5AD reader is implemented (v0.0.2).
