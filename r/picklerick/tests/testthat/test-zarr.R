# AnnData Zarr stores read through scx-core's Zarr reader. The contract is
# "the same as reading the same AnnData as h5ad": the fixtures are one AnnData
# written by anndata as h5ad, Zarr v2 and Zarr v3 (sharded), see
# scripts/prepare_zarr_fixtures.py in the scx repo.

zarr_fixture <- function(name) {
  p <- file.path(SCX_ROOT, "tests", "fixtures", "zarr", name)
  skip_if_no_fixture(p)
  p
}

for (version in c("v2", "v3")) {
  test_that(sprintf("Zarr %s reads into the same SingleCellExperiment as the h5ad", version), {
    skip_if_not_installed("SingleCellExperiment")
    z <- read_zarr(zarr_fixture(sprintf("small_%s.zarr", version)))
    h <- read_h5ad(zarr_fixture("small.h5ad"))
    expect_identical(dim(z), dim(h))
    expect_identical(dimnames(z), dimnames(h))
    expect_identical(SummarizedExperiment::assayNames(z), SummarizedExperiment::assayNames(h))
    for (a in SummarizedExperiment::assayNames(h)) {
      expect_identical(SummarizedExperiment::assay(z, a), SummarizedExperiment::assay(h, a))
    }
    expect_equal(SummarizedExperiment::colData(z), SummarizedExperiment::colData(h))
    expect_equal(SummarizedExperiment::rowData(z), SummarizedExperiment::rowData(h))
    expect_equal(SingleCellExperiment::reducedDims(z), SingleCellExperiment::reducedDims(h))
  })

  test_that(sprintf("Zarr %s reads into the same Seurat object as the h5ad, both constructors", version), {
    skip_if_not_installed("Seurat")
    for (constructor in c("standard", "fast")) {
      z <- read_zarr(zarr_fixture(sprintf("small_%s.zarr", version)), as = "Seurat",
                     constructor = constructor)
      h <- read_h5ad(zarr_fixture("small.h5ad"), as = "Seurat", constructor = constructor)
      expect_identical(SeuratObject::Cells(z), SeuratObject::Cells(h))
      expect_identical(SeuratObject::Features(z), SeuratObject::Features(h))
      expect_identical(SeuratObject::LayerData(z, "counts"), SeuratObject::LayerData(h, "counts"))
      expect_equal(z[[]], h[[]])
    }
  })
}

test_that("read_h5ad() detects a Zarr store by content too", {
  skip_if_not_installed("SingleCellExperiment")
  p <- zarr_fixture("small_v3.zarr")
  expect_identical(SummarizedExperiment::assay(read_h5ad(p)),
                   SummarizedExperiment::assay(read_zarr(p)))
  expect_identical(read_zarr(p, as = "list")$format, "Zarr (AnnData)")
})

test_that("uns of a Zarr store is available through parse_uns = TRUE only", {
  skip_if_not_installed("SingleCellExperiment")
  z <- read_zarr(zarr_fixture("small_v3.zarr"), parse_uns = TRUE)
  h <- read_h5ad(zarr_fixture("small.h5ad"), parse_uns = TRUE)
  expect_identical(S4Vectors::metadata(z)$title, "scx zarr fixture")
  expect_equal(S4Vectors::metadata(z)$params, S4Vectors::metadata(h)$params)
  # The on-demand accessor reads HDF5, so a directory store records no path.
  expect_null(S4Vectors::metadata(read_zarr(zarr_fixture("small_v3.zarr")))$.uns_path)
  expect_error(uns(read_zarr(zarr_fixture("small_v3.zarr"))), "no source path")
})

test_that("lazy = TRUE stays H5AD-only", {
  skip_if_not_installed("HDF5Array")
  expect_error(read_zarr(zarr_fixture("small_v3.zarr"), lazy = TRUE), "H5AD only")
})

test_that("inspect() reports a Zarr store", {
  info <- inspect(zarr_fixture("small_v3.zarr"))
  expect_identical(info$format, "Zarr (AnnData)")
  expect_identical(c(info$n_obs, info$n_vars), c(60L, 40L))
})
