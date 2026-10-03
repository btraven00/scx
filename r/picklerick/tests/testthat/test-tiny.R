# The tiny fixtures (tests/fixtures/tiny/ in the scx repo): one 6-cell x
# 5-gene dataset in every input format, described by expected.json and written
# by scripts/make_tiny_fixtures.py. They are tracked in git, so these tests run
# in CI without generated golden data.
#
# Tests that start with skip("bug: ...") document known bugs; drop the skip
# when fixing them.

tiny <- function(name) {
  p <- file.path(SCX_ROOT, "tests", "fixtures", "tiny", name)
  skip_if_no_fixture(p)
  p
}

expected <- function() jsonlite::fromJSON(tiny("expected.json"))

test_that("read_h5ad returns X, names and embeddings of the tiny h5ad", {
  skip_if_not_installed("SingleCellExperiment")
  e <- expected()
  sce <- read_h5ad(tiny("tiny.h5ad"))
  expect_identical(dim(sce), c(5L, 6L))
  expect_identical(colnames(sce), e$obs_names)
  expect_identical(rownames(sce), e$var_names)
  x <- as.matrix(SummarizedExperiment::assay(sce, 1))
  expect_equal(unname(x), t(e$X))
  expect_equal(unname(SingleCellExperiment::reducedDim(sce, "X_pca")), e$obsm$X_pca,
               tolerance = 1e-6)
  expect_identical(SummarizedExperiment::rowData(sce)$gene_symbol, e$var$gene_symbol$values)
})

test_that("inspect reports format and shape of the 10x and MTX fixtures", {
  for (f in c("tiny_10x.h5", "tiny_mtx")) {
    i <- inspect(tiny(f))
    expect_identical(c(i$n_obs, i$n_vars), c(6L, 5L), info = f)
  }
})

test_that("convert to h5ad round-trips the matrix", {
  skip_if_not_installed("SingleCellExperiment")
  out <- tempfile(fileext = ".h5ad")
  on.exit(unlink(out))
  convert(tiny("tiny_10x.h5"), out)
  back <- read_h5ad(out)
  expect_equal(unname(as.matrix(SummarizedExperiment::assay(back, 1))), t(expected()$counts))
  expect_identical(colnames(back), expected()$obs_names)
})

test_that("X and a `counts` layer get distinct assay names", {
  skip("bug: X is always named 'counts', so a 'counts' layer duplicates the name")
  sce <- read_h5ad(tiny("tiny.h5ad"))
  expect_false(anyDuplicated(SummarizedExperiment::assayNames(sce)) > 0)
})

test_that("a categorical NA reads as NA", {
  skip("bug: NA code -1 becomes an out-of-range factor code ('malformed factor')")
  sce <- read_h5ad(tiny("tiny.h5ad"))
  expect_identical(as.character(sce$cell_type), expected()$obs$cell_type$values)
})

test_that("convert writes H5Seurat when the output ends in .h5seurat", {
  skip("bug: R convert() always writes H5AD")
  out <- tempfile(fileext = ".h5seurat")
  on.exit(unlink(out))
  convert(tiny("tiny.h5ad"), out)
  expect_match(inspect(out)$format, "H5Seurat")
})
