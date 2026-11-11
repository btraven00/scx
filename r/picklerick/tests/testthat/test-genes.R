# read_genes over a SpatialExperiment .rds: 5000 Visium HD bins x 30 genes
# (scripts/make_spe_fixtures.R), against what R read back from it.

spe_fixture <- function(name) {
  p <- file.path(SCX_ROOT, "tests", "fixtures", "tiny", name)
  skip_if_no_fixture(p)
  p
}

test_that("read_genes reads every gene of an .rds, cells x genes", {
  path <- spe_fixture("visiumhd_spe.rds")
  e <- jsonlite::fromJSON(spe_fixture("visiumhd_spe.json"))
  m <- read_genes(path)
  expect_s4_class(m, "dgCMatrix")
  expect_identical(dim(m), c(5000L, 30L))
  expect_identical(colnames(m), e$var_names)
  expect_identical(rownames(m), e$obs_names)
  expect_equal(unname(Matrix::colSums(m)), e$gene_sums)
  expect_equal(unname(Matrix::rowSums(m)), e$cell_sums)
  expect_identical(length(m@x), as.integer(e$nnz))
})

test_that("read_genes keeps the requested order, by position or name", {
  path <- spe_fixture("visiumhd_spe.rds")
  full <- read_genes(path)
  for (g in list(c(4L, 5L, 6L, 20L), c(20L, 4L, 4L, 9L), 30L)) {
    expect_identical(read_genes(path, g, chunk_size = 2L), full[, g, drop = FALSE])
  }
  names <- colnames(full)[c(7L, 2L)]
  expect_identical(read_genes(path, names), full[, names])
  expect_error(read_genes(path, "NOT_A_GENE"), "NOT_A_GENE")
  expect_error(read_genes(path, 31L), "beyond")
})
