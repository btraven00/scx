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

test_that("gene_blocks covers the genes in order, block by block", {
  path <- spe_fixture("visiumhd_spe.rds")
  full <- read_genes(path)
  it <- gene_blocks(path, block_size = 7L)
  expect_identical(it$n, 5L)
  expect_identical(lengths(it$genes), c(7L, 7L, 7L, 7L, 2L))
  got <- list()
  while (!is.null(m <- it$next_block())) got[[length(got) + 1L]] <- m
  expect_length(got, 5L)
  expect_null(it$next_block())
  expect_identical(do.call(cbind, got), full)
  # Random access, as forked workers would use it.
  expect_identical(it$read(3L), full[, 15:21])
  expect_error(it$read(6L), "no block 6")

  sub <- gene_blocks(path, block_size = 2L, genes = colnames(full)[c(9L, 1L, 4L)])
  expect_identical(do.call(cbind, lapply(seq_len(sub$n), sub$read)), full[, c(9L, 1L, 4L)])
  expect_output(print(sub), "3 genes in 2 blocks")
})

test_that("forked workers read blocks of an iterator opened in the parent", {
  skip_on_os("windows") # no fork
  path <- spe_fixture("visiumhd_spe.rds")
  it <- gene_blocks(path, block_size = 10L)
  first <- it$read(1L) # the parent has used its reader threads before forking
  # A child inheriting rayon's global pool has no threads and hangs; the
  # reader rebuilds its pool per process.
  got <- parallel::mclapply(seq_len(it$n), it$read, mc.cores = 2L)
  expect_identical(got[[1L]], first)
  expect_identical(do.call(cbind, got), read_genes(path))
})
