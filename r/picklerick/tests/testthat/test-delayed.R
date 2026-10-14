# PicklerickMatrix: a DelayedArray seed over anything scx reads. The oracle is
# the eager reader: every block, however the index is written, must equal the
# same slice of the fully read matrix.

delayed_fixture <- function(name) {
  p <- file.path(SCX_ROOT, "tests", "fixtures", "zarr", name)
  skip_if_no_fixture(p)
  p
}

for (name in c("small.h5ad", "small_v3.zarr", "small_v2.zarr")) {
  test_that(sprintf("PicklerickMatrix over %s matches the eager read", name), {
    path <- delayed_fixture(name)
    raw <- read_h5ad(path, as = "list")
    full <- picklerick:::.build_dgc(raw)   # X of the eager read, features x cells

    m <- PicklerickMatrix(path, chunk_size = 7L)
    expect_s4_class(m, "DelayedMatrix")
    expect_identical(dim(m), dim(full))
    expect_identical(dimnames(m), dimnames(full))
    expect_true(DelayedArray::is_sparse(m))
    expect_identical(DelayedArray::chunkdim(DelayedArray::seed(m)), c(nrow(full), 7L))

    expect_equal(as.matrix(m), as.matrix(full))
    # Sorted, unsorted, duplicated and single-cell column selections, with
    # and without a feature selection; one cell crosses two read chunks.
    for (j in list(1:5, c(60L, 3L, 3L, 7L), 20L, c(6L, 8L, 9L, 14L, 15L), seq(2, 60, by = 2))) {
      expect_equal(as.matrix(m[, j]), as.matrix(full[, j]))
      expect_equal(as.matrix(m[c(40L, 1L, 1L), j]), as.matrix(full[c(40L, 1L, 1L), j]))
    }
    expect_equal(as.matrix(m[5:9, ]), as.matrix(full[5:9, ]))
    # Block-wise reductions walk chunkdim-sized column blocks.
    expect_equal(DelayedArray::colSums(m), Matrix::colSums(full))
    expect_equal(DelayedArray::rowSums(m), Matrix::rowSums(full))
    # Sparse extraction keeps the values sparse.
    blk <- SparseArray::extract_sparse_array(DelayedArray::seed(m), list(NULL, 1:3))
    expect_s4_class(blk, "SVT_SparseMatrix")
    expect_equal(as.matrix(blk), unname(as.matrix(full[, 1:3])))
  })
}

test_that("PicklerickMatrix exposes a layer and rejects a missing one", {
  path <- delayed_fixture("small_v3.zarr")
  raw <- read_h5ad(path, as = "list")
  layer <- picklerick:::.csr_to_dgc_T(raw$layers$counts, raw$obs_index, raw$var_index)
  x <- picklerick:::.build_dgc(raw)
  expect_false(identical(as.matrix(layer), as.matrix(x)))  # the layer is not X
  m <- PicklerickMatrix(path, layer = "counts")
  expect_equal(as.matrix(m[, 1:4]), as.matrix(layer[, 1:4]))
  expect_equal(as.matrix(m[c(5L, 2L), c(9L, 1L, 60L)]), as.matrix(layer[c(5L, 2L), c(9L, 1L, 60L)]))
  expect_error(PicklerickMatrix(path, layer = "nope"), "nope")
})

test_that("PicklerickMatrix drops into a SingleCellExperiment and survives serialisation", {
  path <- delayed_fixture("small_v3.zarr")
  m <- PicklerickMatrix(path)
  sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = m))
  expect_identical(dim(sce), dim(m))
  m2 <- unserialize(serialize(m, NULL))
  expect_equal(as.matrix(m2[, 1:3]), as.matrix(m[, 1:3]))
})

test_that("read_zarr(lazy = TRUE) wraps X in a PicklerickMatrix", {
  path <- delayed_fixture("small_v3.zarr")
  sce <- read_zarr(path, lazy = TRUE)
  # The fixture has a `counts` layer, so X is the `X` assay, as when eager.
  expect_identical(SummarizedExperiment::assayNames(sce), "X")
  x <- SummarizedExperiment::assay(sce, "X")
  expect_s4_class(x, "DelayedMatrix")
  expect_s4_class(DelayedArray::seed(x), "PicklerickMatrixSeed")
  eager <- SummarizedExperiment::assay(read_zarr(path), "X")
  expect_equal(as.matrix(x[, 1:5]), as.matrix(eager[, 1:5]))
  expect_identical(SummarizedExperiment::colData(sce)$cluster,
                   SummarizedExperiment::colData(read_zarr(path))$cluster)
})
