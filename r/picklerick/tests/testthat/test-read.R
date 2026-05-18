golden <- function(name) {
  p <- file.path(
    dirname(dirname(dirname(dirname(getwd())))),
    "tests", "golden", name
  )
  if (!file.exists(p)) skip(paste("golden fixture not found:", name))
  p
}

# ---------------------------------------------------------------------------
# as = "list" — raw FFI payload
# ---------------------------------------------------------------------------

test_that("read_h5ad(as = 'list') exposes the expected fields and shapes", {
  input <- golden("pbmc3k_reference.h5ad")
  raw <- read_h5ad(input, as = "list")

  expect_named(raw,
    c("n_obs", "n_vars", "obs_index", "var_index",
      "obs_cols", "var_cols",
      "x_indptr", "x_indices", "x_data",
      "obsm", "varm", "layers", "obsp", "uns_json"),
    ignore.order = TRUE)

  expect_equal(raw$n_obs,  2700L)
  expect_equal(raw$n_vars, 13714L)
  expect_length(raw$obs_index, 2700L)
  expect_length(raw$var_index, 13714L)

  # CSR over cells: indptr length n_obs+1, last entry == nnz.
  expect_length(raw$x_indptr, raw$n_obs + 1L)
  expect_equal(tail(raw$x_indptr, 1L), length(raw$x_indices))
  expect_equal(length(raw$x_data),     length(raw$x_indices))

  # obsm: dense matrices keyed by name.
  expect_true("X_pca"  %in% names(raw$obsm))
  expect_true("X_umap" %in% names(raw$obsm))
  expect_equal(nrow(raw$obsm[["X_pca"]]), raw$n_obs)
})

# ---------------------------------------------------------------------------
# as = "SingleCellExperiment"
# ---------------------------------------------------------------------------

test_that("read_h5ad(as = 'SingleCellExperiment') builds a valid SCE", {
  skip_if_not_installed("SingleCellExperiment")
  input <- golden("pbmc3k_reference.h5ad")
  sce <- read_h5ad(input, as = "SingleCellExperiment")

  expect_s4_class(sce, "SingleCellExperiment")
  expect_equal(ncol(sce), 2700L)
  expect_equal(nrow(sce), 13714L)

  # counts assay present, dgCMatrix, dimnames wired through.
  m <- SummarizedExperiment::assay(sce, "counts")
  expect_s4_class(m, "dgCMatrix")
  expect_equal(dim(m), c(13714L, 2700L))
  expect_false(is.null(rownames(sce)))
  expect_false(is.null(colnames(sce)))

  # Integer X → double on the R side.
  expect_type(m@x, "double")

  # Reductions surfaced via reducedDims.
  rd <- SingleCellExperiment::reducedDimNames(sce)
  expect_true("X_pca"  %in% rd)
  expect_true("X_umap" %in% rd)
  expect_equal(nrow(SingleCellExperiment::reducedDim(sce, "X_pca")), 2700L)
})

# ---------------------------------------------------------------------------
# as = "Seurat" — only if Seurat is installed
# ---------------------------------------------------------------------------

test_that("read_h5ad(as = 'Seurat') builds a valid Seurat object", {
  skip_if_not_installed("Seurat")
  input <- golden("pbmc3k_reference.h5ad")
  obj <- read_h5ad(input, as = "Seurat")

  expect_s4_class(obj, "Seurat")
  expect_equal(ncol(obj), 2700L)
  expect_equal(nrow(obj), 13714L)

  # AnnData X_pca / X_umap → Seurat pca / umap.
  reds <- Seurat::Reductions(obj)
  expect_true("pca"  %in% reds)
  expect_true("umap" %in% reds)

  emb <- Seurat::Embeddings(obj, "pca")
  expect_equal(nrow(emb), 2700L)
  expect_equal(rownames(emb)[1L], colnames(obj)[1L])
})

# ---------------------------------------------------------------------------
# chunk_size invariance — same SCE shape and counts regardless of chunking.
# ---------------------------------------------------------------------------

test_that("chunk_size does not affect the assembled SCE", {
  skip_if_not_installed("SingleCellExperiment")
  input <- golden("pbmc3k_reference.h5ad")
  big <- read_h5ad(input, as = "SingleCellExperiment", chunk_size = 5000L)
  sml <- read_h5ad(input, as = "SingleCellExperiment", chunk_size = 100L)

  expect_equal(dim(big), dim(sml))
  expect_equal(sum(SummarizedExperiment::assay(big, "counts")),
               sum(SummarizedExperiment::assay(sml, "counts")))
})
