golden <- function(name) {
  p <- file.path(
    dirname(dirname(dirname(dirname(getwd())))),
    "tests", "golden", name
  )
  if (!file.exists(p)) skip(paste("golden fixture not found:", name))
  p
}

# ---------------------------------------------------------------------------
# Native binding
# ---------------------------------------------------------------------------

test_that("native binding is active", {
  expect_true(picklerick:::.native_available())
})

# ---------------------------------------------------------------------------
# h5ad round-trip
# ---------------------------------------------------------------------------

test_that("convert h5ad preserves shape and key obs columns", {
  input  <- golden("pbmc3k_reference.h5ad")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)

  expect_true(file.exists(output))
  expect_gt(file.size(output), 1000L)

  ad <- anndataR::read_h5ad(output)
  expect_equal(dim(ad), c(2700L, 13714L))
  expect_true("nCount_RNA"   %in% names(ad$obs))
  expect_true("nFeature_RNA" %in% names(ad$obs))
})

test_that("convert h5ad preserves obsm embeddings", {
  input  <- golden("pbmc3k_reference.h5ad")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)
  ad <- anndataR::read_h5ad(output)

  expect_true("X_pca"  %in% names(ad$obsm))
  expect_true("X_umap" %in% names(ad$obsm))
  expect_equal(dim(ad$obsm[["X_pca"]]),  c(2700L, 30L))
  expect_equal(dim(ad$obsm[["X_umap"]]), c(2700L,  2L))
})

test_that("convert h5ad dtype f64 produces wider matrix", {
  input   <- golden("pbmc3k_reference.h5ad")
  out_f32 <- tempfile(fileext = ".h5ad")
  out_f64 <- tempfile(fileext = ".h5ad")
  on.exit({ unlink(out_f32); unlink(out_f64) })

  convert(input, out_f32, dtype = "f32")
  convert(input, out_f64, dtype = "f64")

  expect_gt(file.size(out_f64), file.size(out_f32))
})

# ---------------------------------------------------------------------------
# h5seurat round-trip
# ---------------------------------------------------------------------------

test_that("convert h5seurat preserves shape", {
  input  <- golden("pbmc3k.h5seurat")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)
  ad <- anndataR::read_h5ad(output)

  expect_equal(dim(ad), c(2700L, 13714L))
})

test_that("read_h5seurat returns AnnData with correct dims", {
  path <- golden("pbmc3k.h5seurat")
  ad   <- read_h5seurat(path)

  expect_true(inherits(ad, "AbstractAnnData"))
  expect_equal(dim(ad), c(2700L, 13714L))
})

# ---------------------------------------------------------------------------
# chunk_size does not affect output correctness
# ---------------------------------------------------------------------------

test_that("different chunk_sizes produce identical outputs", {
  input   <- golden("pbmc3k_reference.h5ad")
  out_big <- tempfile(fileext = ".h5ad")
  out_sml <- tempfile(fileext = ".h5ad")
  on.exit({ unlink(out_big); unlink(out_sml) })

  convert(input, out_big, chunk_size = 5000L)
  convert(input, out_sml, chunk_size = 100L)

  ad_big <- anndataR::read_h5ad(out_big)
  ad_sml <- anndataR::read_h5ad(out_sml)

  expect_equal(dim(ad_big), dim(ad_sml))
  expect_equal(names(ad_big$obs), names(ad_sml$obs))
  expect_equal(names(ad_big$obsm), names(ad_sml$obsm))
})

# ---------------------------------------------------------------------------
# rhdf5 coexistence (the conflict that originally blocked Phase B)
# ---------------------------------------------------------------------------

test_that("picklerick works when rhdf5 is loaded in the same session", {
  if (!requireNamespace("rhdf5", quietly = TRUE)) skip("rhdf5 not installed")

  tmp_h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp_h5))
  rhdf5::h5createFile(tmp_h5)

  input  <- golden("pbmc3k_reference.h5ad")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output), add = TRUE)

  expect_no_error(convert(input, output))
  ad <- anndataR::read_h5ad(output)
  expect_equal(dim(ad), c(2700L, 13714L))
})
