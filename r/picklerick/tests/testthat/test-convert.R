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

test_that("convert h5ad produces a valid output file", {
  input  <- golden("pbmc3k_reference.h5ad")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)
  expect_true(file.exists(output))
  expect_gt(file.size(output), 1000L)
})

test_that("convert h5ad output has correct shape via inspect", {
  input  <- golden("pbmc3k_reference.h5ad")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)
  info <- inspect(output)

  expect_equal(info$n_obs,  2700L)
  expect_equal(info$n_vars, 13714L)
  expect_true("nCount_RNA"   %in% info$obs_cols)
  expect_true("nFeature_RNA" %in% info$obs_cols)
})

test_that("convert h5ad preserves obsm embeddings", {
  input  <- golden("pbmc3k_reference.h5ad")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)
  info <- inspect(output)

  expect_true("X_pca"  %in% info$obsm_keys)
  expect_true("X_umap" %in% info$obsm_keys)
})

test_that("convert h5ad dtype f64 produces wider output file", {
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

test_that("convert h5seurat produces correct shape via inspect", {
  input  <- golden("pbmc3k.h5seurat")
  output <- tempfile(fileext = ".h5ad")
  on.exit(unlink(output))

  convert(input, output)
  info <- inspect(output)

  expect_equal(info$n_obs,  2700L)
  expect_equal(info$n_vars, 13714L)
})

# ---------------------------------------------------------------------------
# chunk_size does not affect output shape
# ---------------------------------------------------------------------------

test_that("different chunk_sizes produce identical shapes", {
  input   <- golden("pbmc3k_reference.h5ad")
  out_big <- tempfile(fileext = ".h5ad")
  out_sml <- tempfile(fileext = ".h5ad")
  on.exit({ unlink(out_big); unlink(out_sml) })

  convert(input, out_big, chunk_size = 5000L)
  convert(input, out_sml, chunk_size = 100L)

  info_big <- inspect(out_big)
  info_sml <- inspect(out_sml)

  expect_equal(info_big$n_obs,     info_sml$n_obs)
  expect_equal(info_big$n_vars,    info_sml$n_vars)
  expect_equal(info_big$obs_cols,  info_sml$obs_cols)
  expect_equal(info_big$obsm_keys, info_sml$obsm_keys)
})

# ---------------------------------------------------------------------------
# rhdf5 coexistence
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
  info <- inspect(output)
  expect_equal(info$n_obs, 2700L)
})
