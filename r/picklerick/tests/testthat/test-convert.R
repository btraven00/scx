test_that("convert() produces a valid H5AD file", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()

  out <- tempfile(fileext = ".h5ad")
  on.exit(unlink(out), add = TRUE)

  result <- convert(H5SEURAT_PATH, out)

  expect_identical(result, out)
  expect_true(file.exists(out))
  expect_gt(file.size(out), 0)
})

test_that("convert() output is readable by anndataR", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()

  out <- tempfile(fileext = ".h5ad")
  on.exit(unlink(out), add = TRUE)

  convert(H5SEURAT_PATH, out)
  adata <- anndataR::read_h5ad(out)

  expect_equal(nrow(adata$obs), EXPECTED_N_OBS)
  expect_equal(nrow(adata$var), EXPECTED_N_VARS)
})

test_that("convert() errors on a non-existent input", {
  skip_if_no_scx()
  expect_error(convert("/nonexistent/file.h5seurat", tempfile()))
})

test_that("convert() errors on unknown dtype", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()
  expect_error(convert(H5SEURAT_PATH, tempfile(), dtype = "float128"))
})
