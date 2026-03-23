test_that("read_h5seurat() returns InMemoryAnnData with correct shape", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()

  adata <- read_h5seurat(H5SEURAT_PATH)

  expect_s3_class(adata, "AbstractAnnData")
  expect_equal(nrow(adata$obs), EXPECTED_N_OBS)
  expect_equal(nrow(adata$var), EXPECTED_N_VARS)
})

test_that("read_h5seurat() X matrix has correct nnz", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()

  adata <- read_h5seurat(H5SEURAT_PATH)
  expect_equal(Matrix::nnzero(adata$X), EXPECTED_NNZ)
})

test_that("read_h5seurat() preserves cell metadata columns", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()

  adata <- read_h5seurat(H5SEURAT_PATH)
  expect_true(ncol(adata$obs) > 0)
})

test_that("read_h5seurat() preserves obsm keys", {
  skip_if_no_fixture(H5SEURAT_PATH)
  skip_if_no_scx()

  adata <- read_h5seurat(H5SEURAT_PATH)
  expect_true("X_pca"  %in% names(adata$obsm))
  expect_true("X_umap" %in% names(adata$obsm))
})

test_that("read_h5ad() returns InMemoryAnnData", {
  skip_if_no_fixture(H5AD_REF_PATH)

  adata <- read_h5ad(H5AD_REF_PATH)

  expect_s3_class(adata, "AbstractAnnData")
  expect_equal(nrow(adata$obs), EXPECTED_N_OBS)
  expect_equal(nrow(adata$var), EXPECTED_N_VARS)
})

test_that("read_dataset() dispatches on extension", {
  skip_if_no_fixture(H5SEURAT_PATH, H5AD_REF_PATH)
  skip_if_no_scx()

  adata_s <- read_dataset(H5SEURAT_PATH)
  adata_a <- read_dataset(H5AD_REF_PATH)

  expect_equal(nrow(adata_s$obs), EXPECTED_N_OBS)
  expect_equal(nrow(adata_a$obs), EXPECTED_N_OBS)
})

test_that("read_h5seurat() output matches reference H5AD", {
  skip_if_no_fixture(H5SEURAT_PATH, H5AD_REF_PATH)
  skip_if_no_scx()

  scx <- read_h5seurat(H5SEURAT_PATH)
  ref <- read_h5ad(H5AD_REF_PATH)

  expect_equal(nrow(scx$obs),  nrow(ref$obs))
  expect_equal(nrow(scx$var),  nrow(ref$var))
  expect_equal(Matrix::nnzero(scx$X), Matrix::nnzero(ref$X))
  expect_identical(scx$obs_names, ref$obs_names)
  expect_identical(scx$var_names, ref$var_names)
})
