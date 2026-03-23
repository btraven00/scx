test_that("X nnz matches expected", {
  skip_if_no_golden(SCX_PATH)
  expect_equal(Matrix::nnzero(scx_ad()$X), EXPECTED_NNZ)
})

test_that("X nnz matches reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(Matrix::nnzero(scx_ad()$X), Matrix::nnzero(ref_ad()$X))
})

test_that("X row sums match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  # X is obs x vars (cells x genes); rowSums = per-cell totals
  expect_equal(
    Matrix::rowSums(scx_ad()$X),
    Matrix::rowSums(ref_ad()$X),
    tolerance = 1e-4
  )
})

test_that("X col sums match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  # colSums = per-gene totals
  expect_equal(
    Matrix::colSums(scx_ad()$X),
    Matrix::colSums(ref_ad()$X),
    tolerance = 1e-4
  )
})

test_that("spot check: first 100 obs x 100 vars exact", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  s <- as.matrix(scx_ad()$X[1:100, 1:100])
  r <- as.matrix(ref_ad()$X[1:100, 1:100])
  expect_equal(s, r, tolerance = 1e-4)
})
