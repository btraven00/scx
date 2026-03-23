test_that("all reference obs columns present in SCX", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  missing <- setdiff(colnames(ref_ad()$obs), colnames(scx_ad()$obs))
  expect_length(missing, 0)
})

test_that("factor columns are factors", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  ref_factors <- names(which(vapply(ref_ad()$obs, is.factor, logical(1))))
  for (col in ref_factors) {
    expect_true(
      is.factor(scx_ad()$obs[[col]]),
      info = paste("expected factor:", col)
    )
  }
})

test_that("numeric obs columns agree within tolerance", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  ref_nums <- names(which(vapply(ref_ad()$obs, is.numeric, logical(1))))
  for (col in ref_nums) {
    expect_equal(
      scx_ad()$obs[[col]],
      ref_ad()$obs[[col]],
      tolerance = 1e-4,
      info = col
    )
  }
})
