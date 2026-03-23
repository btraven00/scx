test_that("n_obs matches expected", {
  skip_if_no_golden(SCX_PATH)
  expect_equal(nrow(scx_ad()$obs), EXPECTED_N_OBS)
})

test_that("n_vars matches expected", {
  skip_if_no_golden(SCX_PATH)
  expect_equal(nrow(scx_ad()$var), EXPECTED_N_VARS)
})

test_that("shape matches reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(nrow(scx_ad()$obs), nrow(ref_ad()$obs))
  expect_equal(nrow(scx_ad()$var), nrow(ref_ad()$var))
})

test_that("obs names match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_identical(scx_ad()$obs_names, ref_ad()$obs_names)
})

test_that("var names match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_identical(scx_ad()$var_names, ref_ad()$var_names)
})
