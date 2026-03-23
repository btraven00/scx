test_that("all reference obsm keys present", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  missing <- setdiff(names(ref_ad()$obsm), names(scx_ad()$obsm))
  expect_length(missing, 0)
})

test_that("X_pca shape correct", {
  skip_if_no_golden(SCX_PATH)
  pca <- scx_ad()$obsm[["X_pca"]]
  expect_equal(nrow(pca), EXPECTED_N_OBS)
  expect_equal(ncol(pca), 30L)
})

test_that("X_umap shape correct", {
  skip_if_no_golden(SCX_PATH)
  umap <- scx_ad()$obsm[["X_umap"]]
  expect_equal(nrow(umap), EXPECTED_N_OBS)
  expect_equal(ncol(umap), 2L)
})

test_that("X_pca shapes match reference", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  expect_equal(dim(scx_ad()$obsm[["X_pca"]]), dim(ref_ad()$obsm[["X_pca"]]))
})

test_that("X_pca values agree per-axis (sign-flip tolerant)", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  s <- scx_ad()$obsm[["X_pca"]]
  r <- ref_ad()$obsm[["X_pca"]]
  n_axes <- min(ncol(s), ncol(r), 10L)
  for (i in seq_len(n_axes)) {
    aligned <- isTRUE(all.equal(s[, i],  r[, i], tolerance = 1e-3)) ||
               isTRUE(all.equal(s[, i], -r[, i], tolerance = 1e-3))
    expect_true(aligned, info = paste("X_pca axis", i, "differs beyond tolerance"))
  }
})

test_that("X_umap values agree with reference (sign-flip tolerant)", {
  skip_if_no_golden(SCX_PATH, REF_PATH)
  s <- scx_ad()$obsm[["X_umap"]]
  r <- ref_ad()$obsm[["X_umap"]]
  for (i in seq_len(ncol(r))) {
    aligned <- isTRUE(all.equal(s[, i],  r[, i], tolerance = 1e-3)) ||
               isTRUE(all.equal(s[, i], -r[, i], tolerance = 1e-3))
    expect_true(aligned, info = paste("X_umap axis", i, "differs beyond tolerance"))
  }
})
