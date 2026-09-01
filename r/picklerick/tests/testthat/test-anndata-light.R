# The light AnnData handle: read_h5ad(as = "AnnData").
#
# Why it exists: building a SingleCellExperiment costs a constant ~3.3 s of S4
# and Bioconductor namespace warm-up on this machine, whatever the dataset size
# (0.05 s vs 3.19 s at 1k cells; 3.72 s vs 7.09 s at 100k). That is not work
# proportional to the data and cannot be optimised inside the conversion, so it
# is deferred until someone asks for an SCE.

test_that("read_h5ad(as = 'AnnData') builds nothing until a field is touched", {
  input <- golden("pbmc3k_reference.h5ad")
  a <- read_h5ad(input, as = "AnnData")

  expect_s3_class(a, "picklerick_anndata")
  # Shape comes from the reader, so it must not trigger any construction.
  expect_equal(a$n_obs, 2700L)
  expect_equal(a$n_vars, 13714L)
  expect_equal(dim(a), c(13714L, 2700L))          # genes x cells, like an SCE
  expect_length(ls(get(".cache", envir = a)), 0L)

  expect_equal(dimnames(a)[[2]][1], a$obs_names[1])
  expect_length(ls(get(".cache", envir = a)), 0L)
})

test_that("fields are built on first access and cached afterwards", {
  input <- golden("pbmc3k_reference.h5ad")
  a <- read_h5ad(input, as = "AnnData")

  obs <- a$obs
  expect_s3_class(obs, "data.frame")
  expect_equal(nrow(obs), 2700L)
  expect_true("obs" %in% ls(get(".cache", envir = a)))

  # Same object back, not a rebuild.
  expect_identical(a$obs, obs)

  # $X must work without anything else having loaded Matrix: resolving the
  # dgCMatrix class by bare name failed here, because new() looks it up in the
  # caller's topenv and only the SCE path had pulled Matrix in.
  m <- a$X
  expect_s4_class(m, "dgCMatrix")
  expect_equal(dim(m), c(13714L, 2700L))
  expect_equal(length(m@x), 2282976L)
})

test_that("as_sce() equals the direct SingleCellExperiment path", {
  skip_if_not_installed("SingleCellExperiment")
  input <- golden("pbmc3k_reference.h5ad")

  converted <- as_sce(read_h5ad(input, as = "AnnData"))
  direct    <- read_h5ad(input, as = "SingleCellExperiment")

  expect_equal(dim(converted), dim(direct))
  expect_equal(dimnames(converted), dimnames(direct))
  expect_equal(
    sum(SummarizedExperiment::assay(converted, 1)@x),
    sum(SummarizedExperiment::assay(direct, 1)@x)
  )
  expect_equal(
    colnames(SummarizedExperiment::colData(converted)),
    colnames(SummarizedExperiment::colData(direct))
  )
  expect_equal(
    SingleCellExperiment::reducedDimNames(converted),
    SingleCellExperiment::reducedDimNames(direct)
  )
})

test_that("the light handle is cheaper than the SingleCellExperiment path", {
  skip_if_not_installed("SingleCellExperiment")
  input <- golden("pbmc3k_reference.h5ad")

  # Warm the S4 machinery first, so this measures the conversion and not
  # one-off namespace loading — the point is that a gap remains even then.
  invisible(read_h5ad(input, as = "SingleCellExperiment"))

  light <- system.time(read_h5ad(input, as = "AnnData"))[["elapsed"]]
  heavy <- system.time(read_h5ad(input, as = "SingleCellExperiment"))[["elapsed"]]
  expect_lt(light, heavy)
})
