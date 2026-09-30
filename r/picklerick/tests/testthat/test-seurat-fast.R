# constructor = "fast" is an unofficial Seurat constructor (R/seurat_fast.R).
# Its whole contract is "identical to CreateSeuratObject()", so every test
# here compares against the standard constructor rather than fixed values.

expect_same_seurat <- function(fast, std) {
  expect_true(methods::validObject(fast))
  expect_identical(SeuratObject::Cells(fast), SeuratObject::Cells(std))
  expect_identical(SeuratObject::Features(fast), SeuratObject::Features(std))
  expect_identical(SeuratObject::Layers(fast), SeuratObject::Layers(std))
  expect_identical(SeuratObject::LayerData(fast, "counts"), SeuratObject::LayerData(std, "counts"))
  expect_identical(colnames(fast[[]]), colnames(std[[]]))
  expect_equal(fast[[]], std[[]], ignore_attr = TRUE)
  expect_identical(vapply(fast[[]], function(x) class(x)[1], ""),
                   vapply(std[[]], function(x) class(x)[1], ""))
  expect_identical(SeuratObject::Idents(fast), SeuratObject::Idents(std))
  expect_identical(SeuratObject::Key(fast[["RNA"]]), SeuratObject::Key(std[["RNA"]]))
  expect_identical(SeuratObject::DefaultAssay(fast), SeuratObject::DefaultAssay(std))
}

toy <- function(rn = paste0("g", 1:4), cn = paste0("c", 1:3), x = c(1, 2, 3, 4, 5)) {
  methods::new("dgCMatrix", i = c(0L, 2L, 1L, 3L, 0L), p = c(0L, 2L, 4L, 5L),
               x = x, Dim = c(4L, 3L), Dimnames = list(rn, cn))
}
std_of <- function(m, meta = data.frame(row.names = colnames(m)))
  SeuratObject::CreateSeuratObject(counts = m, meta.data = meta, assay = "RNA")

test_that("fast constructor matches CreateSeuratObject on the pbmc3k golden file", {
  skip_if_not_installed("Seurat")
  input <- H5AD_REF_PATH
  skip_if_no_fixture(input)
  std  <- read_h5ad(input, as = "Seurat")
  fast <- read_h5ad(input, as = "Seurat", constructor = "fast")
  expect_same_seurat(fast, std)
  expect_identical(SeuratObject::Reductions(fast), SeuratObject::Reductions(std))
  expect_identical(fast@misc, std@misc)

  # The object must behave the same downstream, not just look the same.
  run <- function(o) {
    o <- Seurat::NormalizeData(o, verbose = FALSE)
    Seurat::FindVariableFeatures(o, nfeatures = 500, verbose = FALSE)
  }
  expect_identical(SeuratObject::VariableFeatures(run(fast)),
                   SeuratObject::VariableFeatures(run(std)))
})

test_that("fast constructor keeps obs metadata columns in CreateSeuratObject's order", {
  skip_if_not_installed("SeuratObject")
  m <- toy()
  meta <- data.frame(batch = c("a", "b", "a"), score = c(0.1, 0.2, 0.3),
                     row.names = colnames(m))
  expect_same_seurat(picklerick:::.seurat_fast(m, meta), std_of(m, meta))
})

test_that("stored zeros are not counted as features", {
  skip_if_not_installed("SeuratObject")
  m <- toy(x = c(1, 0, 3, 4, 0))   # two explicit zeros
  fast <- picklerick:::.seurat_fast(m, data.frame(row.names = colnames(m)))
  expect_same_seurat(fast, std_of(m))
  expect_identical(unname(fast$nFeature_RNA), c(1L, 2L, 0L))
})

test_that("input needing CreateSeuratObject's sanitising falls back, with a message", {
  skip_if_not_installed("SeuratObject")
  no_meta <- function(m) data.frame(row.names = colnames(m))
  underscore <- toy(rn = c("g_1", "g2", "g3", "g4"))
  expect_message(r <- picklerick:::.seurat_fast(underscore, no_meta(underscore)), "contain '_'")
  expect_null(r)
  dup <- toy(rn = c("g1", "g1", "g3", "g4"))
  expect_message(r <- picklerick:::.seurat_fast(dup, no_meta(dup)), "duplicate")
  expect_null(r)
  m <- toy()
  expect_message(r <- picklerick:::.seurat_fast(m, data.frame(nCount_RNA = 1:3, row.names = colnames(m))),
                 "nCount_RNA")
  expect_null(r)
})

test_that("read_h5ad falls back to CreateSeuratObject when the fast path does not apply", {
  skip_if_not_installed("Seurat")
  input <- H5AD_REF_PATH
  skip_if_no_fixture(input)
  local_mocked_bindings(.seurat_fast_blocker = function(...) "forced for test",
                        .package = "picklerick")
  expect_message(fast <- read_h5ad(input, as = "Seurat", constructor = "fast"), "forced for test")
  expect_same_seurat(fast, read_h5ad(input, as = "Seurat"))
})
