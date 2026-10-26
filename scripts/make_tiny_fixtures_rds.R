# Seurat .rds versions of the tiny dataset (tests/fixtures/tiny/expected.json):
# the counts plus every obs column, as a v3/v4 `Assay`, a v5 `Assay5`, and an
# `Assay5` split into one layer per donor. Run in the pixi `test` env:
#   pixi run -e test make-tiny-fixtures-rds
suppressMessages({
  library(SeuratObject)
  library(Matrix)
  library(jsonlite)
})

dir <- "tests/fixtures/tiny"
e <- fromJSON(file.path(dir, "expected.json"), simplifyVector = TRUE)

counts <- as(t(e$counts), "CsparseMatrix") # genes x cells
counts <- as(counts * 1.0, "dgCMatrix")
dimnames(counts) <- list(e$var_names, e$obs_names)

meta <- data.frame(row.names = e$obs_names)
for (name in names(e$obs)) {
  col <- e$obs[[name]]
  meta[[name]] <- switch(col$type,
    categorical = factor(col$values, levels = col$levels),
    int = as.integer(col$values),
    bool = as.logical(col$values),
    float = as.double(col$values),
    string = as.character(col$values),
    stop("unknown obs type ", col$type)
  )
}

make <- function(assay_version) {
  options(Seurat.object.assay.version = assay_version)
  CreateSeuratObject(counts = counts, meta.data = meta, min.cells = 0, min.features = 0)
}

v4 <- make("v3")
stopifnot(class(v4[["RNA"]]) == "Assay")
saveRDS(v4, file.path(dir, "tiny_v4.rds"))

v5 <- make("v5")
stopifnot(class(v5[["RNA"]]) == "Assay5")
saveRDS(v5, file.path(dir, "tiny_v5.rds"))

split5 <- split(v5, f = v5$donor)
stopifnot(length(Layers(split5, search = "counts")) == 2)
saveRDS(split5, file.path(dir, "tiny_v5_split.rds"))
