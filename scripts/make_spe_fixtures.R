# SingleCellExperiment / SpatialExperiment .rds fixtures (tests/fixtures/tiny):
#   tiny_sce.rds  the tiny dataset (expected.json) as a SingleCellExperiment:
#                 counts + logcounts (expected.json X), colData = obs,
#                 rowData gene_symbol, reducedDim PCA = obsm X_pca
#   tiny_spe.rds  the same as a SpatialExperiment, plus spatialCoords
#                 (cell k: x = 10k, y = 100 - k) and a stored image
#   visiumhd_spe.rds + visiumhd_spe.json
#                 5000 bins x 30 genes of the public 10x Visium HD lung cancer
#                 8 um SpatialExperiment (pass its .rds as the first argument),
#                 past the reader's lazy-read threshold; the .json holds what R
#                 reads back (names, sums, coordinates) for the Rust test.
# The pixi test env has no Bioconductor; run with
#   pixi exec -c conda-forge -c bioconda --spec r-base \
#     --spec bioconductor-spatialexperiment --spec r-jsonlite \
#     Rscript scripts/make_spe_fixtures.R path/to/visiumhd_008um_spe.rds
suppressMessages({
  library(SpatialExperiment)
  library(jsonlite)
})

dir <- "tests/fixtures/tiny"
e <- fromJSON(file.path(dir, "expected.json"), simplifyVector = TRUE)

as_dgc <- function(m) as(as(t(m) * 1.0, "CsparseMatrix"), "dgCMatrix") # genes x cells
counts <- as_dgc(e$counts)
logcounts <- as_dgc(e$X)
dimnames(counts) <- dimnames(logcounts) <- list(e$var_names, e$obs_names)

obs <- DataFrame(row.names = e$obs_names)
for (name in names(e$obs)) {
  col <- e$obs[[name]]
  obs[[name]] <- switch(col$type,
    categorical = factor(col$values, levels = col$levels),
    int = as.integer(col$values),
    bool = as.logical(col$values),
    float = as.double(col$values),
    string = as.character(col$values),
    stop("unknown obs type ", col$type)
  )
}
pca <- e$obsm$X_pca
rownames(pca) <- e$obs_names

sce <- SingleCellExperiment(
  assays = list(counts = counts, logcounts = logcounts),
  colData = obs,
  rowData = DataFrame(gene_symbol = e$var$gene_symbol$values),
  reducedDims = list(PCA = pca)
)
saveRDS(sce, file.path(dir, "tiny_sce.rds"))

k <- seq_len(e$n_obs) - 1
coords <- cbind(x = 10 * k, y = 100 - k)
spe <- SpatialExperiment(
  assays = list(counts = counts, logcounts = logcounts),
  colData = obs,
  rowData = DataFrame(gene_symbol = e$var$gene_symbol$values),
  reducedDims = list(PCA = pca),
  spatialCoords = coords
)
# A stored image, as Visium objects carry: only a path, nothing to read.
imgData(spe) <- DataFrame(
  sample_id = "sample01", image_id = "lowres",
  data = I(list(SpatialImage("/nonexistent/tissue_lowres_image.png"))),
  scaleFactor = 0.5
)
stopifnot(is(spe@assays, "SimpleAssays"))
saveRDS(spe, file.path(dir, "tiny_spe.rds"))

src <- commandArgs(TRUE)[1]
if (!is.na(src)) {
  x <- readRDS(src)
  set.seed(1)
  keep_genes <- order(-rowSums(counts(x)))[1:30]
  keep_genes <- sort(c(keep_genes[1:28], which(rowSums(counts(x)) == 0)[1:2]))
  keep_bins <- sort(sample(ncol(x), 5000))
  v <- x[keep_genes, keep_bins]
  v@metadata <- list() # local paths of the source 10x files
  saveRDS(v, file.path(dir, "visiumhd_spe.rds"))
  m <- counts(v)
  write_json(list(
    n_obs = ncol(v), n_vars = nrow(v),
    obs_names = colnames(v), var_names = rownames(v),
    var_id = rowData(v)$ID, var_type = as.character(rowData(v)$Type),
    gene_sums = unname(rowSums(m)), cell_sums = unname(colSums(m)),
    nnz = length(m@x),
    in_tissue = v$in_tissue, array_row = v$array_row,
    spatial = unname(spatialCoords(v))
  ), file.path(dir, "visiumhd_spe.json"), digits = NA)
}
