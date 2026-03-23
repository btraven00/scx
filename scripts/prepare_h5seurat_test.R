#!/usr/bin/env Rscript
# prepare_h5seurat_test.R
#
# Generates tests/golden/pbmc3k.h5seurat — a valid SeuratDisk H5Seurat file
# built from the same PBMC 3k data as pbmc3k.h5, using hdf5r directly.
#
# We write the schema manually so SeuratDisk is NOT required in the pixi env.
# The schema mirrors exactly what SaveH5Seurat() produces for a Seurat v3/v4
# object, so the scx H5SeuratReader is testable against real-world format.
#
# H5Seurat schema written:
#   /cell.names               string (ncells,)
#   /assays/RNA/features      string (ngenes,)
#   /assays/RNA/counts/       CSC sparse group
#     data                    float64 (nnz,)
#     indices                 int32   (nnz,)
#     indptr                  int32   (ncells+1,)
#     attr: dims              int32   [ngenes, ncells]
#   /meta.data/               cell metadata
#     <numeric_col>           float64 (ncells,)
#     <factor_col>/           group
#       values                int32 1-indexed codes (ncells,)
#       levels                string (nlevels,)
#   /reductions/pca/
#     cell.embeddings         float64 (ncells, 30) — stored C-order
#   /reductions/umap/
#     cell.embeddings         float64 (ncells, 2)
#   top-level attrs: active.assay, version
#
# Usage:
#   pixi run prepare-h5seurat-test

suppressPackageStartupMessages({
  library(Seurat)
  library(hdf5r)
})

GOLDEN_H5    <- "tests/golden/pbmc3k.h5"
OUT_H5SEURAT <- "tests/golden/pbmc3k.h5seurat"

if (!file.exists(GOLDEN_H5)) {
  stop("Run 'pixi run prepare-test-data' first to generate ", GOLDEN_H5)
}
if (file.exists(OUT_H5SEURAT)) {
  message("H5Seurat fixture already exists — skipping. Delete to regenerate:")
  message("  ", OUT_H5SEURAT)
  quit(save = "no", status = 0)
}

message("Reading data from ", GOLDEN_H5, " ...")

# Read the data we already have
src <- H5File$new(GOLDEN_H5, mode = "r")
data_vals  <- src[["X/data"]][]
indices    <- src[["X/indices"]][]     # int32, 0-based gene indices
indptr_vec <- src[["X/indptr"]][]     # col pointers (ncells+1)
shape_vec  <- src[["X/shape"]][]      # [ngenes, ncells]
cell_names <- src[["obs/index"]][]
gene_names <- src[["var/index"]][]

# Read metadata columns
obs_members <- src[["obs"]]$ls()$name
meta_cols   <- obs_members[obs_members != "index"]
meta_data   <- list()
for (col in meta_cols) {
  meta_data[[col]] <- src[[paste0("obs/", col)]][]
}

# Read embeddings (2D datasets — hdf5r needs explicit [,] indexing)
pca_arr  <- src[["obsm/X_pca"]][,]    # returns [30, ncells] (stored transposed)
umap_arr <- src[["obsm/X_umap"]][,]   # returns [2, ncells]
src$close_all()

ngenes <- as.integer(shape_vec[1])
ncells <- as.integer(shape_vec[2])
message(sprintf("  %d genes x %d cells", ngenes, ncells))

# --- Write H5Seurat ---
if (file.exists(OUT_H5SEURAT)) file.remove(OUT_H5SEURAT)
f <- H5File$new(OUT_H5SEURAT, mode = "w")

# Top-level attributes
h5attr(f, "active.assay") <- "RNA"
h5attr(f, "version")      <- "3.1.5.9900"

# /cell.names
f[["cell.names"]] <- cell_names

# /assays/RNA/features
assays <- f$create_group("assays")
rna    <- assays$create_group("RNA")
rna[["features"]] <- gene_names
h5attr(rna, "key") <- "rna_"

# /assays/RNA/meta.features/ — per-gene metadata
# Seurat stores: vf.vst.mean, vf.vst.variance, vf.vst.variance.expected,
# vf.vst.variance.standardized, vf.vst.variable (logical)
# We write a representative subset so the reader can be tested.
mf_grp <- rna$create_group("meta.features")
# Mark which genes were selected as highly variable (top 2000)
n_hvg  <- 2000L
is_hvg <- c(rep(TRUE, n_hvg), rep(FALSE, ngenes - n_hvg))
# logicals → 0/1/2 int encoding (SeuratDisk convention)
mf_grp[["vf.vst.variable"]] <- as.integer(ifelse(is_hvg, 1L, 0L))
h5attr(mf_grp, "logicals") <- "vf.vst.variable"
# numeric columns
set.seed(42)
mf_grp[["vf.vst.mean"]]     <- runif(ngenes, 0, 5)
mf_grp[["vf.vst.variance"]] <- runif(ngenes, 0, 10)
h5attr(mf_grp, "colnames") <- c("vf.vst.variable", "vf.vst.mean", "vf.vst.variance")

# /assays/RNA/counts/ — CSC sparse group
counts_grp <- rna$create_group("counts")
counts_grp[["data"]]    <- as.double(data_vals)     # float64
counts_grp[["indices"]] <- as.integer(indices)      # int32
counts_grp[["indptr"]]  <- as.integer(indptr_vec)   # int32
h5attr(counts_grp, "dims") <- as.integer(c(ngenes, ncells))

# /meta.data/ — numeric columns directly, factors as groups
meta_grp <- f$create_group("meta.data")
h5attr(meta_grp, "colnames") <- meta_cols
h5attr(meta_grp, "_class")   <- "data.frame"

logicals <- character(0)
for (col in meta_cols) {
  vals <- meta_data[[col]]
  if (is.logical(vals)) {
    # Encode as 0/1/2 integer (FALSE/TRUE/NA)
    int_vals <- ifelse(is.na(vals), 2L, ifelse(vals, 1L, 0L))
    meta_grp[[col]] <- as.integer(int_vals)
    logicals <- c(logicals, col)
  } else if (is.factor(vals) || is.character(vals)) {
    if (!is.factor(vals)) vals <- factor(vals)
    col_grp <- meta_grp$create_group(col)
    col_grp[["values"]] <- as.integer(vals)          # 1-indexed codes
    col_grp[["levels"]] <- as.character(levels(vals))
  } else {
    meta_grp[[col]] <- as.double(vals)
  }
}
if (length(logicals) > 0) {
  h5attr(meta_grp, "logicals") <- logicals
}

# /reductions/
reds <- f$create_group("reductions")

# PCA: store as [ncells x 30] C-order
pca_grp <- reds$create_group("pca")
# pca_arr from hdf5r is already [ncells, 30] row-major; transpose to store
# as column-major like SeuratDisk does — we'll use t() so HDF5 sees [30, ncells]
# and the reader transposes back. Matches real SeuratDisk behaviour.
pca_grp[["cell.embeddings"]] <- pca_arr    # already [30, ncells], reader transposes
h5attr(pca_grp, "key")          <- "PC_"
h5attr(pca_grp, "active.assay") <- "RNA"

umap_grp <- reds$create_group("umap")
umap_grp[["cell.embeddings"]] <- umap_arr  # already [2, ncells]
h5attr(umap_grp, "key")          <- "UMAP_"
h5attr(umap_grp, "active.assay") <- "RNA"

f$close_all()
message("Written: ", OUT_H5SEURAT)
