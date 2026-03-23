#!/usr/bin/env Rscript
# prepare_h5ad_reference.R
#
# Converts tests/golden/pbmc3k.h5seurat → tests/golden/pbmc3k_reference.h5ad
# using hdf5r (direct H5Seurat parse) + anndataR. No SeuratDisk dependency.
#
# H5Seurat schema (as written by prepare_h5seurat_test.R):
#   /cell.names                 string (ncells,)
#   /assays/RNA/features        string (ngenes,)
#   /assays/RNA/counts/         CSC sparse group
#     data                      float64 (nnz,)
#     indices                   int32   (nnz,)    0-based gene indices
#     indptr                    int32   (ncells+1) 0-based col pointers
#     attr: dims                int32   [ngenes, ncells]
#   /assays/RNA/meta.features/  per-gene metadata
#   /meta.data/                 cell metadata
#     <numeric_col>             float64 (ncells,)
#     <factor_col>/             group { values (1-indexed int), levels (string) }
#     attr: logicals            names of logical cols (encoded as 0/1/2)
#   /reductions/pca/cell.embeddings   float64 [30, ncells]  (stored transposed)
#   /reductions/umap/cell.embeddings  float64 [2,  ncells]
#
# Usage:
#   pixi run -e test prepare-h5ad-ref

suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(anndataR)
})

IN_H5SEURAT <- "tests/golden/pbmc3k.h5seurat"
OUT_H5AD    <- "tests/golden/pbmc3k_reference.h5ad"

if (!file.exists(IN_H5SEURAT)) {
  stop("Run 'pixi run -e test prepare-h5seurat' first: ", IN_H5SEURAT)
}
if (file.exists(OUT_H5AD)) {
  message("Reference H5AD already exists — skipping. Delete to regenerate:")
  message("  ", OUT_H5AD)
  quit(save = "no", status = 0)
}

# hdf5r's $exists() throws (not returns FALSE) when an intermediate group is
# missing. Wrap all existence checks.
safe_exists <- function(obj, name) {
  tryCatch(obj$exists(name), error = function(e) FALSE)
}

message("Opening H5Seurat: ", IN_H5SEURAT)
h5 <- H5File$new(IN_H5SEURAT, mode = "r")

# --- cell and gene names ---
cell_names <- h5[["cell.names"]][]
gene_names <- h5[["assays/RNA/features"]][]
n_cells    <- length(cell_names)
n_genes    <- length(gene_names)
message(sprintf("  %d cells x %d genes", n_cells, n_genes))

# --- count matrix (CSC: genes x cells) ---
counts_grp <- h5[["assays/RNA/counts"]]
data_vals  <- counts_grp[["data"]][]
# indices are 0-based gene indices; sparseMatrix i= is 1-based
indices    <- counts_grp[["indices"]][] + 1L
# indptr are 0-based column pointers; sparseMatrix p= is 0-based (keep as-is)
indptr     <- counts_grp[["indptr"]][]

X_csc <- sparseMatrix(
  i    = indices,
  p    = indptr,
  x    = as.numeric(data_vals),
  dims = c(n_genes, n_cells),
  dimnames = list(gene_names, cell_names),
  repr = "C"
)
# H5AD X is obs x vars (cells x genes) stored as CSR
X_csr <- as(t(X_csc), "RsparseMatrix")

# --- obs metadata ---
obs <- data.frame(row.names = cell_names, check.names = FALSE)
if (safe_exists(h5, "meta.data")) {
  meta_grp  <- h5[["meta.data"]]
  logicals  <- tryCatch(h5attr(meta_grp, "logicals"), error = function(e) character(0))

  for (col in names(meta_grp)) {
    item <- meta_grp[[col]]
    if (inherits(item, "H5Group")) {
      # factor: values are 1-indexed R integer codes
      lvls <- item[["levels"]][]
      vals <- item[["values"]][]
      obs[[col]] <- factor(lvls[vals], levels = lvls)
    } else {
      raw <- item[]
      if (col %in% logicals) {
        # 0=FALSE, 1=TRUE, 2=NA (SeuratDisk logical encoding)
        obs[[col]] <- ifelse(raw == 2L, NA, raw == 1L)
      } else {
        obs[[col]] <- raw
      }
    }
  }
}

# --- var metadata ---
var <- data.frame(row.names = gene_names, check.names = FALSE)
if (safe_exists(h5, "assays") && safe_exists(h5[["assays/RNA"]], "meta.features")) {
  mf_grp      <- h5[["assays/RNA/meta.features"]]
  mf_logicals <- tryCatch(h5attr(mf_grp, "logicals"), error = function(e) character(0))
  for (col in names(mf_grp)) {
    raw <- mf_grp[[col]][]
    if (col %in% mf_logicals) {
      var[[col]] <- ifelse(raw == 2L, NA, raw == 1L)
    } else {
      var[[col]] <- raw
    }
  }
}

# --- obsm (cell embeddings) ---
obsm <- list()
if (safe_exists(h5, "reductions")) {
  reds_grp <- h5[["reductions"]]
  for (rd in names(reds_grp)) {
    if (safe_exists(reds_grp[[rd]], "cell.embeddings")) {
      mat <- reds_grp[[rd]][["cell.embeddings"]][,]
      # Stored as [n_dims, n_cells]; transpose to [n_cells, n_dims]
      if (nrow(mat) != n_cells) mat <- t(mat)
      rownames(mat) <- cell_names
      obsm[[paste0("X_", rd)]] <- mat
    }
  }
}

h5$close_all()

# --- build and write AnnData ---
message("Building AnnData object ...")
adata <- AnnData(
  X    = X_csr,
  obs  = obs,
  var  = var,
  obsm = obsm
)

message("Writing reference H5AD via anndataR: ", OUT_H5AD)
write_h5ad(adata, OUT_H5AD)

# Sanity-check
ref <- read_h5ad(OUT_H5AD)
message(sprintf("  Verified: %d obs x %d vars, %d obs cols, obsm keys: %s",
                nrow(ref$obs), nrow(ref$var),
                ncol(ref$obs),
                paste(names(ref$obsm), collapse = ", ")))
message("Done: ", OUT_H5AD)
