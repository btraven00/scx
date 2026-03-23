#!/usr/bin/env Rscript
# prepare_test_data.R
#
# Downloads PBMC 3k, processes with Seurat, and exports to a simple HDF5
# schema that scx-core reads directly (no SeuratDisk dependency).
#
# Uses hdf5r which writes variable-length UTF-8 strings — compatible with
# the hdf5 Rust crate's VarLenUnicode type.
#
# Output:
#   tests/golden/pbmc3k.h5
#   tests/golden/pbmc3k_meta.json
#
# Schema written to tests/golden/pbmc3k.h5:
#   /X/data      float32 (nnz,)       - count values (CSC: gene-major)
#   /X/indices   int32   (nnz,)       - row (gene) indices
#   /X/indptr    float64 (ncells+1,)  - column (cell) pointers
#   /X/shape     float64 (2,)         - [ngenes, ncells]
#   attrs on /X/data: format="csc", dtype="float32"
#   /obs/index   string  (ncells,)    - cell barcodes
#   /obs/<col>   typed   (ncells,)    - per-cell metadata
#   /var/index   string  (ngenes,)    - gene names
#   /obsm/X_pca  float64 (ncells,30)  - written C-order (row-major)
#   /obsm/X_umap float64 (ncells,2)
#
# Usage:
#   pixi run prepare-test-data

suppressPackageStartupMessages({
  library(Seurat)
  library(hdf5r)
  library(jsonlite)
})

OUT_DIR  <- "tests/golden"
OUT_H5   <- file.path(OUT_DIR, "pbmc3k.h5")
OUT_META <- file.path(OUT_DIR, "pbmc3k_meta.json")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

DATA_URL <- "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"

if (file.exists(OUT_H5) && file.exists(OUT_META)) {
  message("Golden files already exist — skipping download. Delete to regenerate:")
  message("  ", OUT_H5)
  message("  ", OUT_META)
  quit(save = "no", status = 0)
}

# --- Download and extract ---
message("Downloading PBMC 3k from 10x...")
tmp_tar <- tempfile(fileext = ".tar.gz")
download.file(DATA_URL, tmp_tar, quiet = FALSE, mode = "wb")
tmp_dir <- tempfile()
dir.create(tmp_dir)
untar(tmp_tar, exdir = tmp_dir)
unlink(tmp_tar)

matrix_dir <- file.path(tmp_dir, "filtered_gene_bc_matrices", "hg19")
stopifnot(dir.exists(matrix_dir))

# --- Load and process ---
message("Loading into Seurat...")
counts <- Read10X(data.dir = matrix_dir)
obj <- CreateSeuratObject(counts = counts, min.cells = 3, min.features = 200)
message(sprintf("  %d cells x %d genes", ncol(obj), nrow(obj)))

message("Running NormalizeData -> PCA -> UMAP...")
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = 30, verbose = FALSE)
obj <- RunUMAP(obj, dims = 1:30, verbose = FALSE)

# --- Extract raw counts (dgCMatrix, genes x cells, CSC) ---
mat <- LayerData(obj, assay = "RNA", layer = "counts")
stopifnot(inherits(mat, "dgCMatrix"))
ncells <- ncol(mat)
ngenes <- nrow(mat)
nnz    <- length(mat@x)
message(sprintf("  counts: %d genes x %d cells, %d nnz", ngenes, ncells, nnz))

# --- Write HDF5 with hdf5r (VarLen UTF-8 strings) ---
if (file.exists(OUT_H5)) file.remove(OUT_H5)
f <- H5File$new(OUT_H5, mode = "w")

# /X — sparse CSC matrix
X <- f$create_group("X")
X[["data"]]    <- mat@x                          # float64 values
X[["indices"]] <- mat@i                          # int32 gene indices
X[["indptr"]]  <- mat@p                          # int32 col pointers
X[["shape"]]   <- c(ngenes, ncells)              # [ngenes, ncells]
X[["data"]]$create_attr("format", "csc")
X[["data"]]$create_attr("dtype",  "float32")     # logical dtype (values are raw counts)

# /obs — cell metadata
obs <- f$create_group("obs")
obs[["index"]] <- colnames(obj)                  # VarLen UTF-8 strings
meta <- obj@meta.data
for (col in colnames(meta)) {
  vals <- meta[[col]]
  if (is.factor(vals)) vals <- as.character(vals)
  obs[[col]] <- vals
}

# /var — gene metadata
var_grp <- f$create_group("var")
var_grp[["index"]] <- rownames(obj)              # VarLen UTF-8 strings

# /obsm — embeddings, written in C order (row-major: ncells x n_components)
obsm <- f$create_group("obsm")
pca  <- Embeddings(obj, "pca")                   # R: ncells x 30 (col-major in R)
umap <- Embeddings(obj, "umap")                  # R: ncells x 2
# hdf5r writes matrices in C order by default
obsm[["X_pca"]]  <- t(pca)   # transpose so HDF5 stores [ncells, 30] in C order
obsm[["X_umap"]] <- t(umap)

f$close_all()
message(sprintf("Written: %s", OUT_H5))

# --- Metadata sidecar for Rust test assertions ---
meta_out <- list(
  n_obs          = ncells,
  n_vars         = ngenes,
  nnz            = nnz,
  obs_cols       = colnames(obj@meta.data),
  embeddings     = list(
    X_pca  = dim(pca),
    X_umap = dim(umap)
  ),
  seurat_version = as.character(packageVersion("Seurat")),
  schema_version = "2"
)
writeLines(toJSON(meta_out, auto_unbox = TRUE, pretty = TRUE), OUT_META)
message(sprintf("Written: %s", OUT_META))
message("Done.")
