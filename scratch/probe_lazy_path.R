#!/usr/bin/env Rscript
# Localize where the lazy=TRUE path spends its 1209 ms on norman.
#
# Decomposes .as_sce_lazy stage-by-stage. Compares against eager so we can
# pin the gap (lazy 1209 ms vs eager 1016 ms = +193 ms; +14 % regression).
#
# Usage:  Rscript scratch/probe_lazy_path.R [dataset]
# dataset: pbmc3k (default) or norman

suppressPackageStartupMessages({
  library(picklerick)
  library(SingleCellExperiment)
  library(HDF5Array)
})

args <- commandArgs(trailingOnly = TRUE)
ds   <- if (length(args) >= 1) args[[1]] else "norman"
path <- switch(ds,
  pbmc3k = "tests/golden/pbmc3k_reference.h5ad",
  norman = "tests/golden/norman_subset.h5ad",
  stop("unknown dataset: ", ds))

tic <- function() Sys.time()
toc <- function(t0, label) {
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
  cat(sprintf("  %-50s %8.1f ms\n", label, dt))
  invisible(dt)
}

# Warmup so dispatch / dyn.load / HDF5Array setup cost is amortized.
invisible(read_h5ad(path, as = "SingleCellExperiment"))
invisible(read_h5ad(path, as = "SingleCellExperiment", lazy = TRUE))
gc(verbose = FALSE, full = TRUE)

cat(sprintf("=== %s  (%.1f MB) ===\n", ds, file.info(path)$size / 1e6))

cat("\n--- eager (.as_sce) reference ---\n")
t0 <- tic(); res_eager <- read_h5ad(path, as = "SingleCellExperiment");
total_eager <- toc(t0, "TOTAL eager read_h5ad SCE")

cat("\n--- lazy path, stage by stage ---\n")
t0 <- tic()
raw <- picklerick:::scx_read(path, 5000L, read_x = FALSE)
toc(t0, "scx_read(read_x = FALSE) [metadata only]")

t0 <- tic()
x_lazy <- HDF5Array::H5SparseMatrix(filepath = path, group = "/X")
toc(t0, "H5SparseMatrix(filepath, group)")

t0 <- tic()
if (nrow(x_lazy) == raw$n_obs && ncol(x_lazy) == raw$n_vars)
  x_lazy <- t(x_lazy)
toc(t0, "t(x_lazy)")

t0 <- tic()
dimnames(x_lazy) <- list(raw$var_index, raw$obs_index)
toc(t0, "dimnames<-(x_lazy)")

t0 <- tic()
obs_df <- picklerick:::.cols_to_df(raw$obs_cols, raw$obs_index)
var_df <- picklerick:::.cols_to_df(raw$var_cols, raw$var_index)
toc(t0, ".cols_to_df obs + var")

t0 <- tic()
cd <- S4Vectors::DataFrame(obs_df)
rd <- S4Vectors::DataFrame(var_df)
toc(t0, "S4Vectors::DataFrame obs + var")

t0 <- tic()
reduced <- lapply(names(raw$obsm), function(nm) {
  m <- raw$obsm[[nm]]
  if (!is.null(m) && nrow(m) == length(raw$obs_index)) rownames(m) <- raw$obs_index
  m
})
names(reduced) <- names(raw$obsm)
toc(t0, "obsm processing")

t0 <- tic()
sce <- SingleCellExperiment::SingleCellExperiment(
  assays      = list(counts = x_lazy),
  colData     = cd,
  rowData     = rd,
  reducedDims = reduced
)
toc(t0, "SingleCellExperiment constructor")

cat("\n--- full lazy call (sanity) ---\n")
t0 <- tic(); res_lazy <- read_h5ad(path, as = "SingleCellExperiment", lazy = TRUE);
total_lazy <- toc(t0, "TOTAL lazy read_h5ad SCE")

cat(sprintf("\n--> eager %.1f ms / lazy %.1f ms / gap %+.1f ms (%+.1f %%)\n",
            total_eager, total_lazy,
            total_lazy - total_eager,
            100 * (total_lazy / total_eager - 1)))
