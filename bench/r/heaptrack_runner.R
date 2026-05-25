#!/usr/bin/env Rscript
# Minimal driver — one read_h5ad call with everything else stripped.
# Intended to be run under heaptrack so the trace is dominated by the read.
#
# Usage:
#   heaptrack Rscript bench/r/heaptrack_runner.R [dataset] [mode]
#
# dataset: pbmc3k (default) | norman
# mode:    SingleCellExperiment (default) | list

suppressPackageStartupMessages({
  library(picklerick)
  library(SingleCellExperiment)
})

args  <- commandArgs(trailingOnly = TRUE)
ds    <- if (length(args) >= 1) args[[1]] else "pbmc3k"
mode  <- if (length(args) >= 2) args[[2]] else "SingleCellExperiment"

path <- switch(ds,
  pbmc3k = "tests/golden/pbmc3k_reference.h5ad",
  norman = "tests/golden/norman_subset.h5ad",
  stop("unknown dataset: ", ds)
)

# Single call, no warmup — heaptrack stats are process-wide so the warmup
# would dominate the trace. First-call S4 dispatch / class registration is
# present in both baseline and optimized runs, so it cancels out for the
# alloc-count diff (the bytes-per-alloc and structure may still differ).
res <- picklerick::read_h5ad(path, as = mode)
cat("done — read", ds, "as", mode, "\n")
