#!/usr/bin/env Rscript
# Add the BPCells-encoded counts matrix to tests/fixtures/tiny/tiny_v5_bpcells.h5seurat.
#
# scripts/make_tiny_fixtures.py writes the file's names and metadata first;
# this step needs the real BPCells encoder, hence R. The counts come from
# expected.json so both scripts describe the same dataset.
#
# The file's bytes change on every run (HDF5 object timestamps); its content
# does not, so regenerate only when the dataset changes.
#
# usage: pixi run -e bpcells make-tiny-fixtures-bpcells

suppressPackageStartupMessages({
  library(BPCells)
  library(Matrix)
})

dir <- "tests/fixtures/tiny"
h5 <- file.path(dir, "tiny_v5_bpcells.h5seurat")
stopifnot(file.exists(h5))
e <- jsonlite::fromJSON(file.path(dir, "expected.json"))

counts <- t(e$counts) # genes x cells, as Seurat stores it
dimnames(counts) <- list(e$var_names, e$obs_names)
m <- as(as(counts, "CsparseMatrix"), "generalMatrix")
bp <- convert_matrix_type(as(m, "IterableMatrix"), "uint32_t")
invisible(write_matrix_hdf5(bp, h5, "assays/RNA/layers/counts"))
message("wrote BPCells counts into ", h5)
