#!/usr/bin/env Rscript
# Generate BPCells golden fixtures for the Rust compatibility test suite.
# Output: tests/golden/bpcells/<fixture_name>/
#
# Run via:   pixi run -e test prepare-bpcells
# Or:        Rscript scripts/prepare_bpcells_fixtures.R

suppressPackageStartupMessages({
  library(BPCells)
  library(Matrix)
})

OUT <- file.path("tests", "golden", "bpcells")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

write_packed   <- function(bm, name) write_matrix_dir(bm, file.path(OUT, name), compress = TRUE,  overwrite = TRUE)
write_unpacked <- function(bm, name) write_matrix_dir(bm, file.path(OUT, name), compress = FALSE, overwrite = TRUE)
as_bm          <- function(m) convert_matrix_type(as(m, "IterableMatrix"), "uint32_t")

# ---------------------------------------------------------------------------
# Canonical 4×5 test matrix (column-major, uint32)
#
#       col1 col2 col3 col4 col5
# row1:    1    0    3    0    0
# row2:    0    2    0    4    0
# row3:    5    0    0    0    6
# row4:    0    0    7    0    8
#
# Expected dense reconstruction (0-indexed):
#   nnz = 8, values by (col,row): (0,0)=1,(0,2)=5,(1,1)=2,(2,0)=3,(2,3)=7,
#                                  (3,1)=4,(4,2)=6,(4,3)=8
# ---------------------------------------------------------------------------
RAW <- sparseMatrix(
  i = c(1, 3, 2, 1, 4, 2, 3, 4),
  j = c(1, 1, 2, 3, 3, 4, 5, 5),
  x = c(1L, 5L, 2L, 3L, 7L, 4L, 6L, 8L),
  dims = c(4L, 5L),
  dimnames = list(paste0("r", 1:4), paste0("c", 1:5))
)

message("Writing synth_packed_uint_csc   ...")
write_packed(as_bm(RAW), "synth_packed_uint_csc")

message("Writing synth_unpacked_uint_csc ...")
write_unpacked(as_bm(RAW), "synth_unpacked_uint_csc")

# Float variant — val stays uncompressed even in packed mode
message("Writing synth_packed_float_csc  ...")
write_packed(
  convert_matrix_type(as_bm(RAW), "float"),
  "synth_packed_float_csc"
)

# Double variant
message("Writing synth_packed_double_csc ...")
write_packed(
  convert_matrix_type(as_bm(RAW), "double"),
  "synth_packed_double_csc"
)

# CSR (row-major) — storage_order = "row"
message("Writing synth_packed_uint_csr   ...")
write_packed(t(as_bm(t(RAW))), "synth_packed_uint_csr")

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

# Empty matrix (0 nnz)
message("Writing synth_empty             ...")
EMPTY <- sparseMatrix(i = integer(0), j = integer(0), x = integer(0), dims = c(4L, 5L))
write_packed(as_bm(EMPTY), "synth_empty")

# Single-column matrix
message("Writing synth_one_col           ...")
write_packed(as_bm(RAW[, 1, drop = FALSE]), "synth_one_col")

# Large values (close to .Machine$integer.max = 2^31-1)
message("Writing synth_large_vals        ...")
LARGE <- sparseMatrix(
  i = c(1, 2, 3, 4),
  j = c(1, 2, 3, 4),
  x = c(.Machine$integer.max, 2L^30L, 1L, 1000000L),
  dims = c(4L, 4L),
  dimnames = list(paste0("r", 1:4), paste0("c", 1:4))
)
write_packed(as_bm(LARGE), "synth_large_vals")

# Exactly 128 nnz in column 1 — hits the chunk boundary exactly
message("Writing synth_128_boundary      ...")
M128 <- sparseMatrix(
  i = 1:128, j = rep(1L, 128), x = 1:128,
  dims = c(128L, 2L),
  dimnames = list(paste0("r", 1:128), c("c1", "c2"))
)
write_packed(as_bm(M128), "synth_128_boundary")

# Exactly 256 nnz — two full BP-128 chunks
message("Writing synth_256_boundary      ...")
M256 <- sparseMatrix(
  i = 1:256, j = rep(1L, 256), x = 1:256,
  dims = c(256L, 2L),
  dimnames = list(paste0("r", 1:256), c("c1", "c2"))
)
write_packed(as_bm(M256), "synth_256_boundary")

# ---------------------------------------------------------------------------
# Write a companion JSON manifest so Rust tests can load expected values
# without hardcoding them in source.
# ---------------------------------------------------------------------------
library(jsonlite)

dense_to_list <- function(m) {
  dm <- as.matrix(m)
  list(
    nrow   = nrow(dm),
    ncol   = ncol(dm),
    values = as.vector(t(dm))  # row-major flat for easy indexing in Rust
  )
}

manifest <- list(
  synth_packed_uint_csc   = dense_to_list(RAW),
  synth_unpacked_uint_csc = dense_to_list(RAW),
  synth_packed_float_csc  = dense_to_list(RAW),   # same values, different dtype
  synth_packed_double_csc = dense_to_list(RAW),
  synth_packed_uint_csr   = dense_to_list(RAW),
  synth_empty             = dense_to_list(EMPTY),
  synth_one_col           = dense_to_list(RAW[, 1, drop = FALSE]),
  synth_large_vals        = dense_to_list(LARGE),
  synth_128_boundary      = dense_to_list(M128),
  synth_256_boundary      = dense_to_list(M256)
)

write_json(manifest, file.path(OUT, "manifest.json"), auto_unbox = TRUE, pretty = TRUE)
message("Manifest written to ", file.path(OUT, "manifest.json"))

# ---------------------------------------------------------------------------
# Optional: pbmc3k from h5seurat (requires SeuratDisk — may not be installed)
# ---------------------------------------------------------------------------
pbmc_h5 <- file.path("tests", "golden", "pbmc3k.h5seurat")
if (file.exists(pbmc_h5) && requireNamespace("SeuratDisk", quietly = TRUE)) {
  message("Writing pbmc3k_packed_uint_csc  ...")
  pbmc <- SeuratDisk::LoadH5Seurat(pbmc_h5)
  bm   <- as(pbmc[["RNA"]]$counts, "IterableMatrix")
  write_packed(bm, "pbmc3k_packed_uint_csc")
  message("pbmc3k fixture written (nnz = ", sum(pbmc[["RNA"]]$counts@x > 0), ")")
} else {
  message("SKIP: pbmc3k.h5seurat or SeuratDisk not found; skipping pbmc3k fixture")
}

message("\nAll BPCells fixtures written to ", OUT)
