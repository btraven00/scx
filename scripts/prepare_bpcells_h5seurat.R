#!/usr/bin/env Rscript
# prepare_bpcells_h5seurat.R
#
# Generates a BPCells-packed HDF5 fixture for benchmarking scx's BPCells decode
# path (decode_d1z / decode_for with Rayon parallelism).
#
# Uses BPCells' write_matrix_hdf5() to embed the matrix directly into the HDF5
# file at the Seurat v5 assay group path — no Seurat or SeuratDisk required.
# The resulting file is sufficient for scx's H5SeuratReader v5 BPCells path.
#
# Requires: HLCA core h5ad (run `bash scripts/download_large.sh` first).
# Run via:  pixi run -e bpcells prepare-bpcells-h5seurat
#
# Output:
#   tests/golden/seurat_v5_bpcells.h5seurat

suppressPackageStartupMessages(library(BPCells))

H5AD_IN  <- "tests/golden/hlca_core.h5ad"
H5SEURAT <- "tests/golden/seurat_v5_bpcells.h5seurat"
GROUP    <- "assays/RNA/layers/counts"

if (!file.exists(H5AD_IN)) {
  stop("HLCA core fixture not found: ", H5AD_IN,
       "\nRun: bash scripts/download_large.sh")
}

if (file.exists(H5SEURAT)) {
  message("Already exists: ", H5SEURAT, " — delete to regenerate.")
  quit(status = 0)
}

# Open X directly as an on-disk BPCells matrix — no RAM copy of the data.
message("Opening HLCA X as BPCells matrix from h5ad...")
mat <- open_matrix_anndata_hdf5(H5AD_IN)
message(sprintf("  shape: %d genes × %d cells", nrow(mat), ncol(mat)))

# Write the BPCells packed matrix directly into the HDF5 file at the Seurat v5
# assay group path.  This creates all datasets (idxptr, index_data, val_data,
# index_starts, shape, row_names, col_names, …) and the 'version' attribute.
message("Writing BPCells matrix to: ", H5SEURAT, " [", GROUP, "]")
write_matrix_hdf5(mat, H5SEURAT, group = GROUP)

size_mb <- file.size(H5SEURAT) / 1e6
message(sprintf("Done: %s (%.0f MB)", H5SEURAT, size_mb))
