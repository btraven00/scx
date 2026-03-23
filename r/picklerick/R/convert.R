#' Convert a single-cell file to another format
#'
#' Streams the input file through the SCX engine and writes the result to
#' `output`. No R objects are constructed; memory usage is bounded by
#' `chunk_size` regardless of dataset size.
#'
#' Supported input formats (auto-detected by content):
#' - `.h5seurat` — SeuratDisk H5Seurat (Seurat v3/v4)
#' - `.h5ad`     — AnnData H5AD
#' - `.h5`       — SCX internal HDF5 schema
#'
#' @param input  Path to the input file.
#' @param output Path to the output `.h5ad` file.
#' @param chunk_size Number of cells per streaming chunk. Default `5000L`.
#' @param dtype Output numeric type: `"f32"`, `"f64"`, `"i32"`, or `"u32"`.
#'   Default `"f32"`.
#' @param assay Seurat assay to convert (H5Seurat inputs only). Default `"RNA"`.
#' @param layer Seurat layer to convert (H5Seurat inputs only). Default `"counts"`.
#'
#' @return `output` path, invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' convert("pbmc3k.h5seurat", "pbmc3k.h5ad")
#' convert("pbmc3k.h5seurat", "pbmc3k_f64.h5ad", dtype = "f64", chunk_size = 2000L)
#' }
convert <- function(input, output,
                    chunk_size = 5000L,
                    dtype      = "f32",
                    assay      = "RNA",
                    layer      = "counts") {
  if (.native_available()) {
    scx_convert(
      input      = path.expand(input),
      output     = path.expand(output),
      chunk_size = as.integer(chunk_size),
      dtype      = as.character(dtype),
      assay      = as.character(assay),
      layer      = as.character(layer)
    )
  } else {
    .convert_via_cli(input, output, chunk_size, dtype, assay, layer)
  }
  invisible(output)
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.native_available <- function() {
  # Native Rust HDF5 bindings share libhdf5.so with R's hdf5r/rhdf5.
  # HDF5 global state (property list IDs) gets corrupted when the Rust
  # hdf5 crate initialises alongside R's HDF5 runtime.
  # Disabled until hdf5-sys uses static HDF5 linking.
  FALSE
}

.convert_via_cli <- function(input, output, chunk_size, dtype, assay, layer) {
  bin <- .scx_binary()
  args <- c(
    "convert",
    input, output,
    "--chunk-size", as.character(chunk_size),
    "--dtype",      dtype,
    "--assay",      assay,
    "--layer",      layer
  )
  status <- system2(bin, args = args, stdout = "", stderr = "")
  if (status != 0L) {
    stop(sprintf("scx convert failed (exit %d)", status), call. = FALSE)
  }
}
