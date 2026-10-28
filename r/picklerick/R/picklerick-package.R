#' @details
#' Start with [read_h5ad()] to read a file into a `SingleCellExperiment` or
#' `Seurat` object, [inspect()] to look at a file without reading its
#' matrix, and [convert()] to stream one format into another.
#' `vignette("picklerick")` runs each of them on the files shipped in
#' `inst/extdata`.
#'
#' @section The scx binary:
#' The engine is compiled into the package, so no external program is
#' needed. When the compiled entry points are unavailable, [convert()] falls
#' back to the `scx` command-line binary, found in this order:
#' 1. `options(picklerick.scx_binary = "/path/to/scx")`
#' 2. the `PICKLERICK_SCX` environment variable
#' 3. `scx` on the `PATH`
#'
#' @keywords internal
"_PACKAGE"

#' @useDynLib picklerick, .registration = TRUE
#' @importFrom methods setClass setMethod show
#' @importFrom DelayedArray chunkdim
#' @importFrom S4Arrays extract_array is_sparse
#' @importFrom SparseArray extract_sparse_array
NULL
