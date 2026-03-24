#' Read an H5AD file into an AnnData object
#'
#' Thin wrapper around [anndataR::read_h5ad()]. Included for API symmetry so
#' callers can use `picklerick::read_h5ad()` alongside `read_h5seurat()`.
#'
#' @param path Path to the `.h5ad` file.
#' @return An `anndataR::InMemoryAnnData` object.
#' @export
#'
#' @examples
#' \dontrun{
#' adata <- read_h5ad("pbmc3k.h5ad")
#' adata$n_obs   # 2700
#' adata$X       # sparse matrix (obs x vars)
#' }
read_h5ad <- function(path) {
  anndataR::read_h5ad(path)
}

#' Read an H5Seurat file into an AnnData object
#'
#' Converts the H5Seurat file to a temporary H5AD via the SCX streaming
#' engine, then reads the result with [anndataR::read_h5ad()]. The temporary
#' file is deleted on exit. Memory usage during conversion is bounded by
#' `chunk_size`.
#'
#' @param path Path to the `.h5seurat` file.
#' @param assay Seurat assay to read. Default `"RNA"`.
#' @param layer Seurat layer to read (`"counts"` or `"data"`). Default `"counts"`.
#' @param chunk_size Cells per streaming chunk. Default `5000L`.
#' @param dtype Numeric type for the output matrix: `"f32"`, `"f64"`, `"i32"`,
#'   or `"u32"`. Default `"f32"`.
#' @return An `anndataR::InMemoryAnnData` object.
#' @export
#'
#' @examples
#' \dontrun{
#' adata <- read_h5seurat("pbmc3k.h5seurat")
#' adata$obs          # cell metadata data.frame
#' adata$obsm$X_pca   # PCA matrix
#' }
read_h5seurat <- function(path,
                           assay      = "RNA",
                           layer      = "counts",
                           chunk_size = 5000L,
                           dtype      = "f32") {
  tmp <- tempfile(fileext = ".h5ad")
  on.exit(unlink(tmp), add = TRUE)

  convert(
    input      = path,
    output     = tmp,
    chunk_size = chunk_size,
    dtype      = dtype,
    assay      = assay,
    layer      = layer
  )

  anndataR::read_h5ad(tmp)
}

#' Read any supported single-cell format into an AnnData object
#'
#' Format-agnostic wrapper. The format is auto-detected from the file content
#' (not the extension) by the SCX engine.
#'
#' For H5AD inputs this delegates directly to [anndataR::read_h5ad()] without
#' invoking the SCX binary. For all other formats, the file is first converted
#' to a temporary H5AD.
#'
#' @param path Path to the input file (`.h5seurat`, `.h5ad`, or `.h5`).
#' @param ... Additional arguments passed to [read_h5seurat()] or [read_h5ad()].
#' @return An `anndataR::InMemoryAnnData` object.
#' @export
#'
#' @examples
#' \dontrun{
#' adata <- read_dataset("pbmc3k.h5seurat")
#' adata <- read_dataset("pbmc3k.h5ad")
#' }
read_dataset <- function(path, ...) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "h5ad") {
    read_h5ad(path)
  } else {
    read_h5seurat(path, ...)
  }
}

#' Read an H5Seurat file directly into a Seurat object
#'
#' Converts the file to a temporary H5AD via the SCX engine, reads it with
#' [anndataR::read_h5ad()], then coerces to a `Seurat` object via
#' [anndataR::as_Seurat()]. Requires the **Seurat** package (>= 5).
#'
#' @inheritParams read_h5seurat
#' @return A `Seurat` object.
#' @export
#'
#' @examples
#' \dontrun{
#' seu <- read_seurat("pbmc3k.h5seurat")
#' }
read_seurat <- function(path, ...) {
  if (!requireNamespace("Seurat", quietly = TRUE))
    stop("Seurat package required for read_seurat()", call. = FALSE)
  adata <- read_h5seurat(path, ...)
  adata$as_Seurat()
}

#' Read an H5Seurat file directly into a SingleCellExperiment object
#'
#' Converts the file to a temporary H5AD via the SCX engine, reads it with
#' [anndataR::read_h5ad()], then coerces to a `SingleCellExperiment` via
#' [anndataR::as_SingleCellExperiment()]. Requires the
#' **SingleCellExperiment** package.
#'
#' @inheritParams read_h5seurat
#' @return A `SingleCellExperiment` object.
#' @export
#'
#' @examples
#' \dontrun{
#' sce <- read_sce("pbmc3k.h5seurat")
#' }
read_sce <- function(path, ...) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE))
    stop("SingleCellExperiment package required for read_sce()", call. = FALSE)
  adata <- read_h5seurat(path, ...)
  adata$as_SingleCellExperiment()
}
