#' Write an AnnData object to H5AD
#'
#' Thin wrapper around [anndataR::write_h5ad()]. Included for API symmetry.
#' Native streaming write via the SCX Rust engine is planned for 0.0.4.
#'
#' @param adata An `anndataR::InMemoryAnnData` object.
#' @param path  Output path for the `.h5ad` file.
#' @param compression Compression algorithm passed to [anndataR::write_h5ad()].
#'   Default `"gzip"`.
#' @return `path`, invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' adata <- read_h5seurat("pbmc3k.h5seurat")
#' write_h5ad(adata, "pbmc3k_out.h5ad")
#' }
write_h5ad <- function(adata, path, compression = "gzip") {
  anndataR::write_h5ad(adata, path, compression = compression)
  invisible(path)
}
