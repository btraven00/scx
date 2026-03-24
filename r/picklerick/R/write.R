#' Write an AnnData object to H5AD
#'
#' Thin wrapper around [anndataR::write_h5ad()]. Included for API symmetry.
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

#' Write an AnnData object to H5Seurat format
#'
#' Serialises `adata` to a temporary H5AD, then streams it through the SCX
#' engine to produce an H5Seurat file. Memory usage is bounded by `chunk_size`.
#'
#' @param adata An `anndataR::InMemoryAnnData` object.
#' @param path  Output path for the `.h5seurat` file.
#' @param assay Seurat assay name to write. Default `"RNA"`.
#' @param chunk_size Number of cells per streaming chunk. Default `5000L`.
#' @return `path`, invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' adata <- read_h5ad("pbmc3k.h5ad")
#' write_h5seurat(adata, "pbmc3k_out.h5seurat")
#' }
write_h5seurat <- function(adata, path, assay = "RNA", chunk_size = 5000L) {
  tmp <- tempfile(fileext = ".h5ad")
  on.exit(unlink(tmp), add = TRUE)
  anndataR::write_h5ad(adata, tmp)

  if (.native_available()) {
    scx_write_h5seurat(
      input      = tmp,
      output     = path.expand(path),
      chunk_size = as.integer(chunk_size),
      assay      = as.character(assay)
    )
  } else {
    .write_h5seurat_via_cli(tmp, path, chunk_size, assay)
  }
  invisible(path)
}

.write_h5seurat_via_cli <- function(input, output, chunk_size, assay) {
  bin <- .scx_binary()
  args <- c("convert", input, output,
            "--chunk-size", as.character(chunk_size),
            "--assay",      assay)
  status <- system2(bin, args = args, stdout = "", stderr = "")
  if (status != 0L) {
    stop(sprintf("scx convert failed (exit %d)", status), call. = FALSE)
  }
}
