#' Inspect a single-cell file without loading data into memory
#'
#' Reads only the metadata (shape, column names, embedding keys, etc.) from a
#' supported file. No count data is loaded. Memory usage is minimal regardless
#' of dataset size.
#'
#' @param path       Path to the file (`.h5seurat`, `.h5ad`, or `.h5`).
#' @param chunk_size Cells per internal streaming chunk. Default `5000L`.
#'   Only affects obs/var metadata reads; does not load the count matrix.
#'
#' @return A named list with:
#' \describe{
#'   \item{format}{Detected file format string.}
#'   \item{n_obs}{Number of observations (cells).}
#'   \item{n_vars}{Number of variables (genes/features).}
#'   \item{obs_cols}{Character vector of cell metadata column names.}
#'   \item{var_cols}{Character vector of feature metadata column names.}
#'   \item{obsm_keys}{Character vector of low-dimensional embedding keys.}
#'   \item{layers}{Named list with parallel vectors `name`, `nnz`, `nnz_q1`,
#'     `nnz_med`, `nnz_q3`, `nnz_max` — one entry per layer.}
#'   \item{uns_keys}{Character vector of unstructured metadata keys.}
#'   \item{obsp}{Named list with parallel vectors `name`, `nnz`, `nnz_q1`,
#'     `nnz_med`, `nnz_q3`, `nnz_max` — one entry per obsp matrix.}
#'   \item{varm_keys}{Character vector of feature embedding keys.}
#' }
#' @export
#'
#' @examples
#' \dontrun{
#' info <- inspect("pbmc3k.h5seurat")
#' info$n_obs     # 2700
#' info$obs_cols  # c("orig.ident", "nCount_RNA", ...)
#' info$obsm_keys # c("X_pca", "X_umap")
#' }
inspect <- function(path, chunk_size = 5000L) {
  scx_inspect(
    input      = path.expand(path),
    chunk_size = as.integer(chunk_size)
  )
}
