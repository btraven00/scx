#' Write a SingleCellExperiment to an H5AD file
#'
#' Materializes the counts matrix as a `dgCMatrix` and hands it to the SCX
#' Rust engine, which writes the AnnData layout directly. v1 writes only:
#' `X` (counts assay), `obs` (colData), `var` (rowData), and `uns` (metadata
#' coerced to JSON via jsonlite). `obsm`, `obsp`, `varm`, and additional
#' layers are not yet supported and will be silently dropped.
#'
#' @param sce      A `SingleCellExperiment` (or any object with `counts()`,
#'   `colData()`, `rowData()`, `metadata()`).
#' @param path     Output `.h5ad` path.
#' @param assay    Name of the assay to use as `X`. Default `"counts"`.
#' @param dtype    Output X dtype: `"f32"`, `"f64"`, `"i32"`, `"u32"`.
#'   Default `"f32"` (matches scx convert default).
#'
#' @return `path`, invisibly.
#' @export
write_h5ad <- function(sce, path, assay = "counts", dtype = "f32") {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("write_h5ad requires the Matrix package.", call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("write_h5ad requires the jsonlite package.", call. = FALSE)
  }

  # ---- counts: genes x cells (CSC dgCMatrix) is structurally identical to
  # ---- cells x genes (CSR) — same indptr/indices/data triplet.
  m <- SummarizedExperiment::assay(sce, assay)
  m <- methods::as(m, "CsparseMatrix")
  m <- methods::as(m, "dgCMatrix")

  n_genes <- nrow(m)
  n_cells <- ncol(m)

  obs_index <- colnames(sce)
  if (is.null(obs_index)) obs_index <- as.character(seq_len(n_cells))
  var_index <- rownames(sce)
  if (is.null(var_index)) var_index <- as.character(seq_len(n_genes))

  obs_cols <- .df_to_named_list(SummarizedExperiment::colData(sce))
  var_cols <- .df_to_named_list(SummarizedExperiment::rowData(sce))

  uns <- as.list(S4Vectors::metadata(sce))
  uns <- .sanitize_for_json(uns)
  uns_json <- if (length(uns) == 0L) "{}" else
    jsonlite::toJSON(uns, dataframe = "columns", auto_unbox = TRUE,
                     null = "null", force = TRUE)

  scx_write_h5ad(
    output    = path.expand(path),
    n_obs     = as.integer(n_cells),
    n_vars    = as.integer(n_genes),
    x_indptr  = as.integer(m@p),
    x_indices = as.integer(m@i),
    x_data    = m@x,
    obs_index = as.character(obs_index),
    var_index = as.character(var_index),
    obs_cols  = obs_cols,
    var_cols  = var_cols,
    uns_json  = as.character(uns_json),
    dtype     = as.character(dtype)
  )
  invisible(path)
}

# ---------------------------------------------------------------------------
# Internal: coerce a DataFrame / data.frame to a plain named list of vectors,
# stripping any S4Vectors-specific column types.
# ---------------------------------------------------------------------------

# Recursively coerce a value into something jsonlite can serialize. Unknown
# S4 / classed objects are stringified via `format()`. data.frames pass through.
.sanitize_for_json <- function(x, depth = 0L) {
  if (depth > 32L) {
    return(tryCatch(format(x), error = function(e) "<unserializable>"))
  }
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) {
    for (j in seq_along(x)) x[[j]] <- .sanitize_atomic(x[[j]])
    return(x)
  }
  cls <- class(x)
  is_plain_list <- is.list(x) && (identical(cls, "list") || identical(cls, "AsIs"))
  if (is_plain_list) {
    return(lapply(x, .sanitize_for_json, depth = depth + 1L))
  }
  .sanitize_atomic(x)
}

.sanitize_atomic <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.atomic(x) && is.null(attr(x, "class"))) return(x)
  # numeric_version, package_version, Date, POSIXct, factor, S4 objects, ...
  tryCatch(as.character(format(x)),
           error = function(e) tryCatch(as.character(x),
                                        error = function(e2) "<unserializable>"))
}

.df_to_named_list <- function(df) {
  if (is.null(df) || ncol(df) == 0L) return(list())
  out <- vector("list", ncol(df))
  names(out) <- colnames(df)
  for (i in seq_len(ncol(df))) {
    col <- df[[i]]
    # Unwrap Rle, List, etc. Rust side handles atomic vectors + factors only.
    if (methods::is(col, "Rle")) {
      col <- as.vector(col)
    } else if (methods::is(col, "List") || is.list(col)) {
      # nested list-columns are not supported in obs/var — coerce to character
      col <- vapply(col, function(x) paste(as.character(x), collapse = ","), character(1))
    }
    out[[i]] <- col
  }
  out
}
