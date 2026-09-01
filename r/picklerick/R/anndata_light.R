#' A light AnnData handle
#'
#' `read_h5ad(as = "AnnData")` returns one of these instead of a
#' `SingleCellExperiment`. It holds what the Rust reader produced and builds R
#' objects only when a component is actually touched.
#'
#' The reason this exists is measured, not aesthetic. Building a
#' `SingleCellExperiment` costs a constant ~3.3 seconds on this machine
#' regardless of dataset size — 0.05s vs 3.19s for a thousand cells, 3.72s vs
#' 7.09s for a hundred thousand. That cost is S4 method-table and Bioconductor
#' namespace warm-up, not work proportional to the data, so it cannot be
#' optimised away inside the conversion; it can only be deferred until someone
#' asks for an `SingleCellExperiment`.
#'
#' Fields are reached with `$`, and each is built on first access and cached:
#' `$X`, `$obs`, `$var`, `$obsm`, `$layers`, `$obs_names`, `$var_names`,
#' `$n_obs`, `$n_vars`. Convert with [as_sce()] or [as_seurat()].
#'
#' @name AnnDataLight
#' @examples
#' \dontrun{
#' a <- read_h5ad("pbmc.h5ad", as = "AnnData")
#' dim(a)          # cheap: shape comes from the reader, nothing is built
#' a$obs$cell_type # builds and caches obs only
#' sce <- as_sce(a) # pays the S4 cost, once, when it is actually wanted
#' }
NULL

# An environment rather than a list so that lazily built components can be
# cached in place; a list would be copied on every assignment.
.new_anndata_light <- function(raw, path = NULL) {
  self <- new.env(parent = emptyenv())
  self$.raw <- raw
  self$.path <- path
  self$.cache <- new.env(parent = emptyenv())
  class(self) <- "picklerick_anndata"
  self
}

.memo <- function(x, key, build) {
  cache <- get(".cache", envir = x)
  if (!exists(key, envir = cache, inherits = FALSE)) {
    assign(key, build(), envir = cache)
  }
  get(key, envir = cache, inherits = FALSE)
}

#' @export
`$.picklerick_anndata` <- function(x, name) {
  raw <- get(".raw", envir = x)
  switch(name,
    n_obs     = raw$n_obs,
    n_vars    = raw$n_vars,
    obs_names = raw$obs_index,
    var_names = raw$var_index,
    # Built on demand; each is the expensive part for exactly one use case.
    X      = .memo(x, "X", function() .build_dgc(raw)),
    obs    = .memo(x, "obs", function() .cols_to_df(raw$obs_cols, raw$obs_index)),
    var    = .memo(x, "var", function() .cols_to_df(raw$var_cols, raw$var_index)),
    obsm   = .memo(x, "obsm", function() {
      out <- lapply(names(raw$obsm), function(nm)
        .embed_with_rownames(raw$obsm[[nm]], raw$obs_index))
      names(out) <- names(raw$obsm)
      out
    }),
    layers = .memo(x, "layers", function() {
      out <- lapply(names(raw$layers), function(nm)
        .csr_to_dgc_T(raw$layers[[nm]], row_names = raw$obs_index,
                      col_names = raw$var_index))
      names(out) <- names(raw$layers)
      out
    }),
    # Anything else, including the internals, falls through unchanged.
    get(name, envir = x, inherits = FALSE)
  )
}

#' @export
dim.picklerick_anndata <- function(x) {
  raw <- get(".raw", envir = x)
  c(raw$n_vars, raw$n_obs)   # genes x cells, matching SingleCellExperiment
}

#' @export
dimnames.picklerick_anndata <- function(x) {
  raw <- get(".raw", envir = x)
  list(raw$var_index, raw$obs_index)
}

#' @export
print.picklerick_anndata <- function(x, ...) {
  raw <- get(".raw", envir = x)
  built <- ls(get(".cache", envir = x))
  cat(sprintf("<AnnData> %d genes x %d cells\n", raw$n_vars, raw$n_obs))
  cat(sprintf("  obs: %d columns   var: %d columns\n",
              length(raw$obs_cols), length(raw$var_cols)))
  if (length(raw$obsm))   cat("  obsm:  ", paste(names(raw$obsm), collapse = ", "), "\n")
  if (length(raw$layers)) cat("  layers:", paste(names(raw$layers), collapse = ", "), "\n")
  cat("  built:", if (length(built)) paste(built, collapse = ", ") else "nothing yet", "\n")
  invisible(x)
}

#' Convert a light AnnData handle to a Bioconductor or Seurat object
#'
#' The conversion is where the S4 cost is paid, so it is a separate call.
#'
#' @param x An object from [read_h5ad()].
#' @param parse_uns Parse `uns` into `metadata()` (see [read_h5ad()]).
#' @param ... Unused.
#' @return A `SingleCellExperiment` or a `Seurat` object.
#' @export
as_sce <- function(x, ...) UseMethod("as_sce")

#' @rdname as_sce
#' @export
as_sce.picklerick_anndata <- function(x, parse_uns = FALSE, ...) {
  .as_sce(get(".raw", envir = x), path = get(".path", envir = x),
          parse_uns = parse_uns)
}

#' @rdname as_sce
#' @export
as_seurat <- function(x, ...) UseMethod("as_seurat")

#' @rdname as_sce
#' @export
as_seurat.picklerick_anndata <- function(x, parse_uns = FALSE, ...) {
  .as_seurat(get(".raw", envir = x), path = get(".path", envir = x),
             parse_uns = parse_uns)
}
