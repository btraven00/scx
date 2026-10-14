# A DelayedArray seed over any matrix scx reads: X (or a layer) of an h5ad,
# AnnData Zarr store, H5Seurat, 10x HDF5, MatrixMarket or BPCells directory,
# presented as (features x cells) like every other matrix in this package.
#
# Each block request becomes one `scx_read_rows()` call: the selected cells,
# coalesced into row ranges, read through scx-core's row-range path, which on
# h5ad and Zarr touches only those rows (indptr slice + data/indices slices).
# Features are then subset in R.
#
# For plain .h5ad, HDF5Array::H5SparseMatrix is the conventional seed and
# reads the same layout the same way; this one exists for Zarr v3 and the
# other formats, which have no R seed.
#
# ponytail: the file is reopened per block, so the row pointer (8 bytes per
# cell) is reread each time; cache it in the seed if blockApply over a
# multi-million-cell store shows it in a profile.

#' DelayedArray seed backed by the scx engine
#'
#' `PicklerickMatrix()` returns a `DelayedMatrix` (features x cells) whose
#' blocks are read on demand through scx. Column (cell) selections read only
#' the selected cells from disk on h5ad and Zarr stores; other formats read
#' the whole matrix per block and keep the selection.
#'
#' @param path Path to an `.h5ad` file, `.zarr` directory, or any other
#'   format scx reads.
#' @param layer Name of a layer to expose instead of `X` (h5ad and Zarr).
#'   `NULL` (default) exposes `X`.
#' @param chunk_size Cells per internal read chunk; also the seed's
#'   `chunkdim()` width, which block processing follows.
#' @return `PicklerickMatrix()` returns a `DelayedMatrix`;
#'   `PicklerickMatrixSeed()` the underlying seed.
#' @examples
#' \dontrun{
#' m <- PicklerickMatrix("pbmc.zarr")
#' dim(m)
#' m[1:10, 1:5]                       # reads 5 cells
#' DelayedArray::colSums(m)           # block-wise, chunk_size cells at a time
#' }
#' @export
PicklerickMatrix <- function(path, layer = NULL, chunk_size = 5000L) {
  DelayedArray::DelayedArray(PicklerickMatrixSeed(path, layer, chunk_size))
}

#' @rdname PicklerickMatrix
#' @export
setClass("PicklerickMatrixSeed",
  contains = "Array",
  slots = c(
    path       = "character",
    layer      = "character",
    dim        = "integer",
    dimnames   = "list",
    chunk_size = "integer"
  )
)

#' @rdname PicklerickMatrix
#' @export
PicklerickMatrixSeed <- function(path, layer = NULL, chunk_size = 5000L) {
  path <- path.expand(path)
  layer <- if (is.null(layer)) "" else as.character(layer)
  info <- scx_seed_open(path, layer)
  methods::new("PicklerickMatrixSeed",
    path       = path,
    layer      = layer,
    dim        = c(info$n_vars, info$n_obs),
    dimnames   = list(info$var_index, info$obs_index),
    chunk_size = max(1L, as.integer(chunk_size))
  )
}

#' @rdname PicklerickMatrix
#' @param x A `PicklerickMatrixSeed`.
#' @export
setMethod("dim", "PicklerickMatrixSeed", function(x) x@dim)

#' @rdname PicklerickMatrix
#' @export
setMethod("dimnames", "PicklerickMatrixSeed", function(x) x@dimnames)

#' @rdname PicklerickMatrix
#' @param object A `PicklerickMatrixSeed`.
#' @export
setMethod("show", "PicklerickMatrixSeed", function(object) {
  cat(sprintf("<PicklerickMatrixSeed> %d x %d (features x cells), chunk_size %d\n",
              object@dim[[1L]], object@dim[[2L]], object@chunk_size))
  cat("  path:  ", object@path, "\n", sep = "")
  cat("  matrix:", if (nzchar(object@layer)) paste0("layers/", object@layer) else "X", "\n")
})

#' @rdname PicklerickMatrix
#' @export
setMethod("is_sparse", "PicklerickMatrixSeed", function(x) TRUE)

#' @rdname PicklerickMatrix
#' @export
setMethod("chunkdim", "PicklerickMatrixSeed", function(x) {
  c(x@dim[[1L]], min(x@chunk_size, x@dim[[2L]]))
})

# Sorted unique 1-based cell indices -> 0-based half-open ranges, as the two
# integer vectors scx_read_rows() takes.
.index_to_ranges <- function(j) {
  if (length(j) == 0L) return(list(starts = integer(0), ends = integer(0)))
  breaks <- which(diff(j) != 1L)
  starts <- c(j[1L], j[breaks + 1L])
  ends   <- c(j[breaks], j[length(j)])
  list(starts = starts - 1L, ends = ends)
}

# The cells `j` (sorted unique, or NULL for all) as a dgCMatrix
# (features x length(j)), with dimnames.
.load_cells <- function(x, j) {
  r <- if (is.null(j)) list(starts = 0L, ends = x@dim[[2L]]) else .index_to_ranges(j)
  tri <- scx_read_rows(x@path, x@chunk_size, x@layer, r$starts, r$ends)
  .csr_to_dgc_T(tri,
    row_names = if (is.null(j)) x@dimnames[[2L]] else x@dimnames[[2L]][j],
    col_names = x@dimnames[[1L]])
}

# Same shape as HDF5Array's CSC seed: read the sorted unique cells once, then
# let SparseArray apply the original (possibly unsorted, duplicated) index.
.extract_sparse_from_seed <- function(x, index) {
  j <- index[[2L]]
  sort_j <- FALSE
  if (!is.null(j)) {
    j <- as.integer(j)
    sort_j <- is.unsorted(j, strictly = TRUE)
    if (sort_j) {
      j0 <- j
      j <- unique(sort(j))
    }
  }
  svt <- methods::as(.load_cells(x, j), "SVT_SparseMatrix")
  index2 <- list(index[[1L]], if (sort_j) match(j0, j) else NULL)
  SparseArray::extract_sparse_array(svt, index2)
}

#' @rdname PicklerickMatrix
#' @param index A list of two subscripts (features, cells), each an integer
#'   vector or `NULL` for all, as `S4Arrays::extract_array()` passes it.
#' @export
setMethod("extract_sparse_array", "PicklerickMatrixSeed", .extract_sparse_from_seed)

#' @rdname PicklerickMatrix
#' @export
setMethod("extract_array", "PicklerickMatrixSeed", function(x, index) {
  as.matrix(.extract_sparse_from_seed(x, index))
})
