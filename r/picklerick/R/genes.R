#' Read selected genes of a dataset's count matrix
#'
#' Reads only the genes in `genes` from `path`, for per-gene work (model
#' fitting, tests) on a gene subset.
#'
#' Gene-major `.h5ad` / Zarr stores (written with
#' `convert(..., gene_major = TRUE)`) read just those genes. A Seurat,
#' SingleCellExperiment or SpatialExperiment `.rds` is inflated once to
#' index it, and the selected genes are then read in parallel; memory is
#' about 12 bytes per kept nonzero plus 4 bytes per nonzero of the whole
#' matrix. Cell-major `.h5ad` and other formats are refused; convert them
#' with `gene_major = TRUE` first. To read many blocks of one file, use
#' [gene_blocks()], which opens it once.
#'
#' @param path Path to the dataset.
#' @param genes Genes to read: 1-based integer positions or gene names, in
#'   any order. `NULL` reads all genes.
#' @param chunk_size Genes per internal read.
#' @return A `dgCMatrix`, cells x genes: one column per element of `genes`,
#'   in that order, so each gene's counts are contiguous. Use `t()` for genes
#'   x cells.
#' @seealso [gene_blocks()]
#' @export
read_genes <- function(path, genes = NULL, chunk_size = 1000L) {
  r <- GeneReader$open(normalizePath(path, mustWork = TRUE), as.integer(chunk_size))
  .read_gene_set(r, genes, basename(path), r$var_index(), r$obs_index())
}

#' Iterate over a dataset's genes in blocks
#'
#' Opens `path` once and reads its genes a block at a time, as
#' [read_genes()] reads them. For an `.rds`, opening inflates the file once;
#' each block is then read in parallel from the index that pass built, so a
#' block costs a fraction of a [read_genes()] call. Memory holds one block,
#' plus 4 bytes per nonzero of the whole matrix while it is read.
#'
#' Forked workers (e.g. `parallel::mclapply`) can share one iterator and
#' read different blocks with `$read(k)`.
#'
#' @inheritParams read_genes
#' @param block_size Genes per block.
#' @return A `picklerick_gene_blocks` list:
#'   \describe{
#'     \item{`n`}{Number of blocks.}
#'     \item{`genes`}{List of each block's gene positions.}
#'     \item{`read(k)`}{Block `k` as a cells x genes `dgCMatrix`.}
#'     \item{`next_block()`}{The next block, or `NULL` after the last.}
#'   }
#' @examples
#' \dontrun{
#' it <- gene_blocks("spe.rds", block_size = 2000)
#' while (!is.null(m <- it$next_block())) fit(m)
#' fits <- parallel::mclapply(seq_len(it$n), function(k) fit(it$read(k)))
#' }
#' @seealso [read_genes()]
#' @export
gene_blocks <- function(path, block_size = 2000L, genes = NULL, chunk_size = 1000L) {
  r <- GeneReader$open(normalizePath(path, mustWork = TRUE), as.integer(chunk_size))
  # Names once: 443k cell names cost ~0.2 s to hand to R per block.
  var <- r$var_index()
  obs <- r$obs_index()
  if (is.null(genes)) genes <- seq_along(var)
  genes <- .gene_positions(var, genes, basename(path))
  block_size <- as.integer(block_size)
  if (is.na(block_size) || block_size < 1L) stop("block_size must be a positive integer")
  blocks <- split(genes, ceiling(seq_along(genes) / block_size))
  names(blocks) <- NULL
  k <- 0L
  read <- function(k) {
    if (k < 1L || k > length(blocks)) stop("no block ", k, " (have ", length(blocks), ")")
    .read_gene_set(r, blocks[[k]], basename(path), var, obs)
  }
  structure(list(
    n = length(blocks),
    genes = blocks,
    read = read,
    next_block = function() {
      if (k >= length(blocks)) return(NULL)
      k <<- k + 1L
      read(k)
    }
  ), class = "picklerick_gene_blocks")
}

#' @export
print.picklerick_gene_blocks <- function(x, ...) {
  cat(sprintf("<gene_blocks> %d genes in %d blocks\n", length(unlist(x$genes)), x$n))
  invisible(x)
}

# `genes` (positions or names) as 1-based positions, checked.
.gene_positions <- function(var, genes, what) {
  if (is.character(genes)) {
    pos <- match(genes, var)
    if (anyNA(pos)) {
      stop("genes not in ", what, ": ",
           paste(utils::head(genes[is.na(pos)], 5L), collapse = ", "))
    }
    return(pos)
  }
  genes <- as.integer(genes)
  if (anyNA(genes) || any(genes < 1L)) stop("genes must be positive positions or names")
  genes
}

# The genes `genes` (NULL = all) of an open GeneReader with gene names
# `var` and cell names `obs`, cells x genes.
.read_gene_set <- function(r, genes, what, var, obs) {
  if (is.null(genes)) genes <- seq_along(var)
  genes <- .gene_positions(var, genes, what)
  want <- sort(unique(genes))
  rng <- .index_to_ranges(want)
  tri <- r$read(as.integer(rng$starts), as.integer(rng$ends))
  m <- .csr_to_dgc_T(tri, row_names = var[want], col_names = obs)
  if (!identical(genes, want)) m <- m[, match(genes, want), drop = FALSE]
  m
}
