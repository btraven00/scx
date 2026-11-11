#' Read selected genes of a dataset's count matrix
#'
#' Reads only the genes in `genes` from `path`, for per-gene work (model
#' fitting, tests) on a gene subset, e.g. one block per worker.
#'
#' Gene-major `.h5ad` / Zarr stores (written with
#' `convert(..., gene_major = TRUE)`) read just those genes. A Seurat,
#' SingleCellExperiment or SpatialExperiment `.rds` is decompressed once per
#' call, keeping only the selected genes' nonzeros: memory is about 12 bytes
#' per kept nonzero, so give each worker its own block of genes. Cell-major
#' `.h5ad` and other formats are refused; convert them with
#' `gene_major = TRUE` first.
#'
#' @param path Path to the dataset.
#' @param genes Genes to read: 1-based integer positions or gene names, in
#'   any order. `NULL` reads all genes.
#' @param chunk_size Genes per internal read.
#' @return A `dgCMatrix`, cells x genes: one column per element of `genes`,
#'   in that order, so each gene's counts are contiguous. Use `t()` for genes
#'   x cells.
#' @examples
#' \dontrun{
#' blocks <- split(seq_len(n_genes), ceiling(seq_len(n_genes) / 2000))
#' fits <- parallel::mclapply(blocks, function(g) fit(read_genes("spe.rds", g)))
#' }
#' @export
read_genes <- function(path, genes = NULL, chunk_size = 1000L) {
  path <- normalizePath(path, mustWork = TRUE)
  if (is.character(genes)) {
    var <- scx_seed_open(path, "")$var_index
    pos <- match(genes, var)
    if (anyNA(pos)) {
      stop("genes not in ", basename(path), ": ",
           paste(utils::head(genes[is.na(pos)], 5L), collapse = ", "))
    }
    genes <- pos
  }
  if (!is.null(genes)) {
    genes <- as.integer(genes)
    if (anyNA(genes) || any(genes < 1L)) stop("genes must be positive positions or names")
  }
  want <- if (is.null(genes)) NULL else sort(unique(genes))
  # No ranges: every gene.
  r <- if (is.null(want)) list(starts = integer(0), ends = integer(0)) else .index_to_ranges(want)
  tri <- scx_read_cols(path, as.integer(chunk_size), as.integer(r$starts), as.integer(r$ends))
  m <- .csr_to_dgc_T(tri, row_names = tri$var_index, col_names = tri$obs_index)
  if (!is.null(genes) && !identical(genes, want)) m <- m[, match(genes, want), drop = FALSE]
  m
}
