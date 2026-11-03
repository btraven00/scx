#' Convert a single-cell file to another format
#'
#' Streams the input file through the SCX engine and writes the result to
#' `output`. No R objects are constructed; memory usage is bounded by
#' `chunk_size` regardless of dataset size.
#'
#' The input format is detected from the content: AnnData `.h5ad`, AnnData
#' Zarr stores, H5Seurat, 10x HDF5, 10x MatrixMarket directories, BPCells
#' directories. The output format is chosen by the extension of `output`:
#' `.h5ad`, `.h5seurat` (SeuratDisk dgCMatrix layout) or `.zarr` (AnnData
#' Zarr v3 store, a directory).
#'
#' @param input  Path to the input file or directory.
#' @param output Path to the output: `.h5ad`, `.h5seurat` or `.zarr`.
#' @param chunk_size Number of cells per streaming chunk. Default `5000L`.
#' @param dtype Output numeric type: `"f32"`, `"f64"`, `"i32"`, or `"u32"`.
#'   Default `"f32"`.
#' @param assay Seurat assay to convert (H5Seurat inputs only). Default `"RNA"`.
#' @param layer Seurat layer to convert (H5Seurat inputs only). Default `"counts"`.
#' @param where Keep only some cells: a named list mapping `obs` columns to
#'   the values they may take, e.g. `list(cell_type = c("B", "T"), donor =
#'   "d1")`. A cell is kept when every column holds one of its values.
#'   Values are compared as text (`TRUE`/`FALSE` match Python's
#'   `True`/`False`); `NA` never matches. X, layers, raw counts, `obs` and
#'   `obsm` keep the matching cells, and `obsp` graphs those cells on both
#'   axes; genes and `uns` are unchanged. The matrix still streams: h5ad,
#'   Zarr, dgCMatrix H5Seurat and 10x HDF5 inputs read only the matching
#'   rows, others read everything and drop the rest. No match writes a file
#'   with 0 cells; an unknown column is an error. Default `NULL`: every cell.
#'
#' @return `output` path, invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' convert("pbmc3k.h5seurat", "pbmc3k.h5ad")
#' convert("pbmc3k.h5ad", "pbmc3k.zarr")
#' convert("pbmc3k.h5seurat", "pbmc3k_f64.h5ad", dtype = "f64", chunk_size = 2000L)
#' "atlas.h5ad" |> convert("tcells.h5ad", where = list(cell_type = "T cell"))
#' }
convert <- function(input, output,
                    chunk_size = 5000L,
                    dtype      = "f32",
                    assay      = "RNA",
                    layer      = "counts",
                    where      = NULL) {
  w <- .where_pairs(where)
  if (.native_available()) {
    scx_convert(
      input        = path.expand(input),
      output       = path.expand(output),
      chunk_size   = as.integer(chunk_size),
      dtype        = as.character(dtype),
      assay        = as.character(assay),
      layer        = as.character(layer),
      where_cols   = w$cols,
      where_values = w$values
    )
  } else {
    .convert_via_cli(input, output, chunk_size, dtype, assay, layer, w)
  }
  invisible(output)
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.native_available <- function() {
  # Phase B: HDF5 is statically linked in libpicklerick_r.a, so the Rust
  # side has its own isolated HDF5 instance separate from R's rhdf5/hdf5r.
  # The native .Call path is now safe to use.
  is.loaded("wrap__scx_convert", PACKAGE = "picklerick")
}

# `where` as parallel (column, value) vectors, values as the engine compares
# them: logicals as Python spells them, NA dropped (it never matches).
.where_pairs <- function(where) {
  if (is.null(where)) return(list(cols = character(), values = character()))
  if (!is.list(where) || is.null(names(where)) || any(names(where) == "")) {
    stop("`where` must be a named list, e.g. list(cell_type = \"T cell\")",
         call. = FALSE)
  }
  values <- lapply(where, function(v) {
    v <- v[!is.na(v)]
    if (is.logical(v)) ifelse(v, "True", "False") else as.character(v)
  })
  list(cols = rep(names(values), lengths(values)),
       values = unname(unlist(values)) %||% character())
}

.convert_via_cli <- function(input, output, chunk_size, dtype, assay, layer,
                             w = .where_pairs(NULL)) {
  bin <- .scx_binary()
  args <- c(
    "convert",
    input, output,
    "--chunk-size", as.character(chunk_size),
    "--dtype",      dtype,
    "--assay",      assay,
    "--layer",      layer,
    if (length(w$cols)) as.vector(rbind("--where", shQuote(paste0(w$cols, "=", w$values))))
  )
  status <- system2(bin, args = args, stdout = "", stderr = "")
  if (status != 0L) {
    stop(sprintf("scx convert failed (exit %d)", status), call. = FALSE)
  }
}
