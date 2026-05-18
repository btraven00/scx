#' Read a single-cell file into an in-memory R object
#'
#' Reads `.h5ad`, `.h5seurat`, or SCX `.h5` files and returns a
#' `SingleCellExperiment` (default), a `Seurat` object, or the raw named list
#' of pieces (`as = "list"`) for callers that want to assemble something else.
#'
#' Memory: this v1 materializes the X matrix in both Rust and R (~2× peak).
#' For very large datasets prefer `convert()` (streaming, file-to-file).
#'
#' Integer X is cast to `double` to match Seurat conventions and avoid
#' integer-overflow in downstream normalization.
#'
#' Layers whose shape doesn't match `(n_obs, n_vars)` are dropped with a
#' warning. `obsp` is currently only attached to `SingleCellExperiment`
#' (via `colPair`); it is dropped with a warning for Seurat output.
#'
#' @param path       Path to the file.
#' @param as         Target class: `"SingleCellExperiment"` (default),
#'   `"Seurat"`, or `"list"`.
#' @param chunk_size Cells per internal streaming chunk. Default `5000L`.
#' @param lazy       When `TRUE`, leave the X matrix on disk and wrap it in a
#'   `DelayedMatrix` via `HDF5Array::H5SparseMatrix`. The Rust side only
#'   reads metadata (obs / var / obsm / varm / uns), so peak memory is
#'   bounded by metadata size rather than the count matrix. Only supported
#'   for HDF5-backed inputs (`.h5ad`, `.h5`, `.h5seurat`) and only when
#'   `as = "SingleCellExperiment"` (Seurat does not consume DelayedArray
#'   natively). Requires the `HDF5Array` package (Bioconductor).
#'
#' @return A `SingleCellExperiment`, `Seurat`, or named list.
#' @export
#'
#' @examples
#' \dontrun{
#' sce <- read_h5ad("pbmc3k.h5ad")
#' obj <- read_h5ad("pbmc3k.h5ad", as = "Seurat")
#' raw <- read_h5ad("pbmc3k.h5ad", as = "list")  # advanced
#'
#' # Lazy mode — X stays on disk, no full materialization.
#' sce <- read_h5ad("huge.h5ad", lazy = TRUE)
#' counts(sce)            # DelayedMatrix
#' counts(sce)[1:10, ]    # only these rows hit disk
#' }
read_h5ad <- function(path,
                      as         = c("SingleCellExperiment", "Seurat", "list"),
                      chunk_size = 5000L,
                      lazy       = FALSE) {
  as <- match.arg(as)
  path <- path.expand(path)

  if (lazy) {
    if (as == "Seurat") {
      stop("read_h5ad: lazy = TRUE is incompatible with as = 'Seurat' ",
           "(Seurat does not consume DelayedArray).", call. = FALSE)
    }
    if (!requireNamespace("HDF5Array", quietly = TRUE)) {
      stop("read_h5ad(lazy = TRUE) requires the HDF5Array package ",
           "(Bioconductor).", call. = FALSE)
    }
    raw <- scx_read(path, as.integer(chunk_size), read_x = FALSE)
    if (!identical(raw$format, "H5AD")) {
      stop(sprintf(
        "read_h5ad: lazy = TRUE currently supports H5AD only (got %s). ",
        raw$format),
        "Re-run with lazy = FALSE.", call. = FALSE)
    }
    if (as == "list") return(raw)
    return(.as_sce_lazy(raw, path))
  }

  raw <- scx_read(path, as.integer(chunk_size), read_x = TRUE)
  switch(as,
    list                 = raw,
    SingleCellExperiment = .as_sce(raw),
    Seurat               = .as_seurat(raw)
  )
}

# ---------------------------------------------------------------------------
# Shared assembly: scx CSR(cells × genes) is structurally identical to
# CSC(genes × cells) — same indptr/indices/data triplet — so we can build a
# dgCMatrix of (genes × cells) directly. That orientation matches Seurat / SCE
# (features as rows).
# ---------------------------------------------------------------------------

.build_dgc <- function(raw) {
  x <- if (is.integer(raw$x_data)) as.double(raw$x_data) else raw$x_data
  methods::new("dgCMatrix",
    p        = raw$x_indptr,
    i        = raw$x_indices,
    x        = x,
    Dim      = c(as.integer(raw$n_vars), as.integer(raw$n_obs)),
    Dimnames = list(raw$var_index, raw$obs_index)
  )
}

# Build a (rows × cols) dgCMatrix from a triplet returned by the Rust side.
# The triplet is CSR over `rows`, which is CSC over `cols` — so as a dgCMatrix
# we present it as a (cols × rows) matrix. For obsp (square n_obs × n_obs)
# the orientation is symmetric so this is fine; for layers we want
# (genes × cells), same trick as .build_dgc.
.csr_to_dgc_T <- function(triplet, row_names = NULL, col_names = NULL) {
  x <- if (is.integer(triplet$data)) as.double(triplet$data) else triplet$data
  methods::new("dgCMatrix",
    p        = triplet$indptr,
    i        = triplet$indices,
    x        = x,
    Dim      = c(as.integer(triplet$n_cols), as.integer(triplet$n_rows)),
    Dimnames = list(col_names, row_names)
  )
}

.cols_to_df <- function(named_list, row_names) {
  if (length(named_list) == 0L) {
    return(data.frame(row.names = row_names))
  }
  df <- as.data.frame(named_list,
                      stringsAsFactors = FALSE,
                      check.names      = FALSE,
                      optional         = TRUE)
  rownames(df) <- row_names
  df
}

# Embeddings come back as dense matrices with dim = c(rows, cols).
# AnnData convention: obsm rows are cells. For Seurat reductions, we need
# rownames = cell ids.
.embed_with_rownames <- function(m, row_names) {
  if (!is.null(m) && nrow(m) == length(row_names)) rownames(m) <- row_names
  m
}

.parse_uns <- function(uns_json) {
  if (!nzchar(uns_json) || identical(uns_json, "{}")) return(list())
  tryCatch(jsonlite::fromJSON(uns_json, simplifyVector = FALSE),
           error = function(e) {
             warning("read_h5ad: failed to parse uns JSON: ", conditionMessage(e))
             list()
           })
}

# ---------------------------------------------------------------------------
# SingleCellExperiment assembler
# ---------------------------------------------------------------------------

.as_sce <- function(raw) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("read_h5ad(as = 'SingleCellExperiment') requires the ",
         "SingleCellExperiment package.", call. = FALSE)
  }

  m <- .build_dgc(raw)

  obs_df <- .cols_to_df(raw$obs_cols, raw$obs_index)
  var_df <- .cols_to_df(raw$var_cols, raw$var_index)

  # Drop mismatched-shape layers with a warning.
  layer_assays <- list()
  for (nm in names(raw$layers)) {
    tri <- raw$layers[[nm]]
    if (tri$n_rows != raw$n_obs || tri$n_cols != raw$n_vars) {
      warning(sprintf("read_h5ad: layer '%s' has shape %dx%d != (%d, %d); dropped.",
                      nm, tri$n_rows, tri$n_cols, raw$n_obs, raw$n_vars))
      next
    }
    layer_assays[[nm]] <- .csr_to_dgc_T(tri, row_names = raw$obs_index,
                                            col_names = raw$var_index)
  }

  # obsm: AnnData rows = cells; reducedDims wants the same orientation.
  reduced <- lapply(names(raw$obsm), function(nm) {
    .embed_with_rownames(raw$obsm[[nm]], raw$obs_index)
  })
  names(reduced) <- names(raw$obsm)

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays      = c(list(counts = m), layer_assays),
    colData     = S4Vectors::DataFrame(obs_df),
    rowData     = S4Vectors::DataFrame(var_df),
    reducedDims = reduced,
    metadata    = .parse_uns(raw$uns_json)
  )

  # obsp → colPair (n_obs × n_obs sparse matrices).
  for (nm in names(raw$obsp)) {
    tri <- raw$obsp[[nm]]
    if (tri$n_rows != raw$n_obs || tri$n_cols != raw$n_obs) {
      warning(sprintf("read_h5ad: obsp '%s' has shape %dx%d != (%d, %d); dropped.",
                      nm, tri$n_rows, tri$n_cols, raw$n_obs, raw$n_obs))
      next
    }
    g <- .csr_to_dgc_T(tri, row_names = raw$obs_index, col_names = raw$obs_index)
    SingleCellExperiment::colPair(sce, nm) <- g
  }

  # varm → rowData attribute slot (SCE has no first-class varm; stash in metadata).
  if (length(raw$varm)) {
    S4Vectors::metadata(sce)$varm <- lapply(raw$varm, function(m) {
      if (nrow(m) == length(raw$var_index)) rownames(m) <- raw$var_index
      m
    })
  }

  sce
}

# ---------------------------------------------------------------------------
# Seurat assembler
# ---------------------------------------------------------------------------

.as_seurat <- function(raw) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("read_h5ad(as = 'Seurat') requires the Seurat package ",
         "(in Suggests; install separately).", call. = FALSE)
  }

  m <- .build_dgc(raw)
  meta <- .cols_to_df(raw$obs_cols, raw$obs_index)
  obj <- Seurat::CreateSeuratObject(counts = m, meta.data = meta, assay = "RNA")

  # rowData → feature metadata on the RNA assay. AddMetaData has methods for
  # both Assay (v3/v4) and Assay5 (v5), so it's the portable entry point.
  if (length(raw$var_cols)) {
    feat <- .cols_to_df(raw$var_cols, raw$var_index)
    tryCatch(
      obj[["RNA"]] <- Seurat::AddMetaData(obj[["RNA"]], metadata = feat),
      error = function(e) {
        warning("read_h5ad: could not attach feature metadata: ",
                conditionMessage(e))
      }
    )
  }

  # Layers → additional assays. Mismatched shapes dropped with warning.
  for (nm in names(raw$layers)) {
    tri <- raw$layers[[nm]]
    if (tri$n_rows != raw$n_obs || tri$n_cols != raw$n_vars) {
      warning(sprintf("read_h5ad: layer '%s' has shape %dx%d != (%d, %d); dropped.",
                      nm, tri$n_rows, tri$n_cols, raw$n_obs, raw$n_vars))
      next
    }
    a <- .csr_to_dgc_T(tri, row_names = raw$obs_index, col_names = raw$var_index)
    obj[[nm]] <- Seurat::CreateAssayObject(counts = a)
  }

  # obsm → reductions. AnnData "X_pca" → Seurat "pca", etc.
  for (nm in names(raw$obsm)) {
    emb <- .embed_with_rownames(raw$obsm[[nm]], raw$obs_index)
    key <- sub("^X_", "", nm)
    obj[[key]] <- Seurat::CreateDimReducObject(
      embeddings = emb,
      key        = paste0(key, "_"),
      assay      = "RNA"
    )
  }

  # obsp: defer (drop with warning). Seurat::as.Graph could be wired here later.
  for (nm in names(raw$obsp)) {
    warning(sprintf("read_h5ad: obsp '%s' is not yet attached to Seurat output; dropped.",
                    nm))
  }

  # uns → @misc.
  uns <- .parse_uns(raw$uns_json)
  if (length(uns)) obj@misc <- uns

  obj
}

# ---------------------------------------------------------------------------
# Lazy SCE assembler: X stays on disk, wrapped as a DelayedMatrix via
# HDF5Array::H5SparseMatrix. Only h5ad layout is supported (data/indices/
# indptr at /X). obsp / layers are skipped in lazy mode for now — they could
# be wrapped the same way but each one is its own HDF5 group lookup.
# ---------------------------------------------------------------------------

.as_sce_lazy <- function(raw, path) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("read_h5ad(lazy = TRUE) requires the SingleCellExperiment package.",
         call. = FALSE)
  }

  # H5SparseMatrix infers CSR vs CSC from the AnnData encoding-type attribute
  # on /X. For h5ad: encoding-type = "csr_matrix" → shape (n_obs, n_vars).
  # We want (n_vars × n_obs) for SCE — t() is a lazy op on DelayedArray.
  x_lazy <- HDF5Array::H5SparseMatrix(filepath = path, group = "/X")
  if (nrow(x_lazy) == raw$n_obs && ncol(x_lazy) == raw$n_vars) {
    x_lazy <- t(x_lazy)   # → (n_vars × n_obs)
  }
  dimnames(x_lazy) <- list(raw$var_index, raw$obs_index)

  obs_df <- .cols_to_df(raw$obs_cols, raw$obs_index)
  var_df <- .cols_to_df(raw$var_cols, raw$var_index)

  reduced <- lapply(names(raw$obsm), function(nm) {
    .embed_with_rownames(raw$obsm[[nm]], raw$obs_index)
  })
  names(reduced) <- names(raw$obsm)

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays      = list(counts = x_lazy),
    colData     = S4Vectors::DataFrame(obs_df),
    rowData     = S4Vectors::DataFrame(var_df),
    reducedDims = reduced,
    metadata    = .parse_uns(raw$uns_json)
  )

  if (length(raw$varm)) {
    S4Vectors::metadata(sce)$varm <- lapply(raw$varm, function(m) {
      if (nrow(m) == length(raw$var_index)) rownames(m) <- raw$var_index
      m
    })
  }

  sce
}
