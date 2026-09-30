# Unofficial, experimental Seurat constructor (read_h5ad(constructor = "fast")).
#
# SeuratObject::CreateSeuratObject(counts = m) costs ~2.2x the matrix in
# transient allocations on top of the matrix itself (Rprofmem, 10x 10k cells,
# SeuratObject 5.4.0):
#   - LayerData<-.Assay5 stores m[features, cells]: a full copy, even though
#     the subset is the identity;
#   - .CalcN reads the layer back through LayerData.Assay5, another full
#     subset copy, then builds `m > 0` to count features.
# Here the Assay5 is assembled from its slots with the layer set to m itself,
# nCount/nFeature come from the dgCMatrix structure, and SeuratObject's own
# CalcN is switched off. Result: +0 MB over the matrix instead of +535 MB, and
# an object identical to CreateSeuratObject()'s (tests/testthat/test-seurat-fast.R).
#
# This reaches into SeuratObject internals: the Assay5 slot layout, LogMap,
# and the Seurat.object.assay.calcn option. It MAY DRIFT with a SeuratObject
# release, so it only runs on the major version it was checked against, and
# only on input needing none of CreateSeuratObject()'s sanitising. Anything
# else returns NULL, and the caller falls back to CreateSeuratObject().

.SEURAT_FAST_MAJOR <- 5L   # SeuratObject major version this was checked against

.seurat_fast <- function(m, meta, assay = "RNA") {
  why <- .seurat_fast_blocker(m, meta, assay)
  if (!is.null(why)) {
    message("read_h5ad: constructor = 'fast' falling back to ",
            "CreateSeuratObject(): ", why)
    return(NULL)
  }

  # Which cells and features the counts layer covers: all of them, in order.
  cells <- SeuratObject::LogMap(colnames(m))
  cells[["counts"]] <- seq_len(ncol(m))
  features <- SeuratObject::LogMap(rownames(m))
  features[["counts"]] <- seq_len(nrow(m))
  a5 <- methods::new("Assay5",
    layers = list(counts = m), cells = cells, features = features,
    default = 1L, assay.orig = character(), key = paste0(tolower(assay), "_"),
    meta.data = data.frame(row.names = rownames(m)), misc = list())
  methods::validObject(a5)

  # What CalcN would have computed, without reading the layer back. Stored
  # zeros are not features, as in CalcN's `m > 0`.
  n_feature <- diff(m@p)
  if (any(z <- m@x == 0))
    n_feature <- n_feature - tabulate(rep.int(seq_len(ncol(m)), diff(m@p))[z], ncol(m))
  qc <- data.frame(Matrix::colSums(m), n_feature, row.names = colnames(m))
  names(qc) <- paste0(c("nCount_", "nFeature_"), assay)
  meta <- if (ncol(meta)) cbind(qc, meta) else qc

  op <- options(Seurat.object.assay.calcn = FALSE)
  on.exit(options(op), add = TRUE)
  SeuratObject::CreateSeuratObject(a5, assay = assay, meta.data = meta)
}

# NULL when the fast path reproduces CreateSeuratObject() exactly, otherwise
# the reason it would not.
.seurat_fast_blocker <- function(m, meta, assay) {
  if (!requireNamespace("SeuratObject", quietly = TRUE))
    return("SeuratObject not installed")
  v <- utils::packageVersion("SeuratObject")
  if (v$major != .SEURAT_FAST_MAJOR)
    return(sprintf("untested SeuratObject %s (checked against %d.x)", v, .SEURAT_FAST_MAJOR))
  if (!methods::is(m, "dgCMatrix")) return("counts is not a dgCMatrix")
  f <- rownames(m); cl <- colnames(m)
  if (is.null(f) || is.null(cl)) return("missing feature or cell names")
  # CreateSeuratObject rewrites these (with a warning) or rejects them.
  if (any(grepl("_", f, fixed = TRUE))) return("feature names contain '_'")
  if (any(!nzchar(f)) || any(!nzchar(cl))) return("empty feature or cell names")
  if (anyDuplicated(f) || anyDuplicated(cl)) return("duplicate feature or cell names")
  # These would collide with the columns CreateSeuratObject adds itself.
  clash <- intersect(names(meta), c("orig.ident", paste0(c("nCount_", "nFeature_"), assay)))
  if (length(clash)) return(paste("obs already has", paste(clash, collapse = ", ")))
  NULL
}
