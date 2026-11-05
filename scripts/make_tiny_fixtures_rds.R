# Seurat .rds versions of the tiny dataset (tests/fixtures/tiny/expected.json):
# the counts plus every obs column, as a v3/v4 `Assay`, a v5 `Assay5`, and an
# `Assay5` split into one layer per donor. Run in the pixi `test` env:
#   pixi run -e test make-tiny-fixtures-rds
suppressMessages({
  library(SeuratObject)
  library(Matrix)
  library(jsonlite)
})

dir <- "tests/fixtures/tiny"
e <- fromJSON(file.path(dir, "expected.json"), simplifyVector = TRUE)

counts <- as(t(e$counts), "CsparseMatrix") # genes x cells
counts <- as(counts * 1.0, "dgCMatrix")
dimnames(counts) <- list(e$var_names, e$obs_names)

meta <- data.frame(row.names = e$obs_names)
for (name in names(e$obs)) {
  col <- e$obs[[name]]
  meta[[name]] <- switch(col$type,
    categorical = factor(col$values, levels = col$levels),
    int = as.integer(col$values),
    bool = as.logical(col$values),
    float = as.double(col$values),
    string = as.character(col$values),
    stop("unknown obs type ", col$type)
  )
}

# An R ordered factor, not in expected.json: the reader must keep `ordered`.
meta$cell_type_ordered <- factor(meta$cell_type, levels = levels(meta$cell_type), ordered = TRUE)

# Reductions and a graph from expected.json: pca (X_pca + PCs loadings), a
# umap without loadings, and obsp$connectivities as the RNA_snn Graph. The
# graph is asymmetric, so a reader that forgets to transpose R's CSC fails.
add_slots <- function(o, emb, loadings, graph, cells, genes) {
  rownames(emb) <- cells; colnames(emb) <- paste0("PC_", seq_len(ncol(emb)))
  rownames(loadings) <- genes; colnames(loadings) <- colnames(emb)
  o[["pca"]] <- CreateDimReducObject(embeddings = emb, loadings = loadings, key = "PC_", assay = "RNA")
  umap <- emb * 2; colnames(umap) <- paste0("UMAP_", seq_len(ncol(umap)))
  o[["umap"]] <- CreateDimReducObject(embeddings = umap, key = "UMAP_", assay = "RNA")
  dimnames(graph) <- list(cells, cells)
  g <- as.Graph(as(graph, "CsparseMatrix")); DefaultAssay(g) <- "RNA"
  o[["RNA_snn"]] <- g
  o
}

# A `data` layer (expected.json's normalised X) and a feature-metadata
# column (gene_symbol): on `Assay` they are @data and @meta.features, on
# `Assay5` a layer and @meta.data.
data_m <- as(t(e$X), "CsparseMatrix"); data_m <- as(data_m * 1.0, "dgCMatrix")
dimnames(data_m) <- dimnames(counts)
add_data <- function(o, data, symbols) {
  LayerData(o, "data") <- data
  o[["RNA"]][["gene_symbol"]] <- symbols
  o
}

make <- function(assay_version) {
  options(Seurat.object.assay.version = assay_version)
  o <- CreateSeuratObject(counts = counts, meta.data = meta, min.cells = 0, min.features = 0)
  o <- add_data(o, data_m, setNames(e$var$gene_symbol$values, e$var_names))
  add_slots(o, e$obsm$X_pca, e$varm$PCs, e$obsp$connectivities, e$obs_names, e$var_names)
}

v4 <- make("v3")
stopifnot(class(v4[["RNA"]]) == "Assay")
saveRDS(v4, file.path(dir, "tiny_v4.rds"))

v5 <- make("v5")
stopifnot(class(v5[["RNA"]]) == "Assay5")
saveRDS(v5, file.path(dir, "tiny_v5.rds"))

# A dense counts layer, which the reader refuses (dgCMatrix layers only).
dense5 <- v5
LayerData(dense5, "counts") <- as.matrix(LayerData(dense5, "counts"))
saveRDS(dense5, file.path(dir, "tiny_v5_dense.rds"))

split5 <- split(v5, f = v5$donor)
stopifnot(length(Layers(split5, search = "counts")) == 2)
saveRDS(split5, file.path(dir, "tiny_v5_split.rds"))

# big_*: larger than the parser's LAZY_MIN (4096 elements), so the reader takes
# the path real files take: @i, @x and the meta.data columns are left in the
# stream and fetched later. Values follow formulas the Rust test recomputes
# (crates/scx-core/tests/rds.rs), for cell k and gene g (0-based):
#   X[k, g] = (7k + 3g) %% 5, zeros not stored
#   grp = c("a","b","c")[k %% 3 + 1], NA when k %% 50 == 0
#   score = k / 4, NA when k %% 40 == 0;  n = k, NA when k %% 30 == 0
#   flag = k %% 2 == 0, NA when k %% 60 == 0;  label = "s<k>", NA when k %% 70 == 0
#   batch = k %% 2 (the v5 object is split by it)
# plus a complex and a list column, which the reader skips, and
#   pca: emb[k, c] = k + c / 10 (c = 0, 1), loadings[g, c] = g - c
#   RNA_snn: cell k -> (k + 1) %% nc weight k %% 7 + 1, and -> (k + 2) %% nc weight 0.5
#   data layer = 2 * X; feature column gene_symbol = "id<g>"
# (umap = 2 * pca, no loadings), all past LAZY_MIN.
nc <- 5000; ng <- 6
k <- 0:(nc - 1)
vals <- outer(k, 0:(ng - 1), function(k, g) (7 * k + 3 * g) %% 5) # cells x genes
big_counts <- as(Matrix(t(vals) * 1.0, sparse = TRUE), "dgCMatrix")
dimnames(big_counts) <- list(paste0("g", 0:(ng - 1)), paste0("c", k))
na_at <- function(x, m) { x[k %% m == 0] <- NA; x }
big_meta <- data.frame(
  grp = na_at(factor(c("a", "b", "c")[k %% 3 + 1], levels = c("a", "b", "c")), 50),
  score = na_at(k / 4, 40),
  n = na_at(as.integer(k), 30),
  flag = na_at(k %% 2 == 0, 60),
  label = na_at(paste0("s", k), 70),
  batch = factor(k %% 2),
  row.names = paste0("c", k)
)
make_big <- function(assay_version) {
  options(Seurat.object.assay.version = assay_version)
  o <- CreateSeuratObject(counts = big_counts, meta.data = big_meta, min.cells = 0, min.features = 0)
  o@meta.data$cplx <- complex(real = k, imaginary = 1)
  o@meta.data$lst <- I(as.list(k))
  o <- add_data(o, big_counts * 2, setNames(paste0("id", 0:(ng - 1)), rownames(big_counts)))
  emb <- outer(k, 0:1, function(k, c) k + c / 10)
  loadings <- outer(0:(ng - 1), 0:1, function(g, c) g - c)
  graph <- sparseMatrix(i = c(k, k) + 1, j = c((k + 1) %% nc, (k + 2) %% nc) + 1,
                        x = c(k %% 7 + 1, rep(0.5, nc)), dims = c(nc, nc))
  add_slots(o, emb, loadings, graph, paste0("c", k), paste0("g", 0:(ng - 1)))
}
big4 <- make_big("v3")
saveRDS(big4, file.path(dir, "big_v4.rds"))
big5 <- make_big("v5")
big5 <- split(big5, f = big5$batch)
stopifnot(length(Layers(big5, search = "counts")) == 2)
saveRDS(big5, file.path(dir, "big_v5_split.rds"))
note <- "saved next to the object"
save(note, big4, file = file.path(dir, "big_v4.rda"))

# One assay, not called RNA (as SeuratData's spatial objects): with no
# --assay the reader must take the object's active assay.
spatial <- CreateSeuratObject(counts = counts, meta.data = meta, assay = "Spatial",
                              min.cells = 0, min.features = 0)
stopifnot(identical(DefaultAssay(spatial), "Spatial"), identical(Assays(spatial), "Spatial"))
saveRDS(spatial, file.path(dir, "tiny_v5_spatial.rds"))
