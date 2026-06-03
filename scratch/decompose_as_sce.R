#!/usr/bin/env Rscript
# Find where the ~400 MB of R-side allocations in .as_sce() actually go.
# Compares stage-by-stage allocs against the full call.

suppressPackageStartupMessages({
  library(picklerick)
  library(bench)
  library(Matrix)
  library(SingleCellExperiment)
})

path <- "tests/golden/pbmc3k_reference.h5ad"
raw  <- picklerick:::scx_read(path, 5000L, read_x = TRUE)

# Each stage in isolation, measured via bench::mark which reports R-GC-visible
# allocations. We re-fetch raw fresh each iteration to avoid refcount bias.
stage_alloc <- function(expr_quoted) {
  m <- bench::mark(eval(expr_quoted), iterations = 3, check = FALSE)
  list(median_ms  = as.numeric(m$median) * 1000,
       alloc_mb   = as.numeric(m$mem_alloc) / 1e6)
}

# Stage 1: build the dgCMatrix only.
s1 <- stage_alloc(quote(picklerick:::.build_dgc(raw)))

# Stage 2: build obs/var data.frames.
s2 <- stage_alloc(quote(picklerick:::.cols_to_df(raw$obs_cols, raw$obs_index)))
s3 <- stage_alloc(quote(picklerick:::.cols_to_df(raw$var_cols, raw$var_index)))

# Stage 4: wrap each in S4Vectors::DataFrame.
obs_df <- picklerick:::.cols_to_df(raw$obs_cols, raw$obs_index)
var_df <- picklerick:::.cols_to_df(raw$var_cols, raw$var_index)
s4 <- stage_alloc(quote(S4Vectors::DataFrame(obs_df)))
s5 <- stage_alloc(quote(S4Vectors::DataFrame(var_df)))

# Stage 6: obsm processing.
s6 <- stage_alloc(quote({
  lapply(names(raw$obsm), function(nm) {
    m <- raw$obsm[[nm]]
    if (!is.null(m) && nrow(m) == length(raw$obs_index)) rownames(m) <- raw$obs_index
    m
  })
}))

# Stage 7: SCE constructor with pre-built pieces.
m_pre        <- picklerick:::.build_dgc(raw)
obs_df_pre   <- S4Vectors::DataFrame(picklerick:::.cols_to_df(raw$obs_cols, raw$obs_index))
var_df_pre   <- S4Vectors::DataFrame(picklerick:::.cols_to_df(raw$var_cols, raw$var_index))
reduced_pre  <- list()
for (nm in names(raw$obsm)) {
  m <- raw$obsm[[nm]]
  if (!is.null(m) && nrow(m) == length(raw$obs_index)) rownames(m) <- raw$obs_index
  reduced_pre[[nm]] <- m
}
s7 <- stage_alloc(quote(SingleCellExperiment::SingleCellExperiment(
  assays      = list(counts = m_pre),
  colData     = obs_df_pre,
  rowData     = var_df_pre,
  reducedDims = reduced_pre
)))

# Full call for reference.
full <- stage_alloc(quote(picklerick::read_h5ad(path, as = "SingleCellExperiment")))

stages <- list(
  "1. .build_dgc"                  = s1,
  "2. .cols_to_df(obs)"            = s2,
  "3. .cols_to_df(var)"            = s3,
  "4. S4Vectors::DataFrame(obs)"   = s4,
  "5. S4Vectors::DataFrame(var)"   = s5,
  "6. obsm processing"             = s6,
  "7. SCE constructor (pre-built)" = s7,
  "= FULL read_h5ad SCE"           = full
)

cat(sprintf("%-40s  %8s  %10s\n", "stage", "ms", "alloc MB"))
cat(strrep("-", 64), "\n")
sum_ms <- 0; sum_mb <- 0
for (nm in names(stages)) {
  s <- stages[[nm]]
  cat(sprintf("%-40s  %8.1f  %10.1f\n", nm, s$median_ms, s$alloc_mb))
  if (!startsWith(nm, "=")) { sum_ms <- sum_ms + s$median_ms; sum_mb <- sum_mb + s$alloc_mb }
}
cat(strrep("-", 64), "\n")
cat(sprintf("%-40s  %8.1f  %10.1f\n", "  (sum of stages 1-7)", sum_ms, sum_mb))
