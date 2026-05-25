#!/usr/bin/env Rscript
# Benchmark picklerick::read_h5ad — time, peak alloc, GC across modes/datasets.
#
# Usage:
#   Rscript bench/r/read_h5ad_bench.R [label]
#
# label defaults to the current git short SHA (or "uncommitted").
# Output: bench/results/<label>.json

suppressPackageStartupMessages({
  library(picklerick)
  library(bench)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)

git_sha <- tryCatch(
  trimws(system("git rev-parse --short HEAD", intern = TRUE)),
  error = function(e) "uncommitted"
)
label <- if (length(args) >= 1) args[[1]] else git_sha

# Datasets — keep within iteration-friendly size. hlca_core (5.6 GB) excluded.
DATASETS <- list(
  pbmc3k  = "tests/golden/pbmc3k_reference.h5ad",
  norman  = "tests/golden/norman_subset.h5ad"
)

# Modes — `list` is the rawest path (no SCE assembly), SCE is the realistic
# user call, lazy is the metadata-only baseline.
MODES <- c("list", "SingleCellExperiment", "lazy")

# Peak RSS via /proc/self/status — captures Rust-side allocations that
# bench::mark misses (it only tracks R-GC-visible allocs).
peak_rss_mb <- function() {
  s <- readLines("/proc/self/status")
  m <- grep("^VmHWM:", s, value = TRUE)
  if (length(m) == 0) return(NA_real_)
  as.numeric(sub("[^0-9]*([0-9]+).*", "\\1", m)) / 1024
}

reset_rss <- function() {
  gc(verbose = FALSE, full = TRUE)
  # VmHWM is high-water mark since process start; can't reset. We record
  # rss_before/after instead and report the delta per-iteration.
  invisible(NULL)
}

run_one <- function(path, mode) {
  call <- if (mode == "lazy") {
    function() picklerick::read_h5ad(path, as = "SingleCellExperiment", lazy = TRUE)
  } else {
    function() picklerick::read_h5ad(path, as = mode)
  }

  # Warmup — fold one-time S4 dispatch / Rust dyn.load costs out of the
  # measurement. The original 434 MB pbmc3k SCE alloc was this artifact.
  invisible(call()); gc(verbose = FALSE, full = TRUE)

  rss_before <- peak_rss_mb()
  m <- bench::mark(call(),
                   iterations = 10, check = FALSE,
                   filter_gc = FALSE, memory = TRUE)
  rss_after <- peak_rss_mb()
  list(mark = m, rss_before_mb = rss_before, rss_after_mb = rss_after)
}

results <- list()
for (ds_name in names(DATASETS)) {
  path <- DATASETS[[ds_name]]
  if (!file.exists(path)) {
    message(sprintf("SKIP %s — missing %s", ds_name, path))
    next
  }
  size_mb <- round(file.info(path)$size / 1e6, 1)
  for (mode in MODES) {
    key <- sprintf("%s/%s", ds_name, mode)
    message(sprintf("Running %s (size=%.1f MB) ...", key, size_mb))
    out <- tryCatch(run_one(path, mode), error = function(e) {
      message("  ERROR: ", conditionMessage(e))
      NULL
    })
    if (is.null(out)) next
    res    <- out$mark
    iter_s <- as.numeric(res$time[[1]])
    results[[key]] <- list(
      dataset            = ds_name,
      mode               = mode,
      file_size_mb       = size_mb,
      min_s              = as.numeric(res$min),
      median_s           = as.numeric(res$median),
      mean_s             = mean(iter_s),
      max_s              = max(iter_s),
      iter_s             = iter_s,
      mem_alloc_b        = as.numeric(res$mem_alloc),
      total_time_s       = as.numeric(res$total_time),
      n_itr              = as.integer(res$n_itr),
      n_gc               = as.integer(res$n_gc),
      rss_before_mb      = out$rss_before_mb,
      rss_after_mb       = out$rss_after_mb,
      rss_delta_mb       = out$rss_after_mb - out$rss_before_mb
    )
  }
}

meta <- list(
  label       = label,
  git_sha     = git_sha,
  branch      = tryCatch(trimws(system("git rev-parse --abbrev-ref HEAD", intern = TRUE)),
                         error = function(e) NA_character_),
  timestamp   = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  r_version   = paste(R.version$major, R.version$minor, sep = "."),
  picklerick  = as.character(packageVersion("picklerick")),
  results     = results
)

out <- file.path("bench/results", paste0(label, ".json"))
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
write_json(meta, out, pretty = TRUE, auto_unbox = TRUE)
message(sprintf("Wrote %s", out))

# Console summary
cat("\n=== Summary ===\n")
for (key in names(results)) {
  r <- results[[key]]
  cat(sprintf("  %-32s median=%7.1fms  R-alloc=%6.1fMB  RSS+=%6.1fMB  gc=%d\n",
              key, r$median_s * 1000, r$mem_alloc_b / 1e6,
              r$rss_delta_mb, r$n_gc))
}
