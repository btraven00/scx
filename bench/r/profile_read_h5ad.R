#!/usr/bin/env Rscript
# Profile picklerick::read_h5ad with profvis on the pbmc3k SCE path.
# This is the dominant R-side allocation case (434 MB for a 28 MB file).
#
# Usage:
#   Rscript bench/r/profile_read_h5ad.R [out_html]
#
# Default out: bench/results/profvis-<sha>.html

suppressPackageStartupMessages({
  library(picklerick)
  library(profvis)
})

args <- commandArgs(trailingOnly = TRUE)
sha  <- tryCatch(trimws(system("git rev-parse --short HEAD", intern = TRUE)),
                 error = function(e) "uncommitted")
out  <- if (length(args) >= 1) args[[1]] else
        file.path("bench/results", sprintf("profvis-%s.html", sha))

path <- "tests/golden/pbmc3k_reference.h5ad"
if (!file.exists(path)) stop("missing fixture: ", path)

# Warmup so we don't measure dyn.load / first-page-fault noise.
invisible(picklerick::read_h5ad(path, as = "SingleCellExperiment"))
gc(verbose = FALSE)

p <- profvis::profvis(
  for (i in 1:3) picklerick::read_h5ad(path, as = "SingleCellExperiment"),
  interval = 0.005
)

htmlwidgets::saveWidget(p, out, selfcontained = TRUE)
message("Wrote ", out)

# Also dump a quick text summary of the top R-level functions by self-time
# to stdout so terminal-only review is possible.
prof_data <- p$x$message$prof
if (!is.null(prof_data) && nrow(prof_data) > 0) {
  tbl <- as.data.frame(table(prof_data$label))
  tbl <- tbl[order(-tbl$Freq), ]
  cat("\n=== Top 20 functions by sample count (5ms interval) ===\n")
  print(head(tbl, 20), row.names = FALSE)
}
