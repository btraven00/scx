#!/usr/bin/env Rscript
# Profvis the full lazy=TRUE call on norman to find the 580 ms unaccounted
# for in the stage-by-stage decomposition.

suppressPackageStartupMessages({
  library(picklerick)
  library(SingleCellExperiment)
  library(HDF5Array)
  library(profvis)
})

path <- "tests/golden/norman_subset.h5ad"

# Warmup everything.
invisible(read_h5ad(path, as = "SingleCellExperiment"))
invisible(read_h5ad(path, as = "SingleCellExperiment", lazy = TRUE))
gc(verbose = FALSE, full = TRUE)

p <- profvis::profvis(
  for (i in 1:5) read_h5ad(path, as = "SingleCellExperiment", lazy = TRUE),
  interval = 0.005
)
htmlwidgets::saveWidget(p, "scratch/profvis-lazy-norman.html", selfcontained = TRUE)

# Console-friendly top-N summary by sample count.
pd <- p$x$message$prof
if (!is.null(pd) && nrow(pd) > 0) {
  tbl <- as.data.frame(table(pd$label))
  tbl <- tbl[order(-tbl$Freq), ]
  cat("\n=== Top 30 by sample count (5ms interval, 5 iterations) ===\n")
  print(head(tbl, 30), row.names = FALSE)
}
