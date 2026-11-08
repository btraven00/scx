#!/usr/bin/env Rscript
# One small h5ad per anndataR generator type, written by anndataR itself, as
# a corpus for run_corpus.py: each file holds one type in one slot (X a CSR
# numeric matrix, every other slot empty), plus one file with every type.
#
#   Rscript gen_types.R out_dir      (needs anndataR >= 1.0; pixi env `test`)
#
# Writes out_dir/<id>/<id>.h5ad and out_dir/types_anndataR.tsv in the
# cxg_sample.py layout (id in dataset_version_id, slot/type in title). A type
# anndataR cannot write is listed with assay "write_failed" and no file.
suppressPackageStartupMessages(library(anndataR))
out <- commandArgs(trailingOnly = TRUE)[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
types <- get_generator_types()
slots <- c(X = "x_type", layers = "layer_types", obs = "obs_types", var = "var_types",
           obsm = "obsm_types", varm = "varm_types", obsp = "obsp_types",
           varp = "varp_types", uns = "uns_types")
none <- setNames(rep(list(character(0)), length(slots) - 1), unname(slots[-1]))

cases <- list(all = list())
for (slot in names(slots)) {
  for (t in types[[slot]]) {
    args <- none
    if (slot == "X") args$x_type <- t else { args$x_type <- "numeric_rsparse"; args[[slots[[slot]]]] <- t }
    cases[[paste0("anndataR-", slot, "-", t)]] <- args
  }
}

rows <- character(0)
for (id in names(cases)) {
  path <- file.path(out, id, paste0(id, ".h5ad"))
  dir.create(dirname(path), showWarnings = FALSE)
  ok <- tryCatch({
    ad <- do.call(generate_dataset, c(list(n_obs = 10L, n_vars = 20L, format = "AnnData"), cases[[id]]))
    write_h5ad(ad, path, mode = "w")
    TRUE
  }, error = function(e) { message(id, ": ", conditionMessage(e)); unlink(dirname(path), TRUE); FALSE })
  size <- if (ok) round(file.size(path) / 1e6, 3) else 0
  rows <- c(rows, paste("gen", id, "-", "-", 10, size, "-",
                        if (ok) "generated" else "write_failed", "-", id, sep = "\t"))
}
writeLines(c(sprintf("# anndataR %s generate_dataset, one type per file", packageVersion("anndataR")),
             paste("year", "dataset_version_id", "dataset_id", "collection_id", "cell_count",
                   "size_mb", "organism", "assay", "url", "title", sep = "\t"),
             rows), file.path(out, "types_anndataR.tsv"))
cat(length(rows), "cases,", sum(grepl("\tgenerated\t", rows)), "written\n")
