suppressPackageStartupMessages({
  library(anndataR)
  library(Matrix)
})

GOLDEN     <- Sys.getenv("SCX_GOLDEN", unset = "../golden")
REF_PATH   <- file.path(GOLDEN, "pbmc3k_reference.h5ad")
SCX_PATH   <- file.path(GOLDEN, "pbmc3k_scx.h5ad")

EXPECTED_N_OBS  <- 2700L
EXPECTED_N_VARS <- 13714L
EXPECTED_NNZ    <- 2282976L

skip_if_no_golden <- function(...) {
  paths <- c(...)
  for (p in paths) {
    if (!file.exists(p)) {
      testthat::skip(paste0("golden file not found: ", p,
                            " — run `pixi run -e test fixtures`"))
    }
  }
}

# Load once per session; individual tests call skip_if_no_golden() first.
.scx_ad <- NULL
.ref_ad <- NULL

scx_ad <- function() {
  if (is.null(.scx_ad)) {
    skip_if_no_golden(SCX_PATH)
    .scx_ad <<- read_h5ad(SCX_PATH)
  }
  .scx_ad
}

ref_ad <- function() {
  if (is.null(.ref_ad)) {
    skip_if_no_golden(REF_PATH)
    .ref_ad <<- read_h5ad(REF_PATH)
  }
  .ref_ad
}
