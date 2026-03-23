suppressPackageStartupMessages({
  library(anndataR)
  library(Matrix)
})

# Locate golden fixtures relative to the package root.
# testthat::test_dir() sets cwd to the test directory, so go up twice.
PKG_ROOT   <- normalizePath(file.path(getwd(), "../.."))
SCX_ROOT   <- Sys.getenv("SCX_ROOT", unset = normalizePath(file.path(PKG_ROOT, "../..")))
GOLDEN     <- Sys.getenv("SCX_GOLDEN", unset = file.path(SCX_ROOT, "tests/golden"))

H5SEURAT_PATH <- file.path(GOLDEN, "pbmc3k.h5seurat")
H5AD_REF_PATH <- file.path(GOLDEN, "pbmc3k_reference.h5ad")

EXPECTED_N_OBS  <- 2700L
EXPECTED_N_VARS <- 13714L
EXPECTED_NNZ    <- 2282976L

skip_if_no_fixture <- function(...) {
  paths <- c(...)
  for (p in paths) {
    if (!file.exists(p)) {
      testthat::skip(paste0("fixture not found: ", p,
                            " — run `pixi run -e test fixtures`"))
    }
  }
}

skip_if_no_scx <- function() {
  # Native Rust binding is sufficient — no CLI binary needed
  if (picklerick:::.native_available()) return(invisible())
  bin <- getOption("picklerick.scx_binary") %||%
         Sys.getenv("PICKLERICK_SCX", unset = "") %||%
         Sys.which("scx")
  if (!nzchar(bin)) {
    testthat::skip("scx not available — install with native bindings or put scx on PATH")
  }
}

`%||%` <- function(x, y) if (!is.null(x) && nzchar(x)) x else y
