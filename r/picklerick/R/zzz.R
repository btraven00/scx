#' @keywords internal
.pkg_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Resolve scx binary once at load time; cache in .pkg_env.
  .pkg_env$scx_bin <- .resolve_scx_binary(warn = FALSE)
}

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

#' Locate the scx binary
#'
#' Resolution order:
#' 1. `options(picklerick.scx_binary = "/path/to/scx")`
#' 2. `PICKLERICK_SCX` environment variable
#' 3. `scx` on PATH (via `Sys.which`)
#'
#' @param warn Emit a warning when the binary cannot be found.
#' @return Absolute path string, or `NULL` if not found.
#' @keywords internal
.resolve_scx_binary <- function(warn = TRUE) {
  opt <- getOption("picklerick.scx_binary")
  if (!is.null(opt) && nchar(opt) > 0) return(opt)

  env <- Sys.getenv("PICKLERICK_SCX", unset = "")
  if (nchar(env) > 0) return(env)

  found <- Sys.which("scx")
  if (nchar(found) > 0) return(found)

  if (warn) {
    warning(
      "scx binary not found. Set options(picklerick.scx_binary = '/path/to/scx') ",
      "or PICKLERICK_SCX env var, or put scx on PATH.",
      call. = FALSE
    )
  }
  NULL
}

#' @keywords internal
.scx_binary <- function() {
  bin <- .pkg_env$scx_bin %||% .resolve_scx_binary(warn = TRUE)
  if (is.null(bin)) stop("scx binary not found — see ?picklerick for setup.", call. = FALSE)
  bin
}

# Backport `%||%` for R < 4.4
`%||%` <- function(x, y) if (!is.null(x)) x else y
