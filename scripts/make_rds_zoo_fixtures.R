# Write tests/fixtures/tiny/zoo*.rds|rda: one of each base R object type the
# .rds parser (crates/scx-core/src/rds/xdr.rs) must step over or expand, so
# its tests don't depend on what a Seurat object happens to contain.
#
#   zoo.rds        saveRDS default: format version 3, gzip
#   zoo_v2.rds     format version 2 (no native-encoding header)
#   zoo.rda        save(): an RDX3 header and a pairlist of saved variables
#   zoo_ascii.rds  ASCII serialization, which the parser refuses
#
# Base R only. usage: Rscript scripts/make_rds_zoo_fixtures.R
out <- file.path("tests", "fixtures", "tiny")
e <- new.env()
assign("v", 1, envir = e)
assign("self", e, envir = e) # an environment that refers back to itself
zoo <- list(
  intseq = 1:5,                          # ALTREP compact_intseq
  realseq = (2^31):(2^31 + 2),           # ALTREP compact_realseq
  wrap_real = sort(c(3, 1, 2)),          # ALTREP wrap_real
  wrap_int = sort(c(3L, 1L, 2L)),        # ALTREP wrap_integer
  deferred = as.character(1:3),          # ALTREP deferred_string
  cplx = c(1 + 2i, 3 - 1i),
  raw = as.raw(1:4),
  latin1 = iconv("café", "UTF-8", "latin1"),
  strs = c("a", NA),
  lgl = c(TRUE, NA),
  env = e,
  fn = compiler::cmpfun(function(x, y) {
    z <- x + y
    if (z > 0) z * z else -z
  }),                                    # bytecode
  sd = stats::sd,                        # closure in a namespace
  builtin = sum,
  special = quote,
  extptr = new("externalptr"),
  formula = y ~ x,
  many = as.list(1:15)
)
saveRDS(zoo, file.path(out, "zoo.rds"))
saveRDS(zoo, file.path(out, "zoo_v2.rds"), version = 2)
save(zoo, file = file.path(out, "zoo.rda"))
saveRDS(zoo, file.path(out, "zoo_ascii.rds"), ascii = TRUE)
