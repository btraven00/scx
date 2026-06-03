# Cutting picklerick's 4× memory expansion — ranked options

Captured 2026-05-25 after the zero-copy investigation on
`feat/picklerick-zerocopy` (now squash-merged as 63ab9b6). Companion to
`scratch/r-altrep.md` (ALTREP scope) and the "Picklerick R zero-copy
thread" section of `docs/roadmap.md` (next-steps list).

## The number

pbmc3k SCE: 29 MB file → 113 MB RSS+ during `read_h5ad(as="SCE")` =
**~4×**. Heaptrack peak heap 622 MB on the same call, dominated by R's
own SEXP page allocator (`GetNewPage` = 451 MB / 72%).

## Diagnosis: the X matrix is materialized twice in R

The pbmc3k X payload is ~27 MB of SEXP-resident data:

  - `x_data`    REALSXP, 2.3 M × 8 B  = 18 MB
  - `x_indices` INTSXP,  2.3 M × 4 B  =  9 MB
  - `x_indptr`  INTSXP,  ~2.7 K × 4 B = ~11 KB

The current path materialises this **twice** simultaneously at peak:

  1. `picklerick:::scx_read()` returns a raw R list with
     `raw$x_data`, `raw$x_indices`, `raw$x_indptr` fields — these are
     R SEXPs allocated by the extendr bridge.
  2. `.as_sce()` calls `.build_dgc(raw)` →
     `methods::new("dgCMatrix", p = raw$x_indptr, i = raw$x_indices,
     x = raw$x_data, …)` — R must **copy** each vector into the S4
     object's slot because the raw list still holds references at
     refcount ≥ 2.

Both copies are alive at peak. `SingleCellExperiment(...)` then wraps
them in an `Assays` object on top. That's the ~3-4× headline.

Why this is invisible to bench::mark's `mem_alloc` and partially
invisible to RSS HWM: R's GetNewPage allocates fixed-size pages and
reuses them — peak shows R's page-pool size, not the discrete copies.
Heaptrack confirms the copy structurally (411 K allocations, of which
56,874 are GetNewPage calls).

## Ranked options to attack it

### Option 1 — Rust constructs the dgCMatrix directly  (~25 % cut)

Have `scx_read` (or a new `scx_read_dgc`) allocate the dgCMatrix S4
SEXP from Rust and fill `@i` / `@p` / `@x` directly from the streaming
reader. Return the dgCMatrix in place of the raw `x_data`/`x_indices`/
`x_indptr` triplet. Eliminates copy (2).

  - Saving:    ~27 MB on pbmc3k  ⇒  113 MB → ~86 MB  (4× → ~3×)
  - Scope:     localized to r/picklerick/src/rust/src/lib.rs + a
               replacement .as_sce path; keep `as="list"` on the
               current path for callers that need raw triplets
  - Risk:      extendr's S4 support; dgCMatrix slot validation; need
               to populate `Dim` / `Dimnames` / `factors` slots
  - Verify:    heaptrack diff against 63ab9b6 baseline

### Option 2 — Make `lazy = TRUE` the default for SCE  (~25× cut, if fast)

Already implemented; uses `HDF5Array::H5SparseMatrix` so X stays on
disk as a DelayedMatrix. Peak drops to metadata-only (a few MB).
Biggest possible memory win.

  - Blocker:   on norman (83 MB), lazy is *slower* than eager —
               1209 ms vs 1016 ms. Suspect HDF5Array per-call file-
               open / setup tax. Need profiling to confirm and
               likely a file-handle cache.
  - Saving:    ~100 MB → ~5 MB on pbmc3k SCE peak  (if fast enough)
  - Scope:     R-side; investigate HDF5Array overhead before
               flipping the default
  - Risk:      changing default behaviour breaks code expecting
               in-memory `counts()`; need a `lazy = "auto"` heuristic
               or a deprecation path
  - Verify:    bench harness already covers list/SCE/lazy; rerun

### Option 3 — Stream chunks into a preallocated dgCMatrix  (chunk-bounded peak)

Strictly stronger than option 1. Two-pass: (a) read metadata + nnz,
(b) allocate dgCMatrix `@i`/`@p`/`@x` upfront, stream chunks
straight into the slot buffers without ever building a Rust-side
concatenated `Vec<f64>`.

  - Saving:    peak ≈ chunk_size, not dataset-size. Beats option 1.
               Pbmc3k could hit ~5–10 MB peak; atlas-scale wins are
               proportional.
  - Scope:    bigger: needs a `H5AdReader::nnz()` (or a streaming
               nnz probe), plus a metadata-first read pass; plus the
               extendr glue to fill slot buffers incrementally
  - Risk:    second open of the same file is page-cache hot but
              still costs syscalls; for compressed h5ad it doubles
              decompression work for the nnz pass — measure first
  - Verify:    heaptrack + bench harness

## Update 2026-05-25 (later): uns parsing dominates on norman

Profiled the lazy path on norman to chase the suspected "lazy slower than
eager" slowdown. Two findings reshape this whole document:

### 1. "Lazy slower than eager" was system contention, not structural

Under proper warmup on a quiet machine: eager 1006 ms, lazy 1024 ms.
+1.9 % is within noise. The 1209 vs 1016 ms numbers in the 63ab9b6
baseline JSON were taken under zed-editor + firefox + claude CPU
contention. Option 2 above as a *time* fix is moot — lazy is already
≈ eager.

### 2. `.parse_uns(raw$uns_json)` is the dominant cost on norman

profvis on full lazy call:
  .parse_uns / jsonlite::fromJSON / parseJSON  = 282 / 385 samples (73 %)

The norman uns_json is **33 MB of raw JSON** parsing to a **284 MB
R list** in 707 ms (jsonlite::fromJSON, simplifyVector=FALSE). The
five top-level keys are perturb-seq workflow annotations:
non_dropout_gene_idx, non_zeros_gene_idx, rank_genes_groups_cov_all,
top_non_dropout_de_20, top_non_zero_de_20 — nested lists of gene-index
vectors per condition.

Verified by hand-rolling a lazy path that skips `.parse_uns`:
  current full lazy SCE                1020.7 ms
  hand-rolled lazy SCE, no uns parse    420.5 ms   ← 2.4× faster
  parsed uns held in SCE metadata: 283.8 MB

So norman's 390 MB RSS+ SCE peak is *not* dominated by X copies (as we
diagnosed for pbmc3k above) — it's ~284 MB of parsed uns + ~50 MB of
X + ~50 MB of overhead. **Skipping uns parsing saves 284 MB and
600 ms in one change.**

### Reshaping the ranked options

For pbmc3k-class data (small uns) the original ranking holds: 4× peak
is dominated by the X-double-materialization, and the fix is option 1
(Rust-built dgCMatrix) or option 3 (preallocated streaming fill).

For norman-class data (rich perturb-seq / atlas uns) a **new top
option** dominates:

### Option 0 — Lazy uns by default  (~75 % cut on norman SCE)  — SHIPPED on feat/picklerick-uns-lazy

Measured on `feat/picklerick-uns-lazy` against v0.2.0 baseline:

  | mode             | baseline           | uns-lazy         | speedup |
  |------------------|-------------------:|-----------------:|--------:|
  | norman / list    |   983 ms /     ?   |     4 ms / 0 MB  |  245×   |
  | norman / SCE     |  2893 ms / 608 MB  |    45 ms / 0 MB  |   64×   |
  | norman / lazy    |  2056 ms /  81 MB  |    95 ms / 0 MB  |   22×   |
  | pbmc3k / SCE     |   162 ms / 165 MB  |   135 ms / 165MB |  -17 %  |

Why so much bigger than the predicted 60 %: the Rust side also stopped
serializing uns to JSON when read_uns=FALSE, which avoided not just the
R-side jsonlite parse but the 33 MB string allocation + HDF5 `/uns`
group walk on the Rust side too. Most of "metadata read" cost on
norman was actually uns enumeration + JSON serialisation, not obs/var.

Bonus: parse_uns = TRUE (opt-in eager) now also routes via rhdf5,
so the parsed uns is **110 MB** instead of 284 MB even when requested
— native int32 typing instead of jsonlite's JSON-number → REALSXP
inflation.

What landed (see commit on feat/picklerick-uns-lazy):
  - read_h5ad gains parse_uns = FALSE default
  - new uns() accessor: uns(sce), uns(sce, key), uns(sce, key, sub_key)
  - scx_read gains read_uns: bool arg (default FALSE from R) — Rust
    skips reader.uns().await and JSON serialisation entirely
  - parse_uns = TRUE routes through rhdf5::h5read("/uns") for a typed
    eager parse; the old JSON path remains as fallback for non-H5AD

`read_h5ad` should not eagerly parse `uns` into an R list. The raw
JSON string is already returned by `scx_read`; just store it (or a
deferred-parse wrapper) in SCE metadata instead of running
jsonlite::fromJSON on it.

  - Saving:    norman:  390 MB → ~106 MB peak  (75 % cut)
               norman:  1020 ms → ~420 ms      (60 % faster)
               pbmc3k:  near-zero (small uns)
  - Scope:     r/picklerick/R/read.R — change `.parse_uns(...)` default
               or add `parse_uns = FALSE` arg; downstream wrappers
               (.as_sce, .as_sce_lazy, .as_seurat) thread it through.
               Optional: helper `parse_uns(sce)` that materialises on
               demand by reading `metadata(sce)$uns_json`.
  - Risk:      callers expecting `metadata(sce)$<key>` to work directly
               will break — needs a parse_uns() helper or a clear note
               in NEWS. The breakage is detectable (NULL instead of
               value), not silent corruption.
  - Verify:    re-run bench harness on norman; expect time drop to
               ~420 ms and RSS+ drop to ~100 MB.

Even better (separate change): parse uns in Rust directly into a
nested Robj — skips the 33 MB JSON intermediate too. But that's
a time optimisation, not a memory one (the parsed list is still 284 MB
either way); only matters if uns *is* requested.

## Recommendation: investigate option 2 first  (superseded)

Superseded by option 0 above (uns parsing is the dominant cost on
norman-class data). Take this path first for any dataset whose uns
is non-trivial; then revisit options 1/2/3 for the dgCMatrix copy
on pbmc3k-class data.

## (Original recommendation, kept for context)

Option 2 is **measurement before code**. Spend a profvis or heaptrack
session on `read_h5ad(..., lazy = TRUE)` for norman, identify why it's
slower than eager (almost certainly HDF5Array setup), and fix that.
If it pans out → lazy becomes the default → 4× becomes ~0.2× (5 MB
peak instead of 113 MB) → done. No new Rust code.

If option 2 doesn't pan out (e.g. HDF5Array overhead is structural),
go to option 1 for a guaranteed ~25 % cut, then option 3 if you want
to push toward chunk-bounded peak.

## What we explicitly ruled out

  - **Trimming Rust-side `Vec<T>` intermediates** in the extendr bridge
    (the single-pass-write change shipped in 63ab9b6). Heaptrack: 0
    bytes of peak movement. R's page allocator dominates; transient
    Rust Vecs are below the resolution of peak-heap measurement.
  - **Real ALTREP+mmap for h5ad.** HDF5's chunked + compressed pipeline
    rules out direct mmap of `/X/data`. The h5ad equivalent of
    "lazy zero-copy" is already option 2 (HDF5Array / DelayedArray).
    See `scratch/r-altrep.md` for the structural argument.

## Verifying any of these

Build → install with the now-correct Makevars dep tracking → run
`Rscript bench/r/heaptrack_runner.R pbmc3k SingleCellExperiment`
under `heaptrack`, then `heaptrack_print` and grep for `GetNewPage`
peak. Compare against 63ab9b6's baseline numbers (above).

Watch for the Makevars trap if numbers look identical: confirm
`r/picklerick/src/rust/target/release/libpicklerick_r.a` mtime moved
after your edit.
