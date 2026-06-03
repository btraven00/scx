# ALTREP: lazy / alternative vector representations in R

## What it is

**ALTREP** ("Alternative Representations") is a C-level API added in R
3.5 (2018) that lets a package define a vector whose underlying storage
is *not* a plain contiguous C array sitting in R's heap.

From R code, an ALTREP vector looks and behaves like a normal vector —
`length()`, `[`, `sum()`, `as.numeric()`, printing, etc. all work. Under
the hood, the package supplies a vtable of C callbacks that R invokes
when it needs data: "give me element i", "give me a pointer to the full
buffer", "what is your length", "are you sorted", "subset me", "iterate
into this destination buffer."

## Examples already in base R

- `1:1e9` does **not** allocate a billion integers. It's an ALTREP
  "compact sequence" that stores `(from, to, by)` — three numbers.
  `length()` and `[]` are computed on demand.
- `sort(x)` returns an ALTREP that remembers it's sorted, so the next
  `sort()` or `is.unsorted()` is free.
- Deferred string conversion: `as.character(1:1000)` doesn't actually
  build the character strings until something asks for them.
- `seq_len(n)`, `rev()` on certain inputs — same trick.

You almost certainly use ALTREP daily without realizing it.

## Why it matters for file I/O

If your on-disk data layout *exactly* matches R's in-memory layout, you
can `mmap` the file and define an ALTREP class whose "data pointer"
callback returns a pointer into the mapped pages. R operations then
read straight from the OS page cache — **zero copy, zero allocation**.

Concrete sweet spot for numpy `.npy` files:
- native-endian float64 → R `numeric`
- native-endian int32 → R `integer`
- contiguous, C-order
- no dtype conversion

Outside this sweet spot (float16, endian-swap, bit64, structured
dtypes, Fortran order), you have to allocate and convert anyway, and
ALTREP buys you nothing over the in-place SEXP write trick.

## The vtable, roughly

You register a class on package load and supply callbacks. The main
ones:

```c
// Required
R_set_altrep_Length_method(cls, my_length);
R_set_altrep_Inspect_method(cls, my_inspect);   // for str() / inspect

// "Standard" vector ops
R_set_altvec_Dataptr_method(cls, my_dataptr);          // mutable ptr; forces materialization
R_set_altvec_Dataptr_or_null_method(cls, my_dataptr_ro); // read-only ptr; can return NULL

// Per-type element accessors (e.g. for REALSXP)
R_set_altreal_Elt_method(cls, my_real_elt);            // x[i]
R_set_altreal_Sum_method(cls, my_real_sum);            // sum(x), optional fast path
R_set_altreal_Min_method(cls, my_real_min);
R_set_altreal_Max_method(cls, my_real_max);

// Optional: copy a range into a caller-provided buffer (the big win)
R_set_altreal_Get_region_method(cls, my_real_get_region);
```

`Get_region` is the callback you most want to implement well: many
R-internals consumers (and packages like `vroom`) ask for a range of
elements to be copied into a destination, which lets you stream from
mmap or disk without ever exposing a full buffer.

The "instance" of an ALTREP vector is itself a SEXP carrying two
auxiliary SEXPs — `data1` and `data2` — where you stash whatever state
your callbacks need (file descriptor, mmap pointer, offset, length,
dtype info, a finalizer-bearing external pointer, etc.).

## The catch: `DATAPTR()` materialization

Many R operations call `DATAPTR(x)` to get a plain `double*` / `int*` so
they can run a tight C loop. If your ALTREP can't satisfy that with a
stable pointer (because data lives in mmapped pages with the right
layout, you can — return the mapped address), R will call your
`Dataptr` method which is *required* to return a real pointer.

For mmap-backed ALTREP with matching layout, `Dataptr` is cheap: hand
back the mapped address. **Zero copy.**

For anything that needs conversion, `Dataptr` has to materialize: it
allocates a full-size buffer and runs your conversion. From that point
on the vector is fully realized in memory and the laziness is gone.

So the practical pattern is:

- Implement `Length`, `Elt`, `Get_region` cheaply (these stay lazy).
- Implement `Dataptr_or_null` to return the mmap pointer when layout
  matches, and `NULL` otherwise (signals "no cheap pointer available,
  please use `Get_region` if you can").
- Implement `Dataptr` as the materializing fallback. Cache the
  materialized buffer in `data2` so it only happens once.

## Lifetime / GC

The mmap (or file handle, or whatever backs your data) outlives the
ALTREP instance, so you need a finalizer.

Idiomatic recipe:
- Create an external pointer (`R_MakeExternalPtr`) wrapping a struct
  with the fd + mapped address + length.
- Register a finalizer on it: `R_RegisterCFinalizerEx(extptr, my_unmap, TRUE)`.
- Stash that extptr as `data1` (or inside a list at `data1`).
- When R GCs the ALTREP, the extptr goes with it, the finalizer fires,
  you `munmap` and `close`.

`TRUE` in `R_RegisterCFinalizerEx` means "run on exit too" — important
so you don't leak mappings if the session is ending.

## Cross-platform

- Linux/macOS: `mmap` + `munmap`, straightforward.
- Windows: `CreateFileMapping` + `MapViewOfFile` + `UnmapViewOfFile`.
  Different API, same shape. Conditional compile.

## ALTREP from Rust (extendr / savvy)

Both crates expose enough of the C API to define ALTREP classes, but:

- Bindings are thinner than the C API. Some callbacks (especially
  `Get_region` for less-common types) may need raw FFI.
- Lifetime story is fiddly: the Rust struct that owns the mmap has to
  live exactly as long as the ALTREP SEXP. The idiomatic move is to
  `Box::into_raw` the owning struct, hand the pointer to an external
  pointer with a finalizer that does `Box::from_raw` + drop.
- No panic across the FFI boundary — every callback must catch and
  translate to `Rf_error`.
- savvy in particular keeps the SEXP API close to the surface, which
  helps; extendr abstracts more, which sometimes gets in the way for
  ALTREP specifically.

For picklerick specifically: ALTREP is mostly interesting for pickled
numpy arrays where the payload is a native-layout numeric block. Pure
Python objects (dicts, lists of mixed types, custom classes) materialize
into R lists and don't benefit.

## When ALTREP is the right call

Use ALTREP when:
- Files are big enough that allocating a full copy hurts (GB+).
- The on-disk layout matches R's in-memory layout for the common case
  (otherwise you're just doing eager conversion behind a lazy facade —
  same perf, more code).
- Downstream consumers either use element-at-a-time access or call
  `Get_region` (vroom-style consumers, head/tail/summary stats). If the
  next thing the user does is always `as.matrix()` or a full-vector C
  loop via `DATAPTR()`, the lazy win evaporates.

Skip ALTREP when:
- Files comfortably fit in 1× their size in memory — the in-place SEXP
  write pattern is simpler and gets you most of the way.
- All dtypes need conversion anyway.
- API surface needs to feel like a normal eager `read_npy()` — adding a
  proxy class changes user expectations (e.g., what `str(x)` shows, how
  serialization round-trips).

A reasonable design is **both**: eager `read_npy()` using the in-place
SEXP trick (your fast path for everything), plus `read_npy(..., lazy =
TRUE)` returning an ALTREP-backed view for the native-layout sweet spot.

## Reading further

- Luke Tierney's original design document:
  <https://svn.r-project.org/R/branches/ALTREP/ALTREP.html>. Canonical
  source for the *why* and overall model. Note: it's the design doc
  from the development branch, not maintained reference docs — some
  specifics drifted post-merge. Cross-check signatures against the
  installed `R_ext/Altrep.h`.
- R Internals manual, ALTREP section:
  <https://cran.r-project.org/doc/manuals/r-release/R-ints.html>.
- `vroom` source — heavy ALTREP usage for lazy CSV columns; the
  canonical real-world example.
- R-internals header: `Rinternals.h` and `R_ext/Altrep.h`.
- Tomas Kalibera's blog posts on ALTREP semantics and GC interactions.

## Findings from the 2026-05-25 picklerick zero-copy session

We spent a session trying to apply this thinking to `picklerick::read_h5ad`
and came back with three concrete lessons that change where ALTREP fits.

### 1. ALTREP+mmap does not apply to h5ad payloads

The original framing here implicitly assumed an on-disk layout you can mmap
and use directly. `.npy` qualifies. `.h5ad` does not: HDF5 stores datasets
through a chunked + filtered + (often) compressed pipeline. You cannot
mmap `/X/data` and treat the bytes as a `double*` — they live inside HDF5
chunks and have to go through `H5Dread` (decompression, layout transforms,
dataspace selection). The "native-layout sweet spot" the doc above
describes is structurally absent in h5ad.

The h5ad-shaped equivalent is `HDF5Array::H5SparseMatrix` + DelayedArray:
not ALTREP, but the same lazy semantics — slot access triggers an HDF5
read, R never materialises the full matrix. picklerick's existing
`lazy = TRUE` mode already uses this. That *is* "the ALTREP for h5ad"
in spirit, even though it's a different mechanism.

Genuine ALTREP+mmap belongs to formats we control with native layouts:
`.npy`, or the future `.scxd` probe described in `scratch/format-bench.md`
and `scratch/format_comparison.md`.

### 2. R's page allocator dominates over Rust-side intermediates

We attempted a Rust-side micro-optimization in
`r/picklerick/src/rust/src/lib.rs` — replace the
`Vec<u64> -> Vec<i32> -> INTSXP` pattern with `Vec<u64> -> INTSXP` single-
pass writes via extendr's `Integers::new(len)` (which calls
`Rf_allocVector(INTSXP, len)` then derefs to a `&mut [Rint]`). For
pbmc3k's ~2.3M x_indices that's a ~9 MB transient Vec eliminated.

Heaptrack on a single pbmc3k SCE read (no warmup) showed it was a no-op:

                          baseline       optimized
  total alloc calls       411,521        411,505    (-16)
  temporary allocs         57,857         57,857     ( 0)
  peak heap consumption   622.12 MB      622.12 MB   ( 0)
  total leaked            573.15 MB      573.15 MB   ( 0)

Top peak consumer in both runs is identical: `GetNewPage` in `libR.so` at
451.25 MB / 56,874 calls — R's own SEXP page allocator, accounting for
~72% of peak. The Rust transient Vec was a brief detour the peak-heap
measurement could not see.

The real implication: to lower peak R memory in `read_h5ad`, the
optimization has to **reduce the number/size of R SEXP allocations** —
either fill a preallocated dgCMatrix slot from streaming chunks (skips
one full SEXP), or wrap on-disk slots as ALTREP-backed (skips the SEXP
entirely until accessed). Trimming Rust intermediates is invisible.

### 3. The Makevars stale-binary trap

`r/picklerick/src/Makevars` had `$(STATLIB):` with no source deps, so
Make only invoked `cargo build` if `libpicklerick_r.a` didn't exist.
Once built, Rust source changes never triggered rebuilds — the loaded
`.so` could be arbitrarily stale. We hit a binary 7 days old. This
masked all our bench iterations until heaptrack showed byte-identical
allocation traces between supposedly different builds. Now fixed.

If you come back to this thread and the bench numbers look suspiciously
unchanged across edits, double-check this first: confirm
`r/picklerick/src/rust/target/release/libpicklerick_r.a` mtime moved
after your edit.

### 4. HDF5 ABI drift surfaced once builds actually worked

Forcing a real rebuild also surfaced an unrelated issue: the test env's
`bioconductor-hdf5array` ships `libhdf5.so.310` (HDF5 1.14+), which
dropped the `H5Literate` symbol that the vendored scx-core archive still
referenced. The stale-binary bug was hiding this too. Fix landed in the
same session — picklerick now uses the workspace's `vendored-hdf5`
default feature (commit 5992b9a's deferred follow-up), bundling HDF5 +
zlib inside `libpicklerick_r.a`. No system libhdf5 at link or load time.

### So when *would* ALTREP help here?

- **`.npy` lazy reads**: original use case. Native-layout mmap, ALTREP
  with `Dataptr_or_null` returning the mapped address. Real zero-copy.
- **`.scxd` probe**: same shape as `.npy` but bundled with sparse
  triplet + metadata. Also a real zero-copy candidate.
- **h5ad**: skip. Use `HDF5Array::H5SparseMatrix` / DelayedArray. The
  picklerick `lazy = TRUE` path already does this.
