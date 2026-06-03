# Skipping the double allocation when reading binary data into R

## Background: what a SEXP is

`SEXP` ("S-expression pointer") is the C-level type for every R object —
integer vector, double vector, list, environment, NULL. The name is a
holdover from R's Lisp ancestry; in practice it just means "pointer to an
R object."

A `.Call` from R into C arrives with each argument as a `SEXP`, and the
C function returns a `SEXP`:

```c
SEXP my_fn(SEXP arg1, SEXP arg2) {
    SEXP out = PROTECT(allocVector(REALSXP, n));   // ask R's heap for a numeric vector
    double *p = REAL(out);                          // raw C pointer to its contents
    // ... fill p[0..n-1] ...
    UNPROTECT(1);
    return out;
}
```

`allocVector(REALSXP, n)` allocates a length-`n` R numeric vector on R's
heap. `REAL(out)` (and `INTEGER`, `RAW`, etc.) hands you a plain C pointer
to its data buffer. `PROTECT`/`UNPROTECT` keep the GC from collecting the
object while you're filling it.

## The two-allocation pattern

The typical R-side recipe for "read a typed binary array from disk" looks
like this:

1. **R side** — `readBin(con, "raw", n = payload_size)`. R allocates a
   `RAWSXP` (raw byte vector) sized to the entire file payload, then
   copies the file contents into it. Allocation #1.
2. **C side** — the C conversion function receives that raw vector,
   calls `allocVector(INTSXP|REALSXP, n_elements)` for the typed output,
   then walks the raw bytes and writes converted values into the output.
   Allocation #2.

Peak memory ≈ raw_buffer + typed_output. For float64 those are the same
size, so peak is ~2× the payload. For narrower on-disk types widened to
R's native int/double, the typed output is actually larger, but the
*redundant* allocation is the raw buffer — it exists only because the
read happened in R and the conversion happens in C.

## The trick: allocate the SEXP first, read into it

The raw buffer is redundant because the conversion only reads it
sequentially, once. There's no need to materialize the whole file as
bytes before converting. Move the read into C:

```c
SEXP read_typed(SEXP path, SEXP offset, SEXP n_elements, SEXP n_bytes_on_disk) {
    SEXP out = PROTECT(allocVector(REALSXP, asInteger(n_elements)));
    double *p = REAL(out);

    FILE *f = fopen(CHAR(STRING_ELT(path, 0)), "rb");
    fseek(f, asInteger(offset), SEEK_SET);

    // Option A: on-disk layout already matches R's in-memory layout
    //          (native-endian float64) → read straight into p
    fread(p, sizeof(double), asInteger(n_elements), f);

    // Option B: needs conversion (narrower type, endian swap, etc.)
    //          → small stack/heap chunk buffer, read+convert in a loop
    //          enum { CHUNK = 64 * 1024 };
    //          uint8_t buf[CHUNK];
    //          ... read CHUNK bytes, widen/swap into p, advance ...

    fclose(f);
    UNPROTECT(1);
    return out;
}
```

Now there's exactly one full-size buffer alive: the output SEXP itself.
The chunk buffer in Option B is bounded (kilobytes), not proportional to
file size. Peak drops from ~2× payload to ~1× payload (or 1× output,
whichever is larger).

## Header parsing: where it stays

For formats with a header (`.npy`, pickle, etc.), the header is small and
typically easier to parse in R using existing libraries (`jsonlite`,
etc.). Two clean options:

- **Parse header in R, pass path + payload offset + dtype info into C.**
  The C function opens the file, seeks past the header, and reads.
  Simple, keeps the R API surface intact.
- **Parse header in C too.** Avoids the file-open-twice pattern but
  reimplements parsing. Worth it only if the header parse is trivial or
  if you're already in C for other reasons.

The first option is almost always the right call — header parsing is a
one-time cost on a few hundred bytes, and the win is entirely on the
payload read.

## Caveats

- **Conversion still costs CPU.** Skipping allocation #1 doesn't change
  conversion work for non-native dtypes (float16, endian-swapped, bit64,
  structured). It only removes the redundant byte buffer.
- **R connections vs. file paths.** If callers pass an open R
  connection rather than a path, you either have to extract the
  underlying fd (fragile, connection API isn't fully exposed) or fall
  back to the old `readBin` path for that case. Easiest: accept paths
  for the fast path, keep the connection-based path as a compatibility
  fallback.
- **Error handling.** `fread` partial reads, `fopen` failures, and
  `fseek` past EOF all need to clean up the `PROTECT`ed SEXP before
  raising via `Rf_error`. Don't leak protect counts on the error path.
- **Concurrency / GC.** Once the SEXP is allocated and `REAL(out)` /
  `INTEGER(out)` is taken, don't call any R API that could trigger GC
  before you're done writing — the pointer is stable across simple R
  calls but `allocVector`, `Rf_eval`, etc., can move things around in
  some configurations. Read into the buffer with plain C I/O, then
  return.

## When ALTREP is the better answer

If the on-disk layout *exactly* matches R's in-memory layout
(native-endian float64, contiguous, no conversion), you can skip even
the output allocation by exposing an ALTREP-backed vector whose data
pointer is an `mmap`ed view of the file. R operations read straight
from mapped pages; the OS handles paging.

Limits:
- Only native-layout dtypes — any conversion forces materialization.
- Many R operations call `DATAPTR()` and silently materialize a full
  copy. The "zero-copy" win is real but depends on which downstream
  ops the user runs.
- Finalizer needs to `munmap` when the vector is GC'd; on Windows it's
  `MapViewOfFile`/`UnmapViewOfFile`. Lifetime bookkeeping is non-trivial
  but well-trodden (`vroom`, `mmap`, `bigmemory`, `ff` do this).

For Rust-backed R packages (extendr / savvy), the in-place SEXP write
trick translates cleanly — you get `&mut [f64]` over the output buffer.
ALTREP from Rust is doable but the bindings are thinner and the
finalizer/lifetime story for Rust-owned backing storage takes care.

## Summary

- One allocation, not two: allocate the output SEXP first, read from
  disk straight into it (with a small bounded chunk buffer if you need
  conversion).
- Keep header parsing in R, pass path + offset + dtype into C.
- ALTREP + mmap is the next step up, but only buys zero-copy for
  native-layout dtypes — everything else still has to materialize.
