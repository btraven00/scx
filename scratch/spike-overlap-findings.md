# Spike: overlap open_stream decode with the consumer reduce (lever #1)

Date: 2026-05-31. Build: `maturin develop --release` (py313). Dataset:
`tests/golden/hlca_core.h5ad`, 584,944 obs, X = gzip-4 CSR f32, 1.14B nnz.
Workload: per-gene sum (`bincount(indices, weights=data.astype(f64))`).

## Goal

The release benchmark left a ~10% gap to anndata backed on h5ad, traced to a
per-gene reduction (~2.8–9 s depending on chunk count) that runs *after* each
chunk's decode rather than overlapping the next chunk. Lever #1 = make decode
and reduce concurrent so wall → ~max(decode, reduce) instead of the sum.

## What I measured (all on the release build)

Scripts (in `scratch/`):
- `spike_overlap_probe.py` — times `next()`(=recv+bytes) vs astype vs bincount.
- prefetch-during-idle one-liner — open stream, `sleep(8)` without consuming,
  then time the first 8 `next()` calls.
- `spike_overlap_prefetch.py` — plain consumption vs a Python thread that
  decodes the next chunk ahead while the main thread reduces.

Results (cs50000, 12 chunks):

| measurement | value |
|---|---|
| sum(next / recv = decode + bytes copy) | 18.94 s |
| sum(astype f64) | 3.54 s |
| sum(bincount) | 5.57 s |
| reduce total | 9.12 s |
| wall | 28.06 s |
| **overlap factor** (recv+reduce)/wall | **1.00× — fully serial** |
| prefetch after 8 s idle sleep | **0 chunks buffered** (every next() ~2.4 s) |
| Python thread-prefetch speedup | **1.01× (none)** |

## Root cause — CORRECTED after Rust-side tracing (2026-05-31, later)

Earlier this doc claimed "P1: the producer never decodes ahead." **That was
WRONG** — a misread of the idle-sleep test (next() was slow for a different
reason, see below). With `eprintln` tracing added to the producer loop and a
slow/​busy consumer:

- **Prefetch WORKS.** The producer decodes chunks back-to-back at ~2.4 s each
  (timestamps 2.35 → 11.97 s) regardless of the consumer, racing ahead into the
  `sync_channel(8)`.
- **Decode is GIL-free.** During an 8 s pure-Python busy loop (GIL held
  continuously, no `next()` calls) the producer still decoded chunks 0–2
  (timestamps < 8 s). So decode runs on its own core independent of the GIL.
- **16 cores** (`nproc`) — not core-starved.

**The real bottleneck: the per-chunk `PyBytes` copy in `chunk_to_py`.** Timing
`next()` on *pre-buffered* chunks (producer confirmed ahead, `recv` instant):
~2.2 s/chunk for ~780 MB = **~355 MB/s**. That is the three
`PyBytes::new_bound` copies (indptr/indices/data), GIL-held and serial. It is
~4× slower even than this box's own `bytes(memoryview)` (~1.5 GB/s) — the source
Vec is cache-cold (written by the producer core) and each chunk allocates three
fresh ~260 MB buffers that fault new pages. The idle-sleep test's "every next()
= 2.4 s" was THIS copy, not a missing decode.

So the no-overlap is: decode (~2.4 s/chunk, hidden, GIL-free) is fully
overlapped, but the consumer is bottlenecked on the ~2.2 s/chunk GIL-held copy,
which dominates and serialises everything. A Python prefetch thread can't help
because the copy itself is the serial cost.

Decode floor ≈ 2.4 s/chunk ≈ anndata backed (~2.56 s/chunk) → **parity on
decode**. anndata adds its `csr.sum` nearly free; open_stream adds the 2.2 s
copy. That copy IS the ~3 s end-to-end gap (hlca stream 31 s vs backed 28 s).

## UPDATE 2 (2026-05-31): zero-copy shipped; parallel decode is gated

Implemented zero-copy (rust-numpy `into_pyarray_bound`, returning real numpy
arrays instead of `PyBytes`; commit pending). Result on hlca cs50000, 6 chunks:

| | pre-zero-copy | zero-copy | anndata |
|---|--:|--:|--:|
| drain (read) | 14.85 s | **13.72 s** | 13.61 s |
| full (+reduce) | 17.67 s | **15.74 s** | 13.85 s |

- Real ~2 s win; **read/decode now at parity with anndata** (13.72 vs 13.61).
- Remaining full-pipeline gap (15.74 vs 13.85) is the **bincount reduce**
  (2.02 s) vs anndata's fused `csr.sum` (0.23 s) — the benchmark's workload,
  not picklerick's delivery. open_stream full == h5py-raw-bincount full (15.7
  vs 15.2), confirming it's the reduction, not the read.
- NOTE: the earlier "2.2 s/chunk = the PyBytes copy at 355 MB/s" was a
  mis-attribution — zero-copy left the prebuffered `next()` at ~2.17 s, so that
  number was mostly DECODE (the producer doesn't effectively prefetch during a
  pure pre-consume sleep). The copy was real but ~1-2 s total, not the bulk.

**Parallel decode is BLOCKED by the hdf5-metno global lock.** Draining two
streams concurrently vs sequentially = **1.08×** (no parallelism). libhdf5's
`H5Dread` (incl. the gzip filter) runs under a process-global lock, so
decompression serialises. anndata/h5py hit the SAME wall — that is why both
reads are ~13.6 s. To beat anndata on decode we must bypass the HDF5 filter
pipeline: `H5Dread_chunk` to read raw gzip blocks (short lock hold) + parallel
inflate (rayon + flate2/zlib-ng) outside the lock + CSR reassembly. Substantial;
reaches into scx-core (`h5ad.rs` `ad_read_chunk`). This is the only path to a
real win over anndata, and it's a genuine structural edge (anndata can't).

## Followup plan (Rust) — CORRECTED priority

Prefetch is fine; do NOT touch the producer threading. The gap is the copy and
the single-threaded decode floor.

1. **Eliminate the per-chunk copy → zero-copy (gets to PARITY).** Move the
   `MatrixChunk` Vecs into numpy via ownership transfer instead of copying into
   `PyBytes`. Two options:
   - `numpy` crate: `vec.into_pyarray_bound(py)` transfers the Rust allocation
     to a numpy array (freed via capsule), zero copy. Simplest; also upgrades
     the API to return real numpy arrays (drop the `np.frombuffer` dance in
     `_api.py`). Cost: adds the numpy Rust dep the design avoided (roadmap 0.1.5
     "design divergence"); arrays become writable (fine / nicer than read-only).
   - manual buffer protocol on a `#[pyclass]` holding the Vec (`__getbuffer__`):
     no new dep, more code, Python still `np.asarray`s it.
   Expected: consumer copy 2.2 s/chunk → ~0; wall → decode-bound ≈ 28.8 s (12ch)
   ≈ anndata 28 s. **Parity, not a win** — decode is the floor.

2. **Parallel chunk decode → BEAT anndata (the real win).** Decode is single-
   threaded at ~2.4 s/chunk and is the floor; anndata is also single-threaded.
   With 16 cores, decode several HDF5 row-blocks concurrently (rayon over
   independent gzip-compressed HDF5 chunks; read raw via `H5Dread_chunk` +
   inflate, or just run N `ad_read_chunk` calls in parallel guarding the
   hdf5-metno global lock). Even 3–4× parallel decode crushes anndata's 28 s.
   Memory stays bounded by (parallelism × chunk bytes); keep the in-flight count
   small. Requires #1 first so decoded chunks aren't re-serialised by the copy.

3. **DONE — dropped `astype(float64)`** (bincount upcasts internally; bit-
   identical, ~5% faster). Shipped in commit ab9e678.

4. **Re-measure** with `spike_overlap_probe.py` +
   `bench/python/profile_read_split.py`. Targets: after #1, hlca stream ≈ 28 s
   (parity); after #2, well under anndata's 28 s.

## Memory note

`sync_channel(8)` prefetch (once it works) multiplies peak RSS by up to 8×
chunk bytes — already the cause of the cs50000 memory crossover vs anndata
backed. Bound the prefetch by bytes, not chunk count, or keep depth small.

See also: `[[project_picklerick_release_build_bench]]`.
