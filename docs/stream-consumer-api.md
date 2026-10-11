# Streaming consumer API: gaps found by a numerical consumer

Status: **spec / draft** (2026-10-10), reviewed against the code 2026-10-11.

| # | gap | status |
|---|---|---|
| 6 | HDF5 filters | PR 1: lzf via hdf5-metno's `lzf` feature + clear error |
| 5 | Python floor | PR 2: abi3-py310 |
| 2 | scipy index dtypes | PR 3: `MatrixChunk.to_scipy()` |
| 4 | obs / var names | PR 4: `inspect(path, names=True)` |
| 1, 7 | dense output, threads | gated on the measurement plan (arms A/B first) |
| 3 | row ranges | deferred until sharding is actually needed |

## Why this exists

`pk.open_stream` was designed for per-gene reductions: gene sums, conversion. Its first numerical consumer is a streaming PCA in brrr-gve (`~/phd/brrr-gve/stream.py`), a single-cell clustering pipeline benchmarked in sc-brrr. That PCA reads X twice in row blocks:

1. **Accumulate the gene × gene Gram matrix:** densify each block, sgemm on 8 BLAS threads.
2. **Project each block** onto the top PCs.

Measured on hao2021 (2000 HVGs, ~24% non-zero, hot page cache):

| 50k cells, one full pass, 8192-row blocks | uncompressed | gzip-4 (the CELLxGENE norm) |
|---|---|---|
| anndata backed (h5py) | 0.13 s | 0.61 s |
| `pk.open_stream` 0.4.0 | 0.13 s | **0.21 s** (parallel inflate) |
| lzf | — | **fails** (see gap 6) |

**Where the consumer's time goes** around the parallel sgemm:
- a single-threaded h5py read;
- a single-threaded scipy `toarray()` densify (~0.4 s per pass at 154k, ~2 s at 1M);
- index copies.

Once the kNN stage is fixed (pynndescent), streaming PCA becomes the largest stage at 1M cells, so these serial parts are what limit multi-core use. Container format matters much less than parallel decode + densify ahead of the GEMM.

Each gap below lists: what's missing, the proposed API, semantics, and an acceptance check.

---

## 1. Dense output into a caller buffer

**Missing.** Every consumer that feeds BLAS has to densify each block: a full extra pass over the block, single-threaded in scipy, plus a fresh allocation per block.

**API.**
```python
buf = np.empty((chunk_size, n_vars), dtype=np.float32)   # reused across blocks
for chunk in pk.open_stream(path, chunk_size=8192):
    X = chunk.to_dense(out=buf[:chunk.nrows])              # returns the filled view
```
- `MatrixChunk.to_dense(out=None, *, dtype=None) -> np.ndarray`
  - `out`: C-contiguous `(nrows, n_vars)`, writable. If None, a new array is allocated.
  - `dtype`: defaults to `out.dtype`, else the chunk's dtype. Allowed: float32, float64.
- Zero-fills `out` and scatters the non-zeros in Rust, parallel over rows (rayon), with the GIL released.

- Duplicate entries are **summed** (`+=`), as scipy's `toarray()` does, or the bitwise check fails on non-canonical input. Integer `data` is cast to the output dtype.
- If gap 7's private pool exists, `to_dense` runs on it too; on rayon's global pool it would oversubscribe the consumer's BLAS threads again.

**Not planned:** `open_stream(..., dense=True)` yields dense blocks from a small ring of reused buffers. It is only safe if the docs state that a yielded block is overwritten after `prefetch` further iterations, so `to_dense(out=)` comes first.

**Acceptance.**
- Equal to `scipy.sparse.csr_matrix(...).toarray()` bitwise, for f32 and f64.
- No allocation when `out` is given.
- Faster than scipy's `toarray()` on an 8192 × 2000 block at 8 threads; set the exact target after measuring. The block is 65 MB, so zero-fill + scatter is memory-bandwidth bound and much of the win is skipping the per-block allocation and page faults, not the parallelism.

**Gate.** Build only if arm A/B of the measurement plan shows read + densify > ~30% of PCA.

## 2. scipy-compatible index dtypes

**Missing.** `indptr` is uint64 and `indices` uint32. scipy's CSR wants signed int32/int64, so every consumer that builds a `csr_matrix` copies both arrays per block (`csr_matrix` upcasts or rejects unsigned).

**API.**
- `MatrixChunk.to_scipy() -> scipy.sparse.csr_array`, built from `indices.view(np.int32)` (identical bits: n_vars < 2**31, so the view is free) and `indptr.astype(np.int32)` when nnz fits. Recent scipy unifies the two index dtypes, so an int64 indptr would upcast, and copy, `indices`.
- No `index_dtype=` option. The core, CLI, R binding and writers all use u64/u32 (offsets and column ids are never negative), and the views make a second dtype unnecessary.

**Acceptance.**
- `chunk.to_scipy()` shares memory with `chunk.data` and `chunk.indices` (`np.shares_memory`). `indptr` is exempt: it is cast to int32, a copy of only nrows + 1 entries.
- `csr_array.has_canonical_format` is preserved.
- scipy is an optional import: error only if `to_scipy` is called without scipy installed.

## 3. Row ranges

**Missing.** Consumers can only stream the whole matrix from row 0. They need:
- a warm-up on the first N rows (brrr runs the whole chain once on N cells before the timed replicates);
- sharding rows across processes.

**Deferred.** The warm-up needs no API: stop iterating after N rows (the reader thread wastes at most the 8 chunks it read ahead). Sharding has no consumer yet. When it does:

**API.** `open_stream(path, *, rows=None, ...)`, where `rows` is `(start, stop)`, half-open.
- Chunks keep **absolute** `row_offset`. The first chunk may be short, so it starts at `start`.
- Reads only the HDF5 chunks that overlap the range, using `indptr` to find the data slice. The CSR readers already slice by `indptr`, so this comes free.
- Implementation: `rows: Option<Range<usize>>` in `OpenOptions`, and `stream::row_chunks` takes a row range instead of `n_rows` (14 call sites, mechanical). Readers that don't go through `row_chunks` (MTX, Parquet) reject `rows`.

**Acceptance.** Concatenating `open_stream(rows=(a, b))` equals rows a:b of the full read. It touches ≤ the chunks overlapping `[indptr[a], indptr[b])` (check via HDF5 read counters or timing on a large file).

## 4. obs / var names without the matrix

**Missing.** The consumer must write per-cell outputs keyed by cell id. Today it opens the file with anndata just for `obs_names`, which ties it to h5ad. BPCells, Zarr and 10x inputs then can't be streamed end to end.

**API.**
- `pk.inspect(path, names=True)` adds `obs_names` / `var_names` (lists of str) to the dict. No new public functions.
- Never reads X. It does read every obs/var column, since `DatasetReader::obs()` has no index-only path; add one only if that measures slow (it fits the planned Arrow-native obs reader).

**Acceptance.** Equal to `anndata.read_h5ad(path, backed="r").obs_names` for h5ad. Works on each format `open_stream` supports. No matrix read.

## 5. Python floor (3.14)

**Status quo.** `requires-python >= 3.14`, with an `abi3-py314` wheel. Consumers in the scientific stack are pinned lower: brrr-gve is on 3.12 because conda-forge's faiss builds lag. So picklerick can't be installed next to them.

**Proposal.** Build `abi3-py310`. Nothing needs 3.14: commit f9f4981 chose py314 only so one wheel covers 3.14+, and pyo3/numpy 0.29 support 3.10.

**Acceptance.** The wheel installs and the test suite passes on 3.12 and 3.14.

## 6. HDF5 filters other than deflate (bug + gap)

**Bug.** Any dataset with a filter not handled by the parallel path falls back to libhdf5. libhdf5 then tries to load a dynamic filter plugin from the **compiled-in plugin directory of the CI build**:

```
PickleRickError: HDF5 error: H5Dread(): can't synchronously read data: can't open directory
(/home/runner/work/scx/scx/target/release/build/hdf5-metno-src-9fd1801934dfc48a/out/lib/plugin). Please verify its existence
```

- Repro: scx-picklerick 0.4.0 on an h5ad written with `adata.write_h5ad(f, compression="lzf")`.
- lzf is h5py's filter (id 32000): it lives in h5py, not in libhdf5. The same happens for blosc / zstd / bitshuffle written via hdf5plugin.

**Fix, in order:**
1. **Clear error.** For a filter the reader can't decode, raise `PickleRickError("dataset X/data uses HDF5 filter <name> (id N), which picklerick can't decode; supported: deflate, shuffle")`. Never leak the build path. Don't depend on the compiled-in `HDF5_PLUGIN_PATH`.
2. **lzf via libhdf5:** enable hdf5-metno's `lzf` feature. It compiles the lzf filter in (statically, so the R staticlib rule holds) and registers it with libhdf5 at init, so lzf decodes on the existing single-threaded fallback with no new code.
3. **Later, if measured:** native lzf in the parallel chunk path (`h5_chunk.rs`, next to deflate + shuffle). blosc via the `blosc*` features likewise. zstd (32015) and bitshuffle (32008) have no hdf5-metno filter, so they stay unsupported.

**Acceptance.**
- An lzf-compressed h5ad streams bitwise-equal to the uncompressed one.
- An unsupported filter gives the clear error above.
- No `/home/runner` string in any error.

## 7. Thread control (decoder vs consumer)

**Missing.** Parallel inflate (and, with gap 1, parallel densify) runs on rayon's global pool, sized to all cores. The consumer runs BLAS on 8 threads at the same time, so both oversubscribe the job's cores.
- brrr-gve measured exactly this with a Python read-ahead thread: within noise. Likely the reader competing with 8 BLAS threads.
- Under a container CPU cap, oversubscription costs more than it gains.

**API.**
- `open_stream(..., threads=None, prefetch=8)`:
  - `threads`: the size of a picklerick-private rayon pool. Default: `RAYON_NUM_THREADS` if set, else the CPUs this process may use (sched affinity / cgroup quota), not `os.cpu_count()`.
  - `prefetch`: the channel depth, currently hard-coded to 8.
- Document the recommended split for decode + BLAS consumers (e.g. 2 decode + 6 BLAS on 8 cores, measured).

**Acceptance.**
- With `threads=2` the decoder never exceeds 2 busy threads (checked with `perf` or `/proc` thread CPU).
- It respects a cgroup CPU quota.
- `prefetch` bounds peak memory to about `(prefetch + 2) × chunk bytes`: one more chunk is being decoded and one is held by the consumer.

**Notes.**
- The proposed default is rayon's existing behaviour: it honours `RAYON_NUM_THREADS` and sizes the pool with `available_parallelism`, which follows sched affinity and the cgroup v2 quota on Linux. Only an explicit `threads=` needs code: a private `ThreadPool` and running the reader thread's loop inside `pool.install`.
- Raw chunk reads stay serialised under the hdf5 global lock; `threads` only bounds inflate (and densify).
- **Gate.** brrr-gve measured oversubscription as within noise, so ship `threads` / `prefetch` only if arm D shows a gain.

---

## Measurement plan (validates the gaps)

Consumer: brrr-gve `stream.py` PCA, two passes, 8 threads, best of 3, idle machine. Inputs: hao2021 154k and a synthetic 1M, each uncompressed / gzip-4 / lzf. Arms:

| arm | read | densify |
|---|---|---|
| A (today) | anndata backed | scipy `toarray` |
| B | `open_stream` | scipy `toarray` |
| C | `open_stream` | `to_dense(out=buf)` (gap 1) |
| D | C, with `threads` split 2 / 6 and 4 / 4 (gap 7) | |

Report per pass: read, densify, sgemm, and total PCA. Gap 1 / gap 7 are worth it if read + densify is > ~30% of PCA in arm A.
