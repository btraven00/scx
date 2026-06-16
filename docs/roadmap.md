# SCX Roadmap

SCX is a lean, format-to-format interoperability engine for single-cell data,
optimized for reproducible benchmarking of conversion correctness, throughput,
and memory use.

## Status (2026-06-16) — robustness pass + 0.3.0 engine work shipped

The "make it solid" robustness pass is **complete**, and the 0.3.0 cycle's
engine work has landed on `main` (a 0.3.0 release tag is not yet cut). All of
the following are merged:

- **Error-path hardening (DONE).** Audited unwrap/expect/panic across scx-core:
  ~95% were idiomatic test-module unwraps; one real panic-on-malformed-input
  (`H5SeuratReader::x_stream` BPCells arm) fixed, plus two tidies (`encode_d1z`
  → `Result`; `merge::apply_patches` match-insert). The engine was already
  disciplined — no broad sweep needed.
- **Coverage + Codecov (DONE).** `cargo llvm-cov` runs in `ci.yml`; gating moved
  off a static `--fail-under-lines` floor to **Codecov** `project`/`patch`
  status checks (`codecov.yml`, `rust` flag, `CODECOV_TOKEN` repo secret + the
  Codecov GitHub App). `tenx.rs` 0%→84% and self-contained h5ad round-trips
  added so CI coverage no longer hinges on the gitignored golden fixtures.
- **Cut tree slop (DONE).** `scratch/` purged from history + gitignored; the
  vendored `scx-core` copy + `sync-scx-core.sh` removed in favour of a **pinned
  git dependency** for the R package (see `docs/packaging.md`).
- **Structural tidy (DONE).** The three 2k-line modules split along
  reader/writer seams into `foo.rs` thin roots + `foo/{reader,writer,…}.rs`
  submodules — h5ad (#16), h5seurat (#17), npy (#18, also `meta`/`format`). No
  module over ~1100 lines; public paths unchanged.
- **0.3.0 engine work (DONE, unreleased).** Streaming BPCells writer (peak
  O(nnz)→O(n_obs+chunk)); `scx merge` gained `obsp` + `uns` slot patches (`varp`
  rejected — not carried by the streaming pipeline); h5ad reader fixed to
  round-trip bool (unsigned u8) and u32 X (#19).

## Next up — pick one

Distribution and broader format support are the open strategic threads:

1. **0.4.0 — Network-backed readers** (Zarr/Parquet over `object_store`); lands
   the `net` feature + tokio runtime gating (`docs/tech-debt-async.md`).
2. **R-universe distribution** — now unblocked by vendored-static HDF5; PyPI is
   the sibling (both deferred per maintainer steer, but R-universe is close).
3. **Format benchmarking** — characterize zero-copy vs decode trade-offs across
   access patterns and dataset scales; tooling (`open_stream`, `.scxd`) is ready.

## 0.0.1 (done)

- `H5SeuratReader` — reads SeuratDisk `.h5seurat` files (Seurat v3/v4)
  - CSC sparse matrix streaming (CSC → CSR, cell-major chunks)
  - Cell metadata (`meta.data/`): numerics, factors, logicals
  - Feature metadata (`meta.features/`): numerics, booleans, factors
  - Dimensional reductions → `obsm` (PCA, UMAP)
- `H5AdWriter` — writes valid AnnData `.h5ad` (scanpy-compatible encoding)
- CLI (`scx convert`) — auto-detects `.h5seurat` vs `.h5` by extension
- Memory-bounded streaming: RSS independent of dataset size
- Criterion benchmarks; golden fixture tests (PBMC 3k)

---

## 0.0.2 (done)

- `H5AdReader` implementing `DatasetReader` — CSR-native streaming, encoding-type
  dispatch, obs/var dataframe recovery, obsm, uns recursive walk
- `detect::sniff()` — content-based format fingerprinting (H5AD / H5Seurat / ScxH5)
  replaces fragile extension routing in the CLI
- Cross-language round-trip test suite:
  - `pixi test` environment: R + Python isolated from the default build env
  - **Python oracle**: 25 pytest tests against `pbmc3k_reference.h5ad` (anndata)
  - **R oracle**: 19 testthat tests with `anndataR` as the reference reader
    (no SeuratDisk or zellkonverter dependency)
  - Reference fixture generated from H5Seurat via hdf5r + anndataR directly
  - All 27 Rust unit tests pass; all 25 Python tests pass; all 19 R tests pass

---

## 0.0.3 (done)

**R bindings scaffold (`picklerick`)**

- `r/picklerick/` R package (extendr scaffold)
- Phase A implemented: R functions call the `scx` CLI binary via `system2`;
  `read_h5seurat` converts to a temp H5AD then returns `anndataR::InMemoryAnnData`
- Native HDF5 global-state conflict between Rust `hdf5` crate and R's `rhdf5`
  identified and documented; Phase B (in-process) deferred

---

## 0.0.4 (done)

**Full slot parity + H5Seurat writer**

- **IR extended**: `Layers`, `Obsp`, `Varp`, `Varm` types added; `SingleCellDataset`
  updated; `DatasetReader` and `DatasetWriter` traits gain 4 reader + 4 writer methods
- **H5AdReader**: reads `layers/`, `obsp/`, `varp/` (dict-of-CSR) and `varm/`
  (dict-of-dense) — graceful empty default when groups absent
- **H5AdWriter**: writes all four slots with correct AnnData encoding attrs
- **H5SeuratReader**: reads extra assay layers, `graphs/` → obsp,
  `reductions/*/feature.loadings` → varm
- **H5SeuratWriter**: implemented; `write_layers`, `write_obsp`, `write_varm`
  are stubs (H5Seurat write path for these slots not yet implemented)
- **ScxH5Reader**: returns empty defaults for all four new slots
- **CLI**: `convert_with_reader` wires up all four new read/write calls
- **`uns` pass-through**: H5Seurat `misc/` → `UnsTable`; H5AD `uns/` recursive walk
- **`--layer data`**: routes H5Seurat reader to `assays/RNA/data/`
- **H5Seurat writer**: `H5SeuratWriter` implementing full `DatasetWriter`
  (CSR chunk accumulation; indptr written at `finalize`)
- **Benchmarks**:
  - Criterion micro-benchmarks updated to exercise all new slots
  - `scripts/benchmark_compare.sh` rewritten with hyperfine + GNU time;
    compares scx vs anndataR (R/Bioconductor)
  - `rproject.toml` with `rv` for reproducible R benchmark dependencies
  - **pbmc3k** (2.7k cells): scx ~0.18 s / 35–81 MB RSS vs anndataR ~2.5 s / 357 MB
  - **HLCA core** (584k cells): in progress

---

## 0.0.5 (done)

**Consolidation**

- **H5SeuratWriter slot parity**: `write_layers`, `write_obsp`, `write_varm` implemented.
  `write_layers` → `assays/{assay}/{name}/`; `write_obsp` → `graphs/{name}/`;
  `write_varm` → `reductions/{name}/feature.loadings` (transposes IR `(n_vars, k)` to
  H5Seurat `(k, n_vars)`)
- **Bug fix**: `read_varm_sync` and `read_obsm_sync` used `arr.t().to_owned()` which
  produces Fortran-layout in ndarray; fixed to `arr.t().as_standard_layout().into_owned()`
- **Slot parity test**: `test_slot_parity_roundtrip` — synthetic 3×4 dataset with
  all four new slots populated, full write → read roundtrip verified
- **ScxH5Writer**: pending
- **Atlas-scale benchmark results**: pending

---

## 0.0.6 (done)

**R bindings Phase B + feature parity with anndataR**

### HDF5 global-state conflict — resolution

Static HDF5 (`hdf5-sys features=["static","zlib"]`) was fully explored and hit
a hard dead end: `libz-sys ≥ 1.1` builds zlib-ng with `-DZ_SOLO`, which
requires explicit non-NULL `zalloc`/`zfree`. HDF5's `H5Z_filter_deflate`
always passes NULL — incompatible. Three workarounds all failed:

| Attempt | Why it failed |
|---|---|
| `-Wl,-Bsymbolic-functions` in `PKG_LIBS` | Made zlib symbols local, but the bundled copy IS the Z_SOLO zlib-ng — same error |
| `LIBZ_SYS_STATIC=0` | `hdf5-sys` build.rs unconditionally emits `rustc-link-lib=static=z`; no `libz.a` provided → link failure |
| `build.rs` emitting system lib search path | Cargo resolves `hdf5-sys` link deps before the crate's own `build.rs` path is in scope |

**Resolution (0.0.6):** dropped static HDF5; switched to dynamic system HDF5.
`Makevars` now uses `pkg-config --libs hdf5` (Ubuntu fallback:
`-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5`). rhdf5 + picklerick
coexistence was empirically confirmed safe for simple open-read-close
conversions. Known limitation: do not load `hdf5r` (not rhdf5) in the same
session as picklerick native mode — `hdf5r` links against a different
`libhdf5.so` build and property-list IDs may corrupt.

**Re-resolution (workspace, current):** the Z_SOLO blocker no longer applies
in the workspace. Current `hdf5-metno-src 0.10.2` pulls `libz-sys 1.1` with
`default-features = false` + `static + libc` only — no `zlib-ng` feature,
so libz-sys builds *stock* zlib that HDF5's `H5Z_filter_deflate` accepts.
The workspace (`scx-cli`, `python/picklerick`) now defaults to vendored
static HDF5 via the `vendored-hdf5` feature on `scx-core`; the `scx` and
`picklerick-python` conda recipes opt out with `--no-default-features` to use
the conda-provided `libhdf5`.

**Update (2026-06-14):** the R `picklerick` package now also builds **vendored
static** HDF5 — it *must*, because it is a Rust `staticlib` and dynamic HDF5
symbols don't survive R's separate `.so` link (`undefined symbol:
H5Sselect_elements`). Its `conda.recipe/r-picklerick` recipe therefore does not
set `HDF5_DIR` or depend on system `hdf5`. Separately, the in-tree vendored copy
of `scx-core` was replaced by a **pinned git dependency**. See
`docs/packaging.md` for both.

### What was delivered

**Dense X / dense layers (`scx-core/src/h5ad.rs`):**
- `H5AdReader` detects dense 2-D dataset vs CSR group at `open()`
  (`file.dataset("X").is_ok() && file.group("X").is_err()`)
- `ad_read_dense_chunk` + `dense_array2_to_csr` helpers; exact zeros dropped
- Same per-entry detection applied to all entries in `layers()`

**Nullable columns (`scx-core/src/h5ad.rs`):**
- `ad_read_nullable()` handles `values` + `mask` sub-dataset groups
  (anndata `IntNA` / `FloatNA` / `BoolNA`)
- Float/int NA → NaN sentinel; bool NA → `FALSE`
- Previously confused with categoricals (codes+categories), causing WARN + skip

**Boolean columns (`scx-core/src/h5ad.rs`):**
- `TypeDescriptor::Boolean` variant now handled; reads `bool` 1-D array directly
- Previously emitted `unsupported column dtype Boolean` warning and skipped

**R API:**
- `write_h5seurat(adata, path, assay, chunk_size)` — serialises via tmp h5ad →
  `scx_write_h5seurat` Rust FFI, with CLI fallback
- `scx_write_h5seurat` Rust FFI — mirrors `scx_convert` but routes to `H5SeuratWriter`
- `read_seurat(path, ...)` — `read_h5seurat()` → `adata$as_Seurat()` (requires Seurat ≥ 5)
- `read_sce(path, ...)` — `read_h5seurat()` → `adata$as_SingleCellExperiment()`
- `.native_available()` now returns `is.loaded("wrap__scx_convert", PACKAGE="picklerick")`
- 36 testthat tests (including explicit rhdf5 coexistence test)
- `README.md` + `docs/usage.md` added; license corrected to GPL-3

---

## 0.0.7 (done) — NPY-backed IR snapshots

**Goal: near-zero-overhead read/write of materialised IRs for benchmarking
and debugging.**

### Motivation

HDF5 format overhead (schema validation, chunk bookkeeping, compression) can
dominate wall time for small-to-medium datasets and makes micro-benchmarking
noisy. A raw-binary format lets you isolate reader vs writer perf and
checkpoint IRs to disk without HDF5 in the loop.

### Format: folder of `.npy` files + `meta.json`

```
ir_snapshot/
  meta.json              # schema: shape, dtype, column names, slot keys
  X_data.npy             # (nnz,) f32|f64
  X_indices.npy          # (nnz,) u32
  X_indptr.npy           # (n_obs+1,) u64
  obs_index.txt          # n_obs lines (cell barcodes)
  var_index.txt          # n_vars lines (gene names)
  obs_{col}.npy          # per-column numeric arrays
  obs_{col}_codes.npy    # categorical codes (i8|i16)
  obs_{col}_levels.txt   # categorical levels, one per line
  obsm_{key}.npy         # (n_obs, k) f64 dense matrix
  varm_{key}.npy         # (n_vars, k) f64 dense matrix
  layers_{name}_data.npy
  layers_{name}_indices.npy
  layers_{name}_indptr.npy
  obsp_{name}_data.npy
  obsp_{name}_indices.npy
  obsp_{name}_indptr.npy
  ...
```

### Implementation scope

- `NpyIrWriter` — serialise a full IR to a directory
- `NpyIrReader` — deserialise back to IR types
- Not a `DatasetReader`/`DatasetWriter` (no streaming) — these are one-shot
  materialised snapshots
- NPY header is ~128 bytes (magic, version, FORTRAN flag, dtype, shape);
  body is raw contiguous bytes. No external crate needed.
- Strings go in plain `.txt` files (one per line) — simplest possible
  encoding, readable by any language
- `meta.json` records dtypes, slot keys, and column metadata so the reader
  knows which files to expect

### Selective dumping

Not every benchmark needs every slot. The snapshot CLI and API should accept
filters so you only materialise what you need:

```bash
# X matrix + obs index only (minimal for write benchmarks)
scx snapshot pbmc.h5seurat ir_dir/ --only X,obs_index

# X + specific obs columns + obsm embeddings
scx snapshot pbmc.h5seurat ir_dir/ --only X,obs:nCount_RNA,obs:orig.ident,obsm

# Everything except layers and obsp
scx snapshot pbmc.h5seurat ir_dir/ --exclude layers,obsp
```

`--only` takes a comma-separated list of slot specifiers:
- `X` — count matrix (data, indices, indptr)
- `obs_index` / `var_index` — just the barcode/gene name arrays
- `obs` — all obs columns; `obs:col_name` — specific column
- `var` — all var columns; `var:col_name` — specific column
- `obsm` — all embeddings; `obsm:X_pca` — specific key
- `layers`, `obsp`, `varp`, `varm` — entire slot or `slot:key`

`--exclude` is the inverse. When neither is given, dump everything.

### Use cases

- **Benchmark isolation**: `scx snapshot pbmc.h5seurat ir_dir/ --only X` then
  `scx convert ir_dir/ out.h5ad` — measures H5AD write speed without
  H5Seurat read overhead or metadata noise
- **Debugging**: dump the IR to disk, inspect arrays with `numpy.load()`
- **Test fixtures**: generate synthetic IRs from Rust, verify readers in
  Python/R without HDF5

### What was delivered

- `NpyIrWriter` / `NpyIrReader` in `crates/scx-core/src/npy.rs`
- `scx snapshot` CLI subcommand with `--only` / `--exclude` filter flags
- `Format::NpyDir` in `detect::sniff()` — snapshot dirs read as conversion source

---

## 0.0.8 (done)

**Seurat v5 + BPCells reader**

### Seurat v5 HDF5 layout

Seurat v5 restructured assay storage: layers live under
`assays/RNA/layers/<name>` rather than `assays/RNA/<name>`. Detect the
version attribute on the assay group and route `H5SeuratReader` accordingly.
Lower priority — v5 files without BPCells backing can wait.

### BPCells native reader (the interesting part)

Seurat v5 adopted BPCells as its default backend for large count matrices,
so most large v5 `.h5seurat` files in the wild have BPCells-backed X.
A native Rust reader unlocks those files without launching R.

#### References

- **Format spec**: https://bnprks.github.io/BPCells/articles/web-only/bitpacking-format.html
- **Reference bitpacking implementation** (plain C++, no SIMD):
  https://github.com/GreenleafLab/BPCells_paper/blob/main/utils/bitpacking-reference-implementation.cpp
- **BPCells repo**: https://github.com/bnprks/BPCells
- **R/C++ glue** (`bitpacking_io.cpp`):
  `r/src/bitpacking_io.cpp` — Rcpp wrappers that expose the read/write primitives to R; good entry point for understanding the API surface
- **Core C++ reader** (`StoredMatrix.h`):
  `r/src/bpcells-cpp/matrixIterators/StoredMatrix.h` — shows exactly which files are opened and how packed vs unpacked paths diverge
- **Binary I/O** (`binaryfile.h`):
  `r/src/bpcells-cpp/arrayIO/binaryfile.h` — 8-byte ASCII header format (`UINT32v1`, `UINT64v1`, `FLOATSv1`, `DOUBLEv1`) + little-endian body
- **BP-128 SIMD** (`simd/bp128/`):
  `r/src/bpcells-cpp/simd/bp128/` — the actual SIMD packing kernels; the reference impl above is the portable scalar equivalent

#### On-disk layout (directory format, v2)

```
matrix_dir/
  version          # text: "packed-uint-matrix-v2" etc.
  storage_order    # text: "col" or "row"
  shape            # UINT32v1 header + 2× u32: [n_rows, n_cols]
  row_names        # text, one name per line
  col_names        # text, one name per line
  idxptr           # UINT64v1 header + (n_cols+1)× u64   [v1: u32]

  # --- unpacked ---
  val              # FLOATSv1/UINT32v1/DOUBLEv1 header + nnz values
  index            # UINT32v1 header + nnz u32 row indices

  # --- packed (uint only for val; index always packed) ---
  val_data         # BP-128-FOR packed u32 values
  val_idx          # u32 chunk offsets
  val_idx_offsets  # u64 overflow offsets (v2)
  index_data       # BP-128-D1Z packed row indices
  index_idx        # u32 chunk offsets
  index_idx_offsets # u64 overflow offsets (v2)
  index_starts     # u32 per-chunk starting values (for D1 decode)
```

For float/double matrices, `val` is always stored uncompressed even in the
packed variant — only integer `val` gets BP-128-FOR treatment.

#### Bitpacking algorithms (summary)

- **BP-128**: pack 128 u32s using B bits each (B = bits needed for max value).
  Interleaved bit layout per Lemire & Boytsov fig 6. Stored as 4B u32 words.
  `idx[i]` points to the start of chunk `i` in `data`.
- **BP-128-FOR** (`val`): subtract 1 from each value before BP-128 (shifts
  range to 0-based so 1-valued data uses 0 bits).
- **BP-128-D1Z** (`index`): difference-encode consecutive values, then
  zigzag-encode the deltas (handles non-monotone runs), then BP-128.
  `starts[i]` = decoded value at index `128*i` (needed for independent chunk
  decoding). Row indices within a column are sorted so deltas are small and
  non-negative; zigzag rarely fires but handles edge cases.

#### What was delivered

1. **Compatibility test suite** — R fixture generator + Rust unit tests for
   each codec + integration tests against known matrices.
2. **Rust BP-128 decoder** — scalar (no SIMD) decode for all three codec
   variants (`packed-uint`, `packed-float`, `packed-double`), Rayon-parallelised
   above 256-chunk threshold. Validated against fixtures.
3. **`BpcellsDirReader` / `BpcellsDatasetReader`** — directory + HDF5 backends;
   both implement `DatasetReader` and yield streaming CSR chunks.
4. **`H5SeuratReader` v5 routing** via `open_h5seurat` — probes candidate group
   paths (`assays/{assay}/{layer}` then `assays/{assay}/layers/{layer}`);
   dispatches to `BpcellsDatasetReader` when `version` attribute is present.
5. **Scalar version attribute fix** — BPCells R writes `version` as
   `H5S_SCALAR` (not a 1-D array); `read_version_attr` now tries scalar reads
   first, falling back to `read_1d` for other writers.

#### BPCells benchmark fixture

Generate a large BPCells-backed `.h5seurat` fixture (Seurat v5) for use in
`scripts/benchmark_compare.sh --large`. This exercises the
Rayon-parallelized `decode_d1z`/`decode_for` paths on a real dataset.

**Recipe (R):**

```r
library(Seurat)
library(BPCells)

# Load existing large fixture (e.g. HLCA core h5ad) into Seurat
adata <- anndataR::read_h5ad("tests/golden/hlca_core.h5ad", as = "InMemoryAnnData")
seu   <- adata$as_Seurat()

# Convert counts layer to BPCells on-disk backing
seu[["RNA"]] <- as(seu[["RNA"]], "Assay5")
seu[["RNA"]]$counts <- as.BPCells(seu[["RNA"]]$counts,
                                   path = "tests/golden/hlca_bpcells_counts/")

# Write to h5seurat — X will be stored as BPCells packed arrays
SeuratDisk::SaveH5Seurat(seu, "tests/golden/hlca_core_bpcells.h5seurat")
```

Then add `hlca_core_bpcells.h5seurat` to the `--large` branch of
`benchmark_compare.sh` alongside the existing HLCA fixture to directly
compare BPCells decode speed (Rayon-parallel) vs plain HDF5 CSR streaming.

---

## 0.0.9 (done) — BPCells write (full format bidirectionality)

**Goal: scx can produce BPCells-packed `.h5seurat` files, completing the hub.**

Today scx reads BPCells but always writes dgCMatrix. With a writer, every
conversion becomes possible in both directions:

```
h5ad          ──→  BPCells h5seurat   (scanpy ecosystem → Seurat v5 native)
h5seurat (dgCMatrix) ──→  BPCells h5seurat   (upgrade v3/v4 files in-place)
npy snapshot  ──→  BPCells h5seurat   (benchmark / debugging round-trips)
BPCells h5seurat ──→  h5ad            (already works; read side complete)
```

No existing tool does this without loading the full matrix into R. scx's
streaming reader means it can process atlas-scale inputs with bounded RSS.

---

### Why BPCells write requires an O(nnz) buffer

The existing `H5SeuratWriter` streams without buffering because CSR(cells ×
genes) and CSC(genes × cells) share the same byte layout — the writer just
appends incoming row chunks directly to the `data` / `indices` HDF5 datasets.

BPCells breaks this: each gene-column is encoded independently by BP-128, so
the encoder needs **all entries for gene `j`** before it can pack column `j`.
Incoming chunks arrive cell-by-cell (rows), so entries for any given gene are
scattered across all chunks.

**Unavoidable consequence**: the write path must accumulate O(nnz) triples
`(gene_idx, cell_idx, value)` in RAM, sort by `(gene_idx, cell_idx)`, then
encode column-by-column.

For practical scRNA-seq datasets:

| Dataset | Cells | Genes | Density | nnz (u32+u32+f32) |
|---------|-------|-------|---------|-------------------|
| PBMC 3k | 2 700 | 32k | ~5 % | ~4 MB |
| HLCA core | 584k | 60k | ~3 % | ~3.2 GB |
| 10M cells (future) | 10M | 60k | ~3 % | ~54 GB |

For HLCA-scale a workstation with 16–32 GB RAM handles this comfortably.
The 0.1.0 streaming writer (two-pass approach, see below) removes this limit.

---

### Codec additions — `crates/scx-core/src/bpcells.rs`

#### `bits_needed(max_val: u32) -> u8`

```rust
fn bits_needed(max_val: u32) -> u8 {
    if max_val == 0 { 0 } else { (32 - max_val.leading_zeros()) as u8 }
}
```

Used by both encoders to choose the minimum bit-width `b` per 128-chunk.

#### `bp128_pack(b: u8, values: &[u32; 128]) -> Vec<u32>`

Inverse of `bp128_unpack`. Interleaved lane layout must match exactly:
- 4 SIMD lanes; lane `l` holds positions `l, l+4, l+8, …, l+124`.
- `out[word * 4 + lane] |= bit-field extracted from values[lane + 4*j]`.
- Returns `4*b` words (empty vec when `b == 0`).

Round-trip identity: `bp128_unpack(b, &bp128_pack(b, v)) == v` for all valid `b`.

#### `encode_for(values: &[u32]) -> (Vec<u32>, Vec<u32>)`

Returns `(val_data, val_idx)`.

Per 128-chunk:
1. Shift: `shifted[i] = values[i].wrapping_sub(1)` (FOR offset; inverse of the `+1` in decode).
2. `b = bits_needed(shifted.iter().max())`.
3. Pack with `bp128_pack(b, &buf)`, append words to `val_data`.
4. Append new word offset to `val_idx`.

Edge: last chunk may have fewer than 128 values — pad `buf` with zeros up to
128 before packing (zeros decode as `0 + 1 = 1`... **wrong**). Instead: pad
with the last valid shifted value so that padding decodes back to the same
value — but the caller truncates output to `count`, so padding is never
observable. Zero-padding is safe.

#### `encode_d1z(values: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>)`

Returns `(index_data, index_idx, index_starts)`.

Per 128-chunk:
1. Record `starts[k] = prev` (the cumulative prefix before this chunk).
2. For each value: `delta = value as i64 - prev as i64` (cast to i32);
   `zz = zigzag_encode(delta as i32)`; `prev = value`.
3. `b = bits_needed(zz_buf.iter().max())`.
4. Pack, append to `index_data`, update `index_idx`.

Row indices within a column are sorted (BPCells invariant), so deltas are
non-negative and zigzag rarely fires — typical `b` is 1–4 bits.

---

### CSR → CSC accumulation

```
BpcellsCscAccumulator {
    n_obs: usize,
    n_vars: usize,
    entries: Vec<(u32 col, u32 row, TypedVal)>,  // gene_idx, cell_idx, value
}
```

- `push_chunk(&MatrixChunk)`: iterate `(row_offset + row, gene_idx, val)` per
  non-zero; push each as `(gene_idx, row_offset + row, val)`.
- `into_csc(self) -> CscMatrix`: `sort_unstable_by_key(|(col, row, _)| (col, row))`;
  build `idxptr` (length `n_vars + 1`), extract sorted `row_indices` and
  `values`.

`TypedVal` is a thin enum `{ F32(f32), F64(f64), U32(u32) }` so the sort
key is always `(col, row)` regardless of value type — avoids monomorphising
the sort.

Memory peak: `n_entries * (4 + 4 + 4) = nnz * 12 bytes` for f32 data. For
HLCA: ~38 GB worst-case; typical scRNA-seq at 5 % density: ~2.8 GB. Acceptable
on any compute node; document the constraint.

---

### HDF5 write — `crates/scx-core/src/h5bpcells.rs`

Implemented. `write_bpcells_h5(...)` now writes BPCells-packed HDF5 groups for:
- primary `X` at `assays/{assay}/{layer}`
- assay layers at `assays/{assay}/{name}`
- observation graphs at `graphs/{name}`

It writes the expected BPCells datasets / attrs:
- `version`
- `storage_order`
- `shape`
- `idxptr`
- `row_names`
- `col_names`
- `index_data` / `index_idx` / `index_idx_offsets` / `index_starts`
- `val_data` / `val_idx` / `val_idx_offsets` for `u32`
- `val` for `f32` / `f64`

Datasets written (all at `{group_path}/…`):

| Dataset | Type | Notes |
|---------|------|-------|
| `version` attr | string | `"packed-uint-matrix-v2"` / `"packed-float-matrix-v2"` / `"packed-double-matrix-v2"` |
| `storage_order` | string\[1\] | `["col"]` |
| `shape` | u32\[2\] | `[n_vars, n_obs]` |
| `idxptr` | u64\[n\_obs+1\] | column (cell) pointers |
| `row_names` | str\[n\_vars\] | gene names |
| `col_names` | str\[n\_obs\] | cell barcodes |
| `index_data` | u32\[\] | D1Z-packed row indices |
| `index_idx` | u32\[n\_chunks+1\] | word offsets into `index_data` |
| `index_idx_offsets` | u64\[n\_chunks\] | all zeros (v2 overflow; not needed for <2³² words) |
| `index_starts` | u32\[n\_chunks\] | per-chunk prefix values for D1Z |
| For U32: `val_data` / `val_idx` / `val_idx_offsets` | — | FOR-packed values |
| For F32: `val` | f32\[\] | uncompressed floats (BPCells spec) |
| For F64: `val` | f64\[\] | uncompressed doubles |

Note: float/double matrices pack only indices (D1Z), not values — that is the
BPCells spec. Only integer counts get FOR packing on values.

---

### `BpcellsH5Writer` — `crates/scx-core/src/h5bpcells.rs`

Implemented.

`BpcellsH5Writer` now supports:
- `write_obs`
- `write_var`
- `write_obsm`
- `write_uns`
- `write_varm`
- `write_x_chunk`
- `begin_sparse("layers", ...)`
- `begin_sparse("obsp", ...)`
- `write_sparse_chunk`
- `end_sparse`
- `finalize`

Behavior:
- `X` is accumulated as CSR row chunks, converted to CSC, then written as BPCells.
- `layers` are accumulated and written as BPCells assay groups.
- `obsp` matrices are accumulated and written as BPCells graph groups.
- `obs`, `var`, `obsm`, `varm`, and `uns` are preserved in Seurat-compatible HDF5 layout.
- `open_h5seurat()` / `H5SeuratReader` were refactored so BPCells is now treated as an internal X/layer/graph backend rather than a separate top-level container reader.

---

### CLI — `crates/scx-cli/src/main.rs`

Implemented.

BPCells is now the default when writing `.h5seurat`, with `--dgcmatrix` as an
explicit opt-out for legacy Seurat targets:

```
scx convert input.h5ad        output.h5seurat
scx convert input.h5seurat    output.h5seurat
scx convert input.h5ad        output.h5seurat --dgcmatrix
```

Routing now uses `BpcellsH5Writer` by default for `.h5seurat` output and falls
back to `H5SeuratWriter` only when `--dgcmatrix` is set.

---

### Two-pass streaming (future — 0.1.0)

For datasets where O(nnz) RAM is impractical, a two-pass approach works when
the source is re-readable (h5ad, npy):

1. **Pass 1**: stream through source, count `nnz_per_gene[j]` → compute
   `idxptr`. Allocate output arrays of size `total_nnz`.
2. **Pass 2**: stream again; for each `(cell, gene, val)` entry, write into
   the pre-allocated position `idxptr[gene]++` (scatter write). After pass 2,
   each gene's entries are in the right position but not sorted within the
   column — however, since we process rows in cell order, entries within each
   gene column ARE sorted by cell index (row index). Sort is unnecessary.

This keeps peak RSS at O(total_nnz) but avoids the intermediate `entries` Vec
and the sort, cutting memory by ~3× and eliminating the sort cost. Implement
in 0.1.0 as part of the streaming writer refactor.

---

### Files to modify / create

| File | Change |
|------|--------|
| `crates/scx-core/src/bpcells.rs` | Add `bits_needed`, `bp128_pack`, `encode_for`, `encode_d1z` |
| `crates/scx-core/src/h5bpcells.rs` | Add `BpcellsCscAccumulator`, `write_bpcells_h5`, `BpcellsH5Writer` |
| `crates/scx-core/src/lib.rs` | Re-export new public types |
| `crates/scx-cli/src/main.rs` | Add `--dgcmatrix` opt-out flag; default h5seurat output to `BpcellsH5Writer` |

No new crate dependencies — `hdf5`, `rayon`, `ndarray` already present.

---

### Verification

Completed with:
1. **Codec round-trips**
   - `encode_for(v)` ↔ `decode_for(...)`
   - `encode_d1z(v)` ↔ `decode_d1z(...)`
   - `bp128_pack(b, v)` ↔ `bp128_unpack(b, ...)`
   - covered for boundary and large lengths

2. **Writer / reader round-trips**
   - BPCells HDF5 group write → reopen
   - BPCells-backed H5Seurat reopen through `H5SeuratReader`
   - verified preservation of:
     - `X`
     - `obs`
     - `var`
     - `obsm`
     - `varm`
     - `uns`
     - `layers`
     - `obsp`

3. **CLI integration**
   - `h5ad -> h5seurat` BPCells conversion
   - `h5seurat -> h5seurat` BPCells re-encode
   - output verified readable through `scx inspect`

Remaining future work is performance-oriented (`0.1.0` two-pass streaming), not format-completeness for the current in-memory buffered writer.

---

## 0.0.10 (done) — Python bindings (`picklerick-py`, pyo3)

**Goal: AnnData drop-in. Return a real `anndata.AnnData` object so it slots into
existing scanpy workflows with no API changes.**

```python
import picklerick as pk

adata = pk.read("pbmc.h5seurat")   # returns anndata.AnnData
pk.write(adata, "pbmc.h5ad")
pk.convert("pbmc.h5seurat", "pbmc.h5ad", chunk_size=5000)
```

**Scope choice:** keep `0.0.10` intentionally simple and eager. Mirror the R
API (`read_h5seurat`, `read_h5ad`, `read_dataset`, `write_h5seurat`,
`write_h5ad`, `convert`) and return normal `anndata.AnnData` objects. Do not
block this milestone on lazy loading, custom matrix backends, or experimental
performance paths. Internally, the implementation can follow the same staged
path as R: start with a thin CLI-backed wrapper, then optionally swap the
implementation to pyo3/native calls behind the same public API.

### What was delivered

- `python/picklerick/` — maturin-based package with pyo3 native backend
- Public API: `read`, `read_h5ad`, `read_h5seurat`, `read_dataset`, `write_h5ad`,
  `write_h5seurat`, `convert`, `inspect`
- CLI fallback when native backend unavailable
- `inspect()` returns a plain `dict` (Polars optional output deferred)
- 36+ tests across convert/read/write/native paths
- PyPI publish: pending

---

## 0.1.2 (done) — Internal NPY snapshot path for benchmarking

**Goal:** keep an internal, low-overhead checkpoint format that helps isolate
benchmark components and reduce measurement noise.

The SCX `.npy` snapshot format is an **internal and exploratory** format. It is
not a primary interoperability target and should not become a new user-facing
product surface unless it proves clearly valuable beyond benchmarking and
debugging.

### Why keep it

- **Benchmark isolation** — separate read costs from write costs in controlled
  experiments.
- **Debugging** — inspect the IR without HDF5 tooling.
- **Fixture generation** — build small targeted tests around the internal IR.
- **Lower overhead experiments** — reduce format overhead when the goal is to
  benchmark the conversion engine rather than an external container format.

### Product stance

- Keep `.npy` snapshots as an internal research aid.
- Do not treat snapshots as a new canonical exchange format.
- Do not block bindings or core conversion milestones on snapshot-specific
  reopen APIs.
- Keep the main product story centered on H5Seurat ↔ H5AD interop.

Future work here should be justified by benchmarking value, not by platform
ambition. Implemented in 0.0.7; stance codified here.

---

## 0.1.3 (done) — Inspect stats + Python/R parity

**Goal: `inspect` becomes a useful diagnostic tool across all three surfaces.**

### CLI: numeric descriptive stats

For each numeric (`int32`, `float64`) obs/var column, append a one-line
summary to the inspect output:

```
  nCount_RNA                     float64   min=201  Q1=1023  med=2105  Q3=4891  max=49832
  nFeature_RNA                   int32     min=88   Q1=542   med=921   Q3=1642  max=7219
```

Categorical/string/bool columns unchanged. Implementation: sort + index (exact
quartiles; data is already fully materialized as `Vec<f64>`/`Vec<i32>`).
No new dependencies.

R and Python deliberately omit this — they have native summary/describe tools.

### Python: `inspect()` native binding

Add `inspect(path, chunk_size=5000)` to `picklerick-py` via PyO3, returning a
plain Python `dict` matching the R binding's named list:

```python
info = pk.inspect("atlas.h5seurat")
# {'format': 'H5Seurat', 'n_obs': 1200000, 'n_vars': 33538,
#  'obs_cols': ['cell_type', 'batch', ...], 'obsm_keys': ['X_pca', 'X_umap'], ...}
```

Reuses the same `scx-core` metadata reads as the R binding. No matrix data
loaded.

### Polars: optional output from Python inspect

When `polars` is importable, `inspect(path, as_polars=True)` returns obs/var
columns as a `polars.DataFrame` (copy; our IR is not Arrow-backed). Guarded
by `try: import polars` so it never becomes a hard dependency. Not exposed in
CLI or R.

**Status: deferred.** The plain `dict` return is sufficient; Polars output can
be added when a concrete use case arises.

### What was delivered

- CLI: min/Q1/med/Q3/max stats for numeric columns; nnz/cell quartiles for X,
  layers, obsp; binary 0/1 columns show counts; BPCells-backed files flagged
- Python: `inspect()` → plain `dict` via native pyo3 binding
- R: `inspect()` → named list (added in 0.0.6 consolidation)

---

## 0.1.4 (done) — Public Rust write API (`scx_core::api::write`)

**Goal:** let other Rust crates hand `scx-core` an in-memory matrix +
obs/var metadata and get back a written file, without touching the
streaming `DatasetReader` / `DatasetWriter` traits.

Now that static HDF5 linking is solved (see 0.0.6 notes), downstream Rust
projects can depend on `scx-core` and delegate format writing.

### What was delivered

- `scx_core::api::write` module with:
  - **Eager writers**: `write_h5ad_csr`, `write_h5ad_dense`,
    `write_bpcells_h5seurat_csr`, `write_bpcells_h5seurat_dense`,
    `write_h5seurat_dgcmatrix_csr`, `write_h5seurat_dgcmatrix_dense`.
  - **Streaming builders**: `H5AdBuilder`, `BpcellsH5SeuratBuilder`,
    `H5SeuratBuilder` — `new`, `obs`, `var`, `add_obsm`, `add_varm`,
    `add_uns`, `add_layer_csr`/`add_obsp_csr` (h5ad only),
    `push_x_csr_chunk`, `finalize`.
  - `H5AdOptions` and `BpcellsOptions` for per-format knobs.
  - `ScxError` — public thiserror enum (`Io`, `Hdf5`, `WrongShape`,
    `WrongOrientation`, `NotImplemented`, `Other`).
- Inputs: `sprs::CsMatViewI<f32, u32>` for sparse, `ndarray::ArrayView2<f32>`
  for dense. CSC inputs are rejected with `WrongOrientation` (no silent
  transpose).
- All public entry points are **synchronous**. Internally they call
  `futures::executor::block_on` on the existing async writer traits — no
  tokio runtime, consistent with the 0.1.3 tokio cleanup.
- `crates/scx-core/examples/write_from_ndarray.rs` demonstrates dense →
  `.h5ad` and sparse → BPCells `.h5seurat` end-to-end.
- 8 round-trip tests cover: CSR h5ad, dense h5ad (zero-drop), builder obsm
  preservation, builder layer preservation, CSC rejection, BPCells CSR,
  BPCells dense, dgCMatrix CSR. Reads back via the existing readers.

### Out of scope (deferred)

- **Compression**: `H5AdOptions::compression` is reserved; `H5AdWriter`
  doesn't expose gzip yet.
- **`add_uns` on h5seurat builders**: present, but uns translation across
  formats is best-effort.
- **Zarr writer**: deferred to 0.3.0 alongside the network reader work.
- **Read-side facade**: not promoted in this milestone. Existing
  `H5AdReader`/`H5SeuratReader` remain the read API.

---

## 0.1.5 (done) — Python streaming iterator → numpy

**Goal: expose chunk-by-chunk matrix iteration to Python for benchmark
and analysis workflows.**

Primary use case: perturbation prediction benchmarks that need to read a
subset of cells (e.g., "all cells treated with Drug X"), run a metric, and
discard. Loading the full matrix into RAM is not feasible at atlas scale.

```python
stream = pk.open_stream("atlas.h5ad", chunk_size=5000)
for chunk in stream:
    # chunk.row_offset, chunk.nrows
    indptr, indices, data = chunk.indptr, chunk.indices, chunk.data  # numpy arrays
    # process this slice of cells, then GC
```

### What was delivered

- `pk.open_stream(path, chunk_size, assay, layer)` yields `MatrixChunk` row
  blocks; format auto-detected (`detect::sniff_dir` → `detect::sniff`), with
  H5AD, H5Seurat (dgCMatrix), ScxH5, and BPCells-dir all feeding one CSR
  `x_stream()` path. Implemented in `python/picklerick/rust/src/lib.rs`
  (`PyMatrixStream` / `PyMatrixChunk`).
- **Bounded memory.** A background reader thread feeds a bounded
  `sync_channel(8)`; `__next__` releases the GIL while waiting and drains on
  early drop. Peak RSS is governed by `chunk_size`, not dataset size.
- **Zero-copy chunks.** `indptr`/`indices`/`data` are numpy arrays that own the
  decoded Rust buffer (moved via rust-numpy `into_pyarray`), writable and valid
  for the array's lifetime — no per-chunk copy.
- **Multicore decode.** For deflate-chunked H5AD `X`, the raw gzip chunks are
  read via `H5Dread_chunk` and inflated across cores (rayon + flate2,
  `scx-core/src/h5_chunk.rs`), bypassing libhdf5's single-threaded filter
  pipeline. Other filter pipelines (shuffle, blosc) fall back to the normal read.
- **Correctness.** Native suite (`test_native.py`) plus bit-exact golden-property
  digests (`test_golden_properties.py`) that match the read against an
  oracle-derived ground truth.

R already has BPCells for lazy on-disk access; `open_stream` is the equivalent on
the Python side, with one streaming API across every supported format.

### Benchmark

Per-gene-sum workload; peak RSS / wall, chunk 5000, release build, 16 cores
(harness in `bench/python/`):

| dataset | size | eager `read_h5ad` | anndata backed | `open_stream` |
|---------|-----:|------------------:|---------------:|--------------:|
| pbmc3k  | 29 MB  | 0.15 GB / 0.42 s | 0.15 GB / 0.43 s | 0.15 GB / 0.44 s |
| norman  | 79 MB  | 0.24 GB / 1.16 s | 0.23 GB / 1.03 s | 0.11 GB / 0.42 s |
| hlca    | 5.7 GB | 18.2 GB / 50 s   | 0.93 GB / 30 s   | 0.55 GB / 17 s   |

Peak RSS is bounded by `chunk_size`, not dataset size. At larger chunk sizes
(e.g. 50000) `open_stream`'s prefetch and per-row-block inflate buffers raise
peak above the single-chunk backed reader (hlca: 3.6 GB vs 2.3 GB), so keep
`chunk_size` modest.

---

## 0.1.6 — Distribution: PyPI + R-universe

**Goal:** ship what's built to the community without requiring conda.

### PyPI (picklerick-py)

- Publish `picklerick` to PyPI via the existing maturin CI job
- Add `pip install picklerick` install path to README
- Verify the CLI-fallback path works when the native extension isn't built
  (i.e., `pip install picklerick` without extras works for pure-CLI use)

### R-universe (picklerick R package)

- Set up `btraven.r-universe.dev` (GitHub repo `btraven/universe` with
  `packages.json`)
- Add `src/install.libs.R` to picklerick — installs `libhdf5-dev` on
  R-universe's Ubuntu build runners before compilation
- Verify binary packages build for Linux and macOS on R-universe infra
- Document install one-liner in README:
  ```r
  install.packages("picklerick", repos = c("https://btraven.r-universe.dev", "https://cloud.r-project.org"))
  ```
- Note: static HDF5 is currently blocked (see 0.0.6 notes). Binary packages
  depend on dynamic libhdf5 at build time; R-universe bundles the `.so` so
  end users don't need it installed separately.

---

## 0.3.0 (done, unreleased) — Truly streaming BPCells write

**Corrected premise (2026-06-14):** the dgCMatrix writer *already streams* —
`H5SeuratWriter::write_x_chunk` appends `data`/`indices` to resizable HDF5
datasets per chunk and holds only an O(n_obs) `indptr` (CSR cells×genes and CSC
genes×cells share per-cell layout, so no transpose). The **only** O(nnz) buffer
is the BPCells writer's `BpcellsCscAccumulator`, which collects every
`(cell, gene, val)` entry and sorts. That buffer is avoidable: BPCells X is also
stored cells-as-columns and streaming readers deliver cells in order, so no
transpose is needed — the only real constraint is BP-128's 128-column run
granularity.

**Approach (chosen): streaming per-run encoder.** Buffer only the trailing
`<128`-cell partial run; encode complete 128-column runs as chunks arrive
(reusing `encode_for`/`encode_d1z` unchanged), append to resizable HDF5 datasets,
accumulate `idxptr` + per-run offsets, write those at `finalize`. No temp file,
no sort, no transpose — mirrors the dgCMatrix writer. Peak drops from O(nnz) to
O(n_obs + chunk) (the `idxptr` is the floor, same as the dgCMatrix writer
already pays). Validate with a **golden-equivalence test**: streaming output
byte-identical to the current buffered `write_bpcells_h5` across dtypes and chunk
sizes (1, <128, =128, >128, non-multiples).

(Renumbered from the old "0.1.7" — 0.2.0 already shipped. The buffered writer
handles up to ~HLCA-scale on a 16–32 GB node; streaming targets >~500M nnz.)

---

## 0.2.0 (done) — Merge / fusion

Combine multiple per-sample `.h5ad` files into a single atlas-scale file with
provenance tracking. Shipped 2026-04-28 (PR #4, squash-merged to main; tag
`v0.2.0` added 2026-05-25). See `feature_merge.md` in project memory for the
design rationale and `docs/merge.md` for end-user docs.

Delivered:
- `scx merge` CLI subcommand — create + append modes, slot patches over
  layers / obs+var columns / obsm+varm
- Per-slot provenance written to `uns["scx_provenance"]`
- `scx export` CSV / Parquet
- Integration tests, HDF5 lock serialisation in test harness

Slot follow-up (landed in the 0.3.0 cycle, 2026-06-15): `obsp` and `uns` slot
patches added to `scx merge` (`obsp/<name>` streams like a layer; `uns/<key>`
copies native scalar/dict entries). `varp` is **not** supported and is rejected
at parse time — it is not carried by the streaming reader/writer pipeline (it
exists only in the npy snapshot path), so supporting it would require extending
`DatasetReader`/`DatasetWriter` and the convert driver across all readers.

---

## Picklerick R zero-copy thread (in progress on `feat/picklerick-zerocopy`)

Not a version milestone, but a research/perf thread worth tracking separately.

Starting point: `picklerick::read_h5ad` on pbmc3k (28 MB file) goes through
the extendr/Rust bridge and uses ~622 MB peak heap when assembling a
`SingleCellExperiment`. Goal: identify where the bytes go and cut them.

What landed in this session (2026-05-25):

1. **Fixed `r/picklerick/src/Makevars` dependency tracking.** `$(STATLIB)`
   had no Rust source deps, so `cargo build` was only invoked when the
   archive didn't exist. Result: edits to `r/picklerick/src/rust/src/lib.rs`
   were silently ignored — the loaded `.so` could be arbitrarily stale (we
   caught a 7-day-old binary masking every benchmark iteration).
2. **Switched picklerick to vendored static HDF5** (the deferred follow-up
   from commit 5992b9a). HDF5 + zlib are now linked statically inside
   `libpicklerick_r.a`; no system libhdf5 at link/load time. Bypasses the
   `H5Literate` ABI drift in conda's HDF5 1.14+.
3. **R bench harness + heaptrack pipeline** under `bench/r/`. Includes
   warmup, /proc/self/status VmHWM tracking, and a working
   `heaptrack`-via-direct-R-launcher recipe (extra friction: `Rscript`'s
   exec model loses `LD_PRELOAD`; you have to call the `R` binary inside
   `lib/R/bin/exec/` with `R_HOME` set).

What we learned (the negative result that shapes the next step):

- **R's SEXP page allocator (`GetNewPage` in libR.so) dominates** picklerick
  peak heap by ~72% (451 MB out of 622 MB on pbmc3k SCE). Trimming Rust-side
  transients (e.g. the `Vec<u64> → Vec<i32> → INTSXP` two-step in the
  extendr bridge) is byte-for-byte invisible to peak memory. We made the
  change anyway (commit 2362b14) because it's slightly tidier, but it's
  a no-op for the metric.
- **ALTREP + mmap zero-copy doesn't apply to h5ad.** HDF5's chunked +
  compressed storage rules it out — you can't mmap `/X/data` and treat
  the bytes as `double*`. The h5ad equivalent of "lazy zero-copy" is
  `HDF5Array::H5SparseMatrix` + DelayedArray, which picklerick's
  `lazy = TRUE` path already uses.

Candidate next steps (none committed; pick one when picking this back up):

- **(a) Investigate why `lazy = TRUE` is *slower* than eager on norman.**
  norman (83 MB): lazy 1209 ms vs eager 1016 ms. Suspect HDF5Array's
  per-call file-open / setup tax. Likely cacheable. Small fix, clear win
  if it pans out.
- **(b) Skip intermediate SEXPs by filling a preallocated dgCMatrix from
  streaming chunks.** This is the only Rust-side change that would
  actually move peak memory: have Rust write straight into the SEXP
  buffers backing the dgCMatrix `i` / `p` / `x` slots, rather than
  materialising those buffers as standalone SEXPs first. Requires the
  dgCMatrix to be allocated R-side (knowing nnz from a metadata pass)
  and its slot pointers handed to Rust. Bigger surgery.
- **(c) Build real ALTREP+mmap for `.npy` (or the `.scxd` probe).** The
  ALTREP+mmap pattern works for native-layout payloads. Doesn't optimise
  h5ad at all but builds the zero-copy primitive on formats where it's
  structurally valid, and feeds the format-benchmarking thread above.

Branch state: `feat/picklerick-zerocopy` ready to merge or iterate from.
The Makevars fix and HDF5 vendoring should land regardless of which (if
any) of (a/b/c) gets picked next — they're independent infrastructure
wins.

---

## 0.4.0 — Network-backed readers (Zarr, Parquet, ranged HDF5)

**Goal:** read directly from object stores (S3/GCS/HTTP) without staging files
locally. Activates the async trait surface that has been structural-only since
0.0.1 — see `docs/tech-debt-async.md` for the runtime-gating plan.

### Targets

- **Zarr** via `zarrs` over `object_store`. Many small range GETs per chunk;
  `buffer_unordered(N)` pipelines reads — the canonical async-I/O win.
- **Parquet** via `parquet::ParquetRecordBatchStreamBuilder` over `object_store`.
  Async-native; footer + projected column reads avoid full-file pulls.
- **Ranged h5ad / h5seurat** over HTTP. Either `ros3` VFD (sync, painful) or
  reimplement chunked HDF5 reads over `object_store` (async, larger project).
  Defer the HDF5-over-HTTP path; Zarr/Parquet cover most cloud-native use cases.

### Tech-debt prerequisite

Land the `net` / `object-store` feature flag from `docs/tech-debt-async.md` first
so the tokio runtime is constructed only when a network reader is actually
instantiated. Local HDF5 conversion stays runtime-free.

### Merge implications

`scx merge` becomes the obvious beneficiary: wrap the per-input pipeline in
`buffer_unordered(N)` once any input is network-backed. The 0.2.0 merge loop
stays sequential; this is a ~10-line change once a network reader exists.

---

## 0.5.0 — TileDB-SOMA reader

**Goal:** read `SOMAExperiment` collections (CZI + TileDB's single-cell spec,
storage layer behind CELLxGENE Census).

### Why deferred to 0.4.0

SOMA is a much larger integration than Zarr/Parquet. There is no first-class
Rust SOMA crate today; `tiledb-rs` bindings are thin and not async-native.
Options:

1. **FFI to `libtiledbsoma` (C++)** — mirrors what Python/R SDKs do. Heavy
   build dependency, but tracks the official spec automatically. Likely path.
2. **Reimplement SOMA reads on `tiledb-rs`** — multi-month effort against an
   evolving spec; rejects the maintenance burden.
3. **Skip in-process; treat SOMA as an external converter source** — use
   `tiledbsoma` Python to materialize SOMA → h5ad, then feed scx. Fallback if
   neither (1) nor (2) is justified by demand.
4. **Wait for upstream Rust SOMA** — TileDB has signalled interest, nothing
   shipped. Monitor.

### Async fit

TileDB's I/O against S3/GCS is concurrent and range-based — same shape as
Zarr/Parquet. Reinforces the "keep async traits" decision from
`docs/tech-debt-async.md`. Whatever path (1)–(3) we pick, the boundary into
`scx-core` stays async.

### Scope (tentative)

- Reader only initially; writing into a SOMAExperiment is a separate project.
- Map SOMA's `obs` / `var` / `X` / `obsm` to existing IR types.
- Audience: CELLxGENE Census users, atlas-scale workflows already on TileDB.

---

## Design notes: bidirectionality

### Why H5AD reader is easy

H5AD is the most regular of the supported formats. Every group carries an
`encoding-type` attribute that declares its content type (`csr_matrix`,
`dataframe`, `categorical`, `array`, `string-array`, `dict`). Reading is a
dispatch table, not format archaeology. The CSR-native storage means streaming
reads require no transpose. The existing writer already embeds all the schema
knowledge needed for the reader.

### Why H5Seurat writer requires care

The fundamental tension: H5Seurat expects CSC (gene-major columns, one column
per cell), but data arrives from the IR as streaming CSR row chunks. These two
orientations are incompatible for true one-pass streaming.

The O(nnz)-buffer approach in 0.0.4 is the right pragmatic choice. It is not a
full-dataset materialization (no dense expansion), and nnz for real datasets is
bounded by biology (typical scRNA-seq: 10–20% non-zero, median 2k genes/cell).
At 1M cells × 20k genes × 15% density × 4 bytes = 12 GB — that is large but
manageable on a compute node, and the user can always route through H5AD
(`--format h5ad`) to avoid it.

The two-pass streaming writer in 0.1.0 removes this constraint entirely.
