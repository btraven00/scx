# SCX Roadmap

SCX is a lean, format-to-format interoperability engine for single-cell data,
optimized for reproducible benchmarking of conversion correctness, throughput,
and memory use.

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

**Resolution:** dropped static HDF5; switched to dynamic system HDF5.
`Makevars` now uses `pkg-config --libs hdf5` (Ubuntu fallback:
`-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5`). rhdf5 + picklerick
coexistence was empirically confirmed safe for simple open-read-close
conversions. Known limitation: do not load `hdf5r` (not rhdf5) in the same
session as picklerick native mode — `hdf5r` links against a different
`libhdf5.so` build and property-list IDs may corrupt.

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

## 0.0.7 — NPY-backed IR snapshots

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

## 0.0.10 — Python bindings (`picklerick-py`, pyo3)

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

Publish to PyPI.

---

## 0.1.0 — Truly streaming H5Seurat write

The 0.0.4 H5Seurat writer buffers O(nnz) in memory. For genuine atlas-scale
(>1B nnz), implement a two-pass streaming approach:

1. **Pass 1 (streaming)**: write a temporary H5AD (CSR, memory-bounded). Also
   accumulate a per-gene nnz count to compute CSC `indptr`.
2. **Pass 2 (streaming)**: read the temp H5AD column-by-column and write CSC directly.

This keeps peak RSS at O(chunk_size) throughout. Only necessary for datasets
where nnz > ~500M; the 0.0.4 writer handles everything smaller.

---

## 0.1.1 — Internal NPY snapshot path for benchmarking

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
ambition.

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
