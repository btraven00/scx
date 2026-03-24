# SCX Roadmap

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

## 0.0.6 — R bindings (`picklerick`) — Phase B

**Goal: make `picklerick` usable in real Seurat workflows.**

### Current state (Phase A, from 0.0.3)

`r/picklerick/` is a working R package (extendr scaffold + CLI shim):

- `convert()` dispatches to `system2("scx", ...)` via `.convert_via_cli()`
- `.native_available()` always returns `FALSE` — the extendr Rust FFI
  (`scx_convert` in `src/rust/src/lib.rs`) compiles but is gated off because
  of the HDF5 global-state conflict
- `read_h5seurat()` converts to a temp H5AD, then `anndataR::read_h5ad()`
- `Makevars` links `libpicklerick_r.a` + `-lhdf5`; `entrypoint.c` is the
  extendr C shim
- Binary discovery: `options(picklerick.scx_binary)` → `PICKLERICK_SCX` env
  var → `Sys.which("scx")`

### Known issues

1. **HDF5 global-state conflict** — the Rust `hdf5` crate and R's `rhdf5` both
   call `H5open()` / `H5close()` on the same `libhdf5.so`. Loading both in one
   R session causes property-list-ID corruption, segfaults, or silent data
   mangling. This is why `.native_available()` is hardcoded to `FALSE`.
2. **CLI subprocess overhead** — `system2("scx", ...)` per call is fine for
   single files but adds startup latency in loops and prevents streaming data
   back into R without a temp file round-trip.
3. ~~**Slot parity gap**~~ — fixed in 0.0.5: `do_convert()` in `lib.rs` now
   wires up `layers`, `obsp`, `varp`, `varm`; `.native_available()` changed
   from hardcoded `FALSE` to `is.loaded("wrap__scx_convert", PACKAGE="picklerick")`.
4. **No `write_h5seurat()`** — the R `write.R` delegates to
   `anndataR::write_h5ad()`, there is no H5Seurat output from R yet.

### Static-linking investigation (completed, not viable as-is)

Option 1 (static HDF5) was fully explored in session 2.  Summary of what was
tried and the dead ends hit, to avoid repeating them:

**What was done**
- Added `hdf5-sys = { version = "0.8", features = ["static", "zlib"] }` to
  `r/picklerick/src/rust/Cargo.toml`.
- `libpicklerick_r.a` built successfully; `R CMD INSTALL` succeeded;
  `.native_available()` returned `TRUE`.
- Two linker obstacles resolved along the way:
  - Removed `-lhdf5` from `PKG_LIBS` (no longer needed; static HDF5 is in .a)
  - Removed the `H5Literate` shim from `entrypoint.c` (HDF5 1.14.x exports it
    as a real symbol; the shim caused a duplicate-symbol link error)

**End-to-end test failure: `inflateInit() failed`**

When converting a gzip-compressed `.h5ad`, HDF5's deflate filter called
`inflateInit_` and it returned `Z_STREAM_ERROR (-2)`.

Root cause (confirmed by disassembly + `nm`):

`libz-sys ≥ 1.1` switched from plain zlib to **zlib-ng**.  Its build.rs
compiles zlib-ng with `-DZ_SOLO` (line 121) and `-fvisibility=hidden`.
`Z_SOLO` mode **requires** explicit non-NULL `zalloc`/`zfree` function
pointers; it returns `Z_STREAM_ERROR` immediately when they are NULL.
HDF5's `H5Z_filter_deflate` always passes `zalloc = Z_NULL, zfree = Z_NULL`,
which is valid for standard zlib but illegal for Z_SOLO.

**Fix attempts and why they failed**

| Attempt | Outcome |
|---|---|
| `-Wl,-Bsymbolic-functions` in PKG_LIBS | Zlib symbols became local (`t`), preventing symbol interposition from R's `libz.so.1.3`, but `inflateInit_` still returned -2 because the local copy IS the Z_SOLO zlib-ng. |
| `LIBZ_SYS_STATIC=0` env var so libz-sys uses system zlib | `hdf5-sys` build.rs unconditionally emits `cargo:rustc-link-lib=static=z`. When libz-sys doesn't provide a `libz.a`, that directive fails with "could not find native static library `z`". |
| Add `build.rs` emitting `cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu` | The system `libz.a` (which has proper zcalloc/zcfree) is there, but cargo checks hdf5-sys's link requirements before our build.rs path is in scope; the build still fails. |

**Remaining options (not yet tried)**

2. **`dlopen` isolation** — `dlopen("libpicklerick_r.so", RTLD_LOCAL)` so
   the Rust HDF5 symbols don't collide with R's. Needs a wrapper or extendr
   change.
3. **Out-of-process worker** — keep scx in a separate process but
   communicate via a Unix socket / named pipe instead of `system2`. Avoids
   the temp-file round-trip and the HDF5 conflict entirely. More plumbing.
4. **Accept the conflict (lowest effort)** — use dynamic system HDF5, flip
   `.native_available()` to `TRUE`. If R's `rhdf5`/`hdf5r` are not loaded in
   the same session the Rust HDF5 is fine. Document the restriction and test
   empirically — the conflict may not manifest for simple open-read-close
   conversions. This is the next thing to try.
5. **Patch the static path** — force `hdf5-sys` to use a pre-built
   `libz.a` from the system by adding a `[patch.crates-io]` for `libz-sys`
   that pins to 1.0.x (pre-zlib-ng), or by writing a wrapper build script
   that provides a `/tmp/z/libz.a` symlink with an explicit search path set
   before hdf5-sys resolves its link deps.

### Exploration plan

- **Try option 4 first** — dynamic system HDF5, empirically test conflict:
  - Revert `Cargo.toml` to no `hdf5-sys` direct dep (scx-core already pulls it)
  - Set `PKG_LIBS = $(STATLIB) -lhdf5 -lz -ldl -lm`
  - Load `rhdf5` + `picklerick` in same R session and run `convert()`
  - If no crash: document "do not mix with hdf5r in same session" and ship
- **Harden CLI shim (near-term, no HDF5 risk)**:
  - Capture `stderr` from `system2` and surface it as R `warning()`/`stop()`
  - `--progress` flag on the CLI so long conversions show a status line
- **CRAN/r-universe packaging**: author `configure` + `configure.win`,
  `src/Makevars.in`, vignette, CI via `r-lib/actions`

### API target

```r
library(picklerick)
adata <- read_h5seurat("pbmc.h5seurat")   # anndataR::InMemoryAnnData
write_h5ad(adata, "pbmc3k.h5ad")
convert("pbmc.h5seurat", "pbmc.h5ad")
```

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

## 0.0.8 — Seurat v5

Seurat v5 restructured assay storage: layers live under
`assays/RNA/layers/<name>` rather than `assays/RNA/<name>`. BPCells-backed
matrices use a different on-disk format. Detect the version attribute on
the assay group and route accordingly.

---

## 0.0.9 — Python bindings (`picklerick-py`, pyo3)

**Goal: AnnData drop-in. Return a real `anndata.AnnData` object so it slots into
existing scanpy workflows with no API changes.**

```python
import picklerick as pk

adata = pk.read("pbmc.h5seurat")   # returns anndata.AnnData
pk.write(adata, "pbmc.h5ad")
pk.convert("pbmc.h5seurat", "pbmc.h5ad", chunk_size=5000)
```

Publish to PyPI.

---

## 0.1.0 — Streaming H5Seurat write + cloud formats

### Truly streaming H5Seurat writer

The 0.0.4 H5Seurat writer buffers O(nnz) in memory. For genuine atlas-scale
(>1B nnz), implement a two-pass streaming approach:

1. **Pass 1 (streaming)**: write a temporary H5AD (CSR, memory-bounded). Also
   accumulate a per-gene nnz count to compute CSC `indptr`.
2. **Pass 2 (streaming)**: read the temp H5AD column-by-column and write CSC directly.

This keeps peak RSS at O(chunk_size) throughout. Only necessary for datasets
where nnz > ~500M; the 0.0.4 writer handles everything smaller.

### Zarr / MuData

AnnData 0.10+ supports a Zarr backend. MuData wraps multiple AnnData objects
for multimodal experiments. Add `ZarrAdReader`/`ZarrAdWriter` so SCX can
operate directly on cloud-hosted datasets without downloading.

### Julia bindings

```julia
using Picklerick
sce = read_dataset("pbmc.h5seurat")
write_dataset(sce, "pbmc.h5ad")
```

Target integration with [Muon.jl](https://github.com/scverse/Muon.jl).

### Other potential formats

| Format | Ecosystem | Notes |
|--------|-----------|-------|
| TileDB-SOMA | CellxGene Census | Columnar, cloud-native; huge public dataset access |
| MEX (10x) | Raw output | `matrix.mtx.gz` + `barcodes.tsv` + `features.tsv` |
| H5 (10x CellRanger) | Raw output | `/matrix` group; simpler than H5Seurat |
| AnnData Zarr | scverse cloud | Same IR, different I/O backend |

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
