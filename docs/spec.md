# Single-Cell Streaming Interop Engine (SCX) — Technical Design v0.1

## 1. Core value proposition

Most major modern single-cell ecosystems (R / Python / Bioconductor) are fragmented around incompatible in-memory object models and inefficient conversion pipelines.

### Problem

Existing workflows (e.g. Seurat ↔ AnnData conversions):

- load entire datasets into memory
- duplicate large sparse matrices during conversion
- lack streaming / backpressure
- tightly couple biological structure to runtime objects
- fail on large datasets (10M–100M cells) due to RAM blowup

### Core idea

Treat single-cell datasets as **streamable columnar + sparse matrix data**, not in-memory objects.

---

## 2. What this system is (and is NOT)

### It IS

- a streaming interop engine
- a format translation layer
- a memory-bounded ETL system for omics matrices
- a Rust-native data pipeline core with bindings

### It is NOT

- a reimplementation of Seurat
- a biology analysis framework
- a graph/trajectory computation engine
- a full semantic fidelity serializer

---

## 3. Design principles

- **Streaming first** — no full dataset materialization
- **Minimal IR** — only core interoperable structures
- **Metadata is secondary**
- **Derived structures excluded** (graphs, clustering, etc.)
- **Deterministic transforms**

---

## 4. System architecture

Input → Reader → IR → Writer → Output

---

## 5. Core Intermediate Representation (IR)

### 5.1 Top-level container

```rust
pub struct SingleCellDataset {
    pub x: MatrixBlock,
    pub layers: Vec<MatrixLayer>,
    pub obs: ObsTable,
    pub var: VarTable,
    pub obsm: Embeddings,
    pub uns: UnsTable,
}
```

---

### 5.2 Matrix representation

```rust
pub enum MatrixBlock {
    SparseCSR(SparseMatrixCSR),
    SparseCSC(SparseMatrixCSC),
    ChunkedStream(ChunkedMatrixStream),
}
```

---

### 5.3 Sparse matrix

```rust
pub struct SparseMatrixCSR {
    pub shape: (usize, usize),
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub data: Vec<f32>,
}
```

---

### 5.4 Streaming abstraction

```rust
pub trait MatrixStream {
    fn next_chunk(&mut self) -> Option<MatrixChunk>;
}
```

```rust
pub struct MatrixChunk {
    pub row_offset: usize,
    pub data: SparseMatrixCSR,
}
```

---

### 5.5 Metadata tables

```rust
pub struct ObsTable {
    pub columns: Vec<Column>,
}

pub struct Column {
    pub name: String,
    pub data: ColumnData,
}
```

```rust
pub enum ColumnData {
    Int(Vec<i32>),
    Float(Vec<f32>),
    String(Vec<String>),
    Bool(Vec<bool>),
}
```

---

### 5.6 Embeddings

```rust
pub struct Embeddings {
    pub map: HashMap<String, Matrix2D>,
}
```

---

### 5.7 Opaque metadata

```rust
pub struct UnsTable {
    pub raw: serde_json::Value,
}
```

---

## 6. Format adapters

```rust
pub trait DatasetReader {
    fn stream(&mut self) -> SingleCellStream;
}

pub trait DatasetWriter {
    fn write_stream(&mut self, stream: SingleCellStream);
}
```

---

## 7. Rust stack decisions

- Rust stable
- HDF5 via `hdf5` crate
- Serde for metadata only
- Rayon for parallel chunk processing

---

## 8. Binding strategy

- Phase 1: CLI only
- Phase 2: Python bindings (`pyo3`)
- Phase 3: R bindings (`extendr`)

### R output types

The R binding (`picklerick-r`) must return objects that slot into the three
major R single-cell workflows without friction. Each maps to a different
package ecosystem:

| Output type | Package | Ecosystem | When to use |
|-------------|---------|-----------|-------------|
| `SingleCellExperiment` | Bioconductor | scran, scater, DESeq2, zellkonverter users | Default; broadest compatibility |
| `anndataR::InMemoryAnnData` | anndataR (scverse) | R users who work directly with AnnData objects | When staying in the scverse R ecosystem |
| `Seurat` (BPCells-backed) | Seurat v5 + BPCells | Seurat v5 pipelines, large datasets | Large datasets; no full materialisation |

```r
library(picklerick)

# Default: SingleCellExperiment (zellkonverter replacement)
sce  <- read_dataset("pbmc.h5seurat")
sce  <- read_dataset("pbmc.h5ad")

# anndataR output: native AnnData object in R, no Python required
adata <- read_dataset("pbmc.h5ad",     output = "anndata")
adata <- read_dataset("pbmc.h5seurat", output = "anndata")
# adata is anndataR::InMemoryAnnData; slots match AnnData spec exactly

# Seurat v5 output: BPCells-backed for large datasets
seu  <- read_dataset("pbmc.h5seurat", output = "seurat", bpcells = TRUE)
```

### R ecosystem analysis: anndataR

[anndataR](https://github.com/scverse/anndataR) (scverse, Bioconductor 3.22,
August 2025) is the native R AnnData implementation that replaces
zellkonverter's Python-backed reader for pure R workflows. Key properties:

- **No Python dependency** — reads H5AD directly via `hdf5r`
- **Three backends**: `InMemoryAnnData` (standard R objects), `HDF5AnnData`
  (lazy read-on-demand via `AnnDataView`), `ReticulateAnnData` (Python escape
  hatch)
- **Sparse matrices**: `dgCMatrix` on materialisation; no mmap, no BPCells
- **Lazy access**: `HDF5AnnData` defers slot reads until forced, but each
  access goes through the HDF5 call stack — not OS mmap

**The gap anndataR doesn't close**: read-heavy workloads (PCA input, kNN
construction) still pay HDF5 I/O overhead per access. The `.scxd` mmap path
avoids this — once the file is mapped, array access is a pointer dereference
and a page fault, with no call stack.

**The integration target**: `picklerick-r` should return
`anndataR::InMemoryAnnData` when `output = "anndata"` is requested. This
means the slot mapping follows the AnnData spec directly (X, obs, var, obsm,
varm, obsp, varp, uns, layers) rather than the SCE mapping. Users who have
adopted anndataR can swap `anndataR::readH5AD()` for
`picklerick::read_dataset(..., output = "anndata")` and get the same object
type with streaming and format-agnostic input.

**BPCells gap**: neither zellkonverter nor anndataR has BPCells integration.
`picklerick-r` with `bpcells = TRUE` is a genuine differentiator — a Seurat v5
user gets a disk-backed matrix from any supported input format without an
intermediate H5Seurat step.

---

## 9. Out of scope

- Clustering / trajectories
- Normalization pipelines
- Biological semantics
- Graph *computation* (kNN construction, UMAP layout)

Note: storing and converting pre-computed graphs (obsp, varp) is in scope.
Graph *construction* from raw data is not.

---

## 10. Testing strategy

### Golden datasets

- PBMC 3k / 10k
- PBMC 68k

### Tests

- Round-trip conversions
- Golden file comparisons
- Streaming correctness
- Performance regression

---

## 11. Performance invariant

Peak memory must scale with chunk size, not dataset size.

---

## 12. MVP scope

- H5Seurat reader
- IR
- AnnData writer
- Sparse streaming pipeline
- PBMC golden test

---

## 13. Vision

A universal streaming layer for single-cell data, analogous to Arrow or ffmpeg.

---

## 14. Key architectural bet

Single-cell data reduces to:

- sparse matrix stream
- columnar metadata
- opaque auxiliary blobs

If this holds, the system scales.

---

## 15. SCX Exchange Format (`.scxd`)

A lightweight on-disk format for IR snapshots: benchmarking checkpoints,
pipeline hand-off between tools, and intermediate results passed between
language bindings without going through a full H5AD or H5Seurat round-trip.

### Design requirements

- **mmap-friendly**: every array must be a contiguous raw binary blob at a
  known offset — no decompression, no seek-heavy tree traversal.
- **Near-memcpy write speed**: writing a snapshot should cost no more than
  flushing the in-memory buffers to disk.
- **Cross-language**: readable from Rust, Python (NumPy), and R without
  non-trivial dependencies.
- **Optional mmap flag**: callers can choose between eager load (copy into
  process heap) and lazy mmap (array points into the mapped file), trading
  memory footprint against address-space pressure.

### Format: directory bundle

A `.scxd` artifact is a directory. Each array is an independent `.npy` file;
a `meta.json` at the root describes the full structure. Independent files mean
each array is independently mmap-able — no shared file offsets to coordinate.

```
pbmc3k.scxd/
  meta.json                   # top-level manifest (see below)
  obs_index.txt               # cell barcodes, one per line
  var_index.txt               # gene names, one per line
  uns.json                    # unstructured metadata (arbitrary JSON)
  X/                          # primary count matrix (CSR)
    data.npy                  # (nnz,)       f32/f64/i32/u32
    indices.npy               # (nnz,)       u32
    indptr.npy                # (n_obs+1,)   u64
  obs/                        # per-cell metadata columns
    n_genes.npy               # int column   → (n_obs,) i32
    cell_type.npy             # categorical codes → (n_obs,) i32
    cell_type_levels.txt      # one level per line
    barcode.npy               # string column → (n_obs,) i32 (indices into …)
    barcode_strings.txt       # one string per line
  var/                        # per-gene metadata (same layout as obs/)
  obsm/
    X_pca.npy                 # (n_obs, k)   f32  — dense
    X_umap.npy                # (n_obs, 2)   f32  — dense
  varm/
    PCs.npy                   # (n_vars, k)  f32  — gene loadings
  layers/
    counts/                   # same layout as X/
      data.npy
      indices.npy
      indptr.npy
  obsp/
    connectivities/           # same layout as X/
      data.npy
      indices.npy
      indptr.npy
  varp/                       # same layout as obsp/
```

### `meta.json` schema

```json
{
  "scxd_version": "0.1",
  "n_obs": 2700,
  "n_vars": 13714,
  "x": {"shape": [2700, 13714], "nnz": 2282976, "dtype": "f32"},
  "obs_index": {"n": 2700},
  "var_index": {"n": 13714},
  "obs": [
    {"name": "n_genes",   "kind": "int",         "shape": [2700]},
    {"name": "cell_type", "kind": "categorical",  "shape": [2700], "n_levels": 8},
    {"name": "barcode",   "kind": "string",       "shape": [2700]}
  ],
  "var": [
    {"name": "gene_name", "kind": "string", "shape": [13714]}
  ],
  "obsm": {
    "X_pca":  {"shape": [2700, 50],  "dtype": "f64"},
    "X_umap": {"shape": [2700, 2],   "dtype": "f64"}
  },
  "varm": {
    "PCs": {"shape": [13714, 50], "dtype": "f64"}
  },
  "layers": {
    "counts": {"shape": [2700, 13714], "nnz": 2282976, "dtype": "f32"}
  },
  "obsp": {
    "connectivities": {"shape": [2700, 2700], "nnz": 27000, "dtype": "f32"}
  },
  "varp": {},
  "uns": true
}
```

`obs` and `var` are **ordered arrays** (not objects) to preserve the column
insertion order from the source file. The `kind` field encodes storage:

| `kind`        | Files written                                    |
|---------------|--------------------------------------------------|
| `"int"`       | `{col}.npy` — `(n,)` i32                        |
| `"float"`     | `{col}.npy` — `(n,)` f32                        |
| `"bool"`      | `{col}.npy` — `(n,)` bool (1-byte)               |
| `"categorical"` | `{col}.npy` (codes, i32) + `{col}_levels.txt` |
| `"string"`    | `{col}.npy` (indices, i32) + `{col}_strings.txt` |

### `.npy` header

NumPy's format v1.0: `\x93NUMPY\x01\x00` + 2-byte header-len (LE u16) +
null-padded ASCII dict describing `descr`, `fortran_order`, `shape` + raw
C-contiguous binary data. The dict is at most 118 bytes for any array SCX
produces, so total header overhead is 128 bytes.

Writing from Rust requires no external crate — construct the 128-byte header
manually and write the raw slice. Reading requires parsing the dict, which
is a trivial sscanf-style scan.

### `.npy` dtype tags used by SCX

| IR type | `.npy` descr | Use                      |
|---------|-------------|--------------------------|
| `f32`   | `<f4`        | X data, obsm, varm       |
| `f64`   | `<f8`        | obsm/varm (high-prec)    |
| `i32`   | `<i4`        | X data (int counts), obs/var codes |
| `u32`   | `<u4`        | X indices                |
| `u64`   | `<u8`        | X indptr                 |
| `bool`  | `\|b1`       | obs/var boolean columns  |

All arrays are little-endian, C-contiguous (fortran_order: False).

### mmap semantics

The `mmap` flag is surfaced at the API level, not the format level. The format
is always the same on disk. The flag controls how the array is returned to the
caller.

**Rust** (`ScxdReader`):
```rust
pub enum LoadMode {
    Eager,          // Vec<T> — data copied into heap
    Mmap,           // Mmap<T> — read-only OS-managed mapping
}

reader.load_obsm("X_pca", LoadMode::Mmap)?  // → &[f32] backed by mmap
reader.load_obsp("connectivities", LoadMode::Mmap)?  // → SparseMatrixCSR with Mmap backing
```

**Python** (`picklerick`):
```python
snap = pk.load_snapshot("pbmc3k.scxd", mmap=True)
snap.obsm["X_pca"]          # np.memmap, shape (2700, 50), dtype float32
snap.obsp["connectivities"] # scipy.sparse.csr_matrix with memmap data/indices/indptr
```

**R** (`picklerick`):
```r
snap <- load_snapshot("pbmc3k.scxd", mmap = TRUE)
snap$obsm[["X_pca"]]         # matrix backed by mmap via bigmemory or ff
snap$obsp[["connectivities"]] # dgCMatrix (mmap only available for dense in R)
```

R's mmap support for sparse matrices is limited (no standard mmap-backed
`dgCMatrix`). When `mmap = TRUE` in R, dense arrays use `bigmemory::big.matrix`
or a raw `mmap::mmap` descriptor; sparse arrays are always eagerly loaded.

### Rust API (`scx-core`)

```rust
/// Controls which slots are written / read.
pub struct SlotFilter {
    pub only:    Option<Vec<String>>,  // None → all slots
    pub exclude: Vec<String>,          // exclusions take priority
}

impl SlotFilter {
    pub fn all() -> Self;
    pub fn from_only(slot: &str) -> Self;
    // Slot names: "X", "obs", "var", "obsm", "varm", "layers", "obsp", "varp", "uns"
    // Prefix matching: "obs" matches "obs" and "obs:cell_type"
}

/// Writes a full SingleCellDataset to a directory of .npy files + meta.json.
pub struct NpyIrWriter;

impl NpyIrWriter {
    pub fn write(root: &Path, dataset: &SingleCellDataset, filter: &SlotFilter) -> Result<()>;
}

/// Reads a .scxd directory into a SingleCellDataset.
/// Implements DatasetReader — x_stream() yields CSR chunks without re-reading disk.
pub struct NpyIrReader { /* ... */ }

impl NpyIrReader {
    pub fn open(root: &Path, chunk_size: usize) -> Result<Self>;
    pub fn into_dataset(self) -> SingleCellDataset;
}

impl DatasetReader for NpyIrReader { /* obs, var, obsm, x_stream, ... */ }
```

### CLI integration

```bash
# Save a full snapshot
scx snapshot --input pbmc.h5seurat --output pbmc.scxd

# Save only selected slots
scx snapshot --input pbmc.h5seurat --output pbmc.scxd --only X obsm

# Save everything except unstructured metadata
scx snapshot --input pbmc.h5seurat --output pbmc.scxd --exclude uns

# Load a snapshot and write to H5AD
scx convert --input pbmc.scxd --output pbmc.h5ad
```

`--only` and `--exclude` accept slot names (`X`, `obs`, `var`, `obsm`, `varm`,
`layers`, `obsp`, `varp`, `uns`) or prefixed column names (`obs:cell_type`).
Exclusions take priority over inclusions.

### Why not alternatives

| Format | mmap | Sparse | Cross-lang | Verdict |
|--------|------|--------|-----------|---------|
| `.npy` single-file | ✅ | ✗ | ✅ | Dense only; use for individual arrays |
| `.npz` | ✗ | ✗ | ✅ | Zip — no mmap |
| `safetensors` | ✅ | ✗ | ✅ (no R) | Dense only; no sparse |
| HDF5 | partial | ✅ | ✅ | Chunk overhead; seek-heavy for mmap |
| Zarr | ✅ | ✗ | ✅ | Chunk overhead; better for cloud |
| Arrow IPC | ✅ | awkward | ✅ | Column-oriented, not matrix-oriented |
| `.scxd` (this) | ✅ | ✅ | ✅ | Designed for this use case |

---

### What we gain: fast paths per language

The central claim is that `.scxd` lets each language runtime use its own most
efficient I/O path without format negotiation. The on-disk layout is fixed;
what varies is how each binding surfaces it to the caller.

#### Rust — OS mmap, zero allocation

```rust
let reader = ScxdReader::open("pbmc3k.scxd")?;
let pca = reader.load_obsm("X_pca", LoadMode::Mmap)?;
// pca.data is a &[f32] pointing into the mapped pages.
// No heap allocation. Page faults bring in only accessed rows.
```

The `.npy` header is 128 bytes. After parsing it once to get shape and dtype,
the data region starts at a known file offset. `mmap(2)` maps that region
read-only; the OS brings pages in on demand and evicts them under memory
pressure. For a 50-component PCA on 1M cells (200 MB), a downstream operation
that only touches the first 10k cells never faults in the remaining pages.

For sparse matrices (CSR), each of the three arrays (`data`, `indices`,
`indptr`) is mapped independently. The `indptr` array (8 bytes × n_obs) is
typically small enough to fault in fully; `data` and `indices` are accessed
in the pattern dictated by the algorithm, so page faults follow the actual
access pattern rather than a full prefetch.

#### Python — `np.memmap`, zero copy into NumPy/scipy

```python
snap = pk.load_snapshot("pbmc3k.scxd", mmap=True)

# Dense: np.memmap — a subclass of np.ndarray, shape and dtype already set.
# Arithmetic, slicing, and sklearn/scanpy functions accept it directly.
pca = snap.obsm["X_pca"]      # np.memmap (n_obs, 50), float32

# Sparse: scipy.sparse.csr_matrix whose .data, .indices, .indptr
# are np.memmap arrays. The csr_matrix itself is a thin Python object;
# the backing arrays are OS-managed pages.
knn = snap.obsp["connectivities"]   # csr_matrix, no heap copy
```

`np.memmap` is a strict subclass of `np.ndarray`, so any code that accepts
an ndarray (scikit-learn, scanpy, cupy for GPU transfer) accepts it without
modification. scipy's `csr_matrix` stores its three arrays by reference, so
constructing it from three `np.memmap` arrays incurs no copy.

The 128-byte `.npy` header offset is handled once at open time:

```python
offset = 128  # fixed for all SCX-produced .npy files
data = np.memmap("X/data.npy", dtype="float32", mode="r",
                 offset=offset, shape=(nnz,))
```

#### R — three modes, one API

R has no single best path for large sparse matrices. The binding exposes three
modes under a common interface; the caller picks based on downstream use:

```r
# Mode 1 (default): eager dgCMatrix — full compatibility, data on heap
snap <- load_snapshot("pbmc3k.scxd")
snap$obsp[["connectivities"]]   # dgCMatrix, works everywhere

# Mode 2: H5SparseMatrix — lazy HDF5-backed, Bioconductor-native
# (requires converting .scxd → temp .h5ad on first access; cached thereafter)
snap <- load_snapshot("pbmc3k.scxd", hdf5_backed = TRUE)
snap$obsp[["connectivities"]]   # H5SparseMatrix, works with SCE/DelayedArray

# Mode 3: BPCells — lazy disk-backed, Seurat v5-native (see below)
snap <- load_snapshot("pbmc3k.scxd", bpcells = TRUE)
snap$X                          # BPCells IterableMatrix, works with Seurat v5
```

Dense arrays (`obsm`, `varm`) support a fourth mode via the `mmap` package:
```r
snap <- load_snapshot("pbmc3k.scxd", mmap = TRUE)
snap$obsm[["X_pca"]]   # mmap::mmap descriptor, 128-byte offset into X_pca.npy
                        # no heap copy; coerce to matrix() when needed
```

---

### BPCells sidecar: the Seurat v5 fast path

[BPCells](https://bnprks.github.io/BPCells/) (Chang lab) is a Bioconductor
package adopted by Seurat v5 as its default backend for large count matrices.
It stores data in a custom bitpacked binary format optimised for small integer
counts (typical in scRNA-seq), and exposes an `IterableMatrix` interface that
supports lazy arithmetic (matrix-vector products, log-normalization, PCA) without
loading the full matrix into memory.

Seurat v5 accepts a BPCells `IterableMatrix` directly in
`CreateSeuratObject(counts = mat)`. This means a `.scxd` file can feed directly
into a Seurat v5 workflow with no intermediate H5Seurat or H5AD step.

#### Sidecar strategy: convert once, reuse forever

BPCells' on-disk format is fundamentally different from `.npy` CSR — it uses
bitpacking and a different traversal layout — so there is no zero-copy path
from `.npy` to BPCells. The binding converts on first access and caches the
result as a sidecar directory:

```
pbmc3k.scxd/          ← canonical, written by SCX (npy bundles)
  X/
    data.npy
    indices.npy
    indptr.npy
  meta.json

pbmc3k.scxd.bpcells/  ← sidecar, written by R on first bpcells=TRUE call
  X/                  ← BPCells bitpacked format for X
    ...
  layers/
    counts/
      ...
```

The sidecar is written by `BPCells::write_matrix_dir()` reading from the
`.scxd` `.npy` arrays. It is an R-specific artifact: other languages ignore it.
It can be deleted at any time to reclaim disk space; the next `bpcells=TRUE`
call regenerates it.

```r
# First call on a new machine: reads .npy, writes sidecar (~30s for 1M cells)
snap <- load_snapshot("pbmc3k.scxd", bpcells = TRUE)

# All subsequent calls: opens existing sidecar instantly
snap <- load_snapshot("pbmc3k.scxd", bpcells = TRUE)

# Hand off to Seurat v5 directly — no format conversion needed
seurat_obj <- CreateSeuratObject(counts = snap$X)
```

#### What BPCells gains over eager dgCMatrix

| Operation | eager `dgCMatrix` | BPCells `IterableMatrix` |
|-----------|-------------------|--------------------------|
| Load 1M-cell matrix into R | ~4 GB RAM | ~0 MB (lazy) |
| `colSums` | materialise first | streamed from disk |
| `log1p` transform | materialise first | lazy, fused with next op |
| PCA (via irlba) | materialise first | matrix-free (matvec only) |
| Feed to Seurat v5 | ✅ (if fits in RAM) | ✅ (native) |
| Feed to scran/scater | ✅ | requires coercion |

BPCells is the right path when the destination is Seurat v5 or when the matrix
does not fit in RAM. For Bioconductor pipelines (scran, scater, DESeq2), the
`hdf5_backed` mode or an eager `dgCMatrix` is more broadly compatible.
