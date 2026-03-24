# SCX MVP Plan

## Goal

Streaming Seurat → AnnData conversion in Rust. Memory-bounded, atlas-scale (10M+ cells).

## Input Strategy

1. **Primary:** H5Seurat reader (SeuratDisk generates `.h5seurat` from R)
2. **Fallback:** Ship a minimal R script (`seurat_to_h5.R`) that exports Seurat objects to a known HDF5 schema if SeuratDisk chokes on v5
3. **Future:** Native RDS parser (no viable Rust crate exists today)

Support both Seurat v4 and v5 H5Seurat layouts.

## Output

`.h5ad` files on disk, conforming to AnnData's HDF5 encoding spec.

---

## Workspace Layout

```
scx/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── scx-core/               # IR + streaming + HDF5 reader/writer
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── ir.rs            # IR structs
│   │       ├── stream.rs        # async DatasetReader/Writer traits
│   │       ├── dtype.rs         # DataType enum
│   │       ├── error.rs
│   │       ├── h5seurat.rs      # H5Seurat reader (v4 + v5)
│   │       ├── h5ad.rs          # AnnData .h5ad writer
│   │       └── sparse.rs        # CSR/CSC helpers
│   └── scx-cli/                 # thin CLI binary
│       └── src/
│           └── main.rs
├── scripts/
│   └── seurat_to_h5.R           # fallback R exporter
├── tests/
│   └── golden/                  # PBMC 3k test fixtures
└── docs/
```

Split `scx-h5` out when a second format adapter is added. Bindings
(`picklerick-py`, `picklerick-r`) will be separate crates with their
own build toolchains (pyo3/maturin, extendr).

---

## Phases

### Phase 1 — `scx-core`: IR + Async Streaming

**`dtype.rs`**
- `DataType` enum: `F32`, `F64`, `I32`, `U32`
- `TypedVec` enum wrapping concrete `Vec<T>` per variant

**`ir.rs`**
- `SparseMatrixCSR { shape, indptr, indices, data: TypedVec }`
- `SparseMatrixCSC { shape, indptr, indices, data: TypedVec }`
- `MatrixChunk { row_offset, nrows, data: SparseMatrixCSR }`
- `ObsTable { columns: Vec<Column> }`
- `VarTable { columns: Vec<Column> }`
- `Column { name: String, data: ColumnData }`
- `ColumnData`: `Int(Vec<i32>)`, `Float(Vec<f64>)`, `String(Vec<String>)`, `Bool(Vec<bool>)`, `Categorical { codes: Vec<u32>, levels: Vec<String> }`
- `Embeddings { map: HashMap<String, DenseMatrix> }`
- `DenseMatrix { shape: (usize, usize), data: Vec<f64> }`
- `UnsTable { raw: serde_json::Value }`

**`stream.rs`**
```rust
#[async_trait]
pub trait DatasetReader: Send {
    async fn obs(&mut self) -> Result<ObsTable>;
    async fn var(&mut self) -> Result<VarTable>;
    async fn obsm(&mut self) -> Result<Embeddings>;
    async fn uns(&mut self) -> Result<UnsTable>;
    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>>;
    fn shape(&self) -> (usize, usize);
    fn dtype(&self) -> DataType;
}

#[async_trait]
pub trait DatasetWriter: Send {
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()>;
    async fn write_var(&mut self, var: &VarTable) -> Result<()>;
    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()>;
    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()>;
    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;
    async fn finalize(&mut self) -> Result<()>;
}
```

**`error.rs`**
- `ScxError` via `thiserror`: `Hdf5`, `Io`, `InvalidFormat`, `UnsupportedVersion`, `DtypeMismatch`

### Phase 2 — `scx-core`: H5Seurat Reader

**Seurat v4 H5Seurat layout:**
```
/assays/RNA/counts     -> dgCMatrix (CSC): data, indices, indptr, dims
/assays/RNA/data       -> normalized (optional, skip for MVP)
/meta.data             -> cell metadata columns
/reductions/pca/cell.embeddings  -> dense matrix
/reductions/umap/cell.embeddings -> dense matrix
```

**Seurat v5 H5Seurat layout (expected differences):**
- Assay5 / StdAssay structure — different group paths
- Possibly layers instead of slots
- Auto-detect version from HDF5 attributes or group structure

**Reader responsibilities:**
- Open HDF5, detect v4 vs v5 layout
- Stream `/assays/RNA/counts` in cell-chunks (CSC→CSR transpose per chunk)
- Read metadata and embeddings (non-streaming, these fit in memory)

### Phase 3 — `scx-core`: H5AD Writer

**AnnData .h5ad spec:**
```
/X                    -> sparse CSR (encoding-type: "csr_matrix")
  /X/data, /X/indices, /X/indptr
/obs                  -> dataframe (encoding-type: "dataframe")
  /obs/_index, /obs/column_name, ...
/var                  -> dataframe
/obsm/X_pca           -> dense array
/obsm/X_umap          -> dense array
/uns                  -> nested dict (encoding-type: "dict")
```

**Critical:** Must set `encoding-type` and `encoding-version` HDF5 attributes on every group/dataset. Without these, `scanpy.read_h5ad()` fails.

**Writer responsibilities:**
- Accumulate CSR chunks into final `/X` (streaming append to HDF5)
- Write obs/var as AnnData-spec dataframes
- Write obsm embeddings as dense arrays
- Write uns as nested HDF5 groups

### Phase 4 — `scx-cli` + Golden Tests

**CLI:**
```
scx convert input.h5seurat output.h5ad [--chunk-size 5000]
```

**Golden tests (PBMC 3k):**
1. R: `SaveH5Seurat(pbmc3k, "test.h5seurat")`
2. Rust: `scx convert test.h5seurat test.h5ad`
3. Python: `adata = sc.read_h5ad("test.h5ad")`
4. Assert: matrix values match, obs/var columns match, embeddings close (f32 tolerance)
5. Memory: peak RSS < 2× chunk memory, not dataset-proportional

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `hdf5` | HDF5 I/O |
| `tokio` | Async runtime |
| `futures` | `Stream` trait |
| `async-trait` | Async in traits |
| `thiserror` | Error types |
| `clap` | CLI args |
| `serde` + `serde_json` | UnsTable |
| `rayon` | Parallel chunk ops |
| `tracing` | Structured logging |

## Deferred

- Native RDS reader
- AnnData → Seurat
- Multi-modal (ATAC, ADT, spatial)
- SCE / loom / Parquet formats
- Cloud streaming (S3/GCS)
- picklerick-py (pyo3 bindings)
- picklerick-r (extendr bindings)
