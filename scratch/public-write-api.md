# Public Rust write API for `scx-core`

## Goal

Let other Rust crates hand `scx-core` an in-memory matrix + obs/var metadata
and get back a written `.h5ad` / BPCells `.h5seurat` / (later) Zarr file —
without learning the streaming `DatasetReader`/`DatasetWriter` traits, without
seeing `async fn`, and without depending on the CLI.

Caller story:

```rust
use scx_core::api::write::{write_h5ad_csr, H5AdOptions};
use scx_core::api::{ObsTable, VarTable};

let csr: sprs::CsMat<f32> = build_my_matrix();  // (n_cells, n_genes)
let obs = ObsTable { index: barcodes, columns: vec![] };
let var = VarTable { index: gene_names, columns: vec![] };
write_h5ad_csr(Path::new("out.h5ad"), csr.view(), &obs, &var, &H5AdOptions::default())?;
```

## Non-goals

- Network-backed writers (Zarr/S3) — phase placeholder only, real work in 0.3.0.
- Reading API. Read path is already usable via `H5AdReader::open()` etc.;
  we may promote a parallel read facade later, but writes are the ask now.
- Wrapping `anndata-rs` or any other crate's types. Inputs are `ndarray` +
  `sprs` (de-facto Rust standards), plus our own `ObsTable`/`VarTable`.
- Backwards-incompatible changes to internal traits. The async surface stays
  exactly as it is; the public API is a sync facade on top.

## Design summary

- New module `crates/scx-core/src/api.rs` (or `api/mod.rs` with submodules).
- `pub mod api { pub mod write; pub use {ObsTable, VarTable, ScxError, ...}; }`.
- Format-specific top-level functions for the common (eager) path.
- Per-format builder structs for streaming callers.
- All public surface is sync; internally calls `futures::executor::block_on`
  on the existing async writer methods (same pattern as the CLI and Python
  binding).
- New crate deps: `sprs = "0.11"` (CSR/CSC view types). `thiserror` already in.

## Public surface (v1)

```rust
// scx_core::api
pub use crate::ir::{ObsTable, VarTable, NamedColumn};
pub use crate::dtype::{DataType, TypedVec};

#[derive(Debug, thiserror::Error)]
pub enum ScxError {
    #[error("I/O error: {0}")]            Io(#[from] std::io::Error),
    #[error("HDF5 error: {0}")]           Hdf5(String),
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    WrongShape { expected: (usize, usize), got: (usize, usize) },
    #[error("expected CSR (cells×genes); got CSC. Call .to_csr() first.")]
    WrongOrientation,
    #[error("not implemented: {0}")]      NotImplemented(&'static str),
    #[error("{0}")]                       Other(String),
}

// scx_core::api::write
#[derive(Default, Clone)]
pub struct H5AdOptions {
    pub compression: Option<u8>,   // gzip 0..=9; None = no compression
    pub chunk_size: Option<usize>, // streaming chunk rows; default 5000
}

#[derive(Clone)]
pub struct BpcellsOptions {
    pub assay: String,   // default "RNA"
    pub layer: String,   // default "counts"
    pub chunk_size: Option<usize>,
}
impl Default for BpcellsOptions { /* assay=RNA, layer=counts */ }

// --- Eager writers ---
pub fn write_h5ad_csr(
    path: &Path,
    x: sprs::CsMatViewI<f32, u32>,  // shape (n_obs, n_vars); CSR
    obs: &ObsTable,
    var: &VarTable,
    opts: &H5AdOptions,
) -> Result<(), ScxError>;

pub fn write_h5ad_dense(
    path: &Path,
    x: &ndarray::ArrayView2<f32>,   // shape (n_obs, n_vars)
    obs: &ObsTable,
    var: &VarTable,
    opts: &H5AdOptions,
) -> Result<(), ScxError>;

pub fn write_bpcells_h5seurat_csr(
    path: &Path,
    x: sprs::CsMatViewI<f32, u32>,
    obs: &ObsTable,
    var: &VarTable,
    opts: &BpcellsOptions,
) -> Result<(), ScxError>;

pub fn write_bpcells_h5seurat_dense(
    path: &Path,
    x: &ndarray::ArrayView2<f32>,
    obs: &ObsTable,
    var: &VarTable,
    opts: &BpcellsOptions,
) -> Result<(), ScxError>;

// Legacy dgCMatrix path (no BPCells packing):
pub fn write_h5seurat_dgcmatrix_csr(...) -> Result<(), ScxError>;
pub fn write_h5seurat_dgcmatrix_dense(...) -> Result<(), ScxError>;

// --- Streaming builders ---
pub struct H5AdBuilder { /* owns H5AdWriter */ }
impl H5AdBuilder {
    pub fn new(path: &Path, n_obs: usize, n_vars: usize, opts: &H5AdOptions) -> Result<Self, ScxError>;
    pub fn obs(&mut self, obs: ObsTable) -> &mut Self;
    pub fn var(&mut self, var: VarTable) -> &mut Self;
    pub fn add_obsm(&mut self, key: &str, arr: ndarray::Array2<f64>) -> &mut Self;
    pub fn add_varm(&mut self, key: &str, arr: ndarray::Array2<f64>) -> &mut Self;
    pub fn add_layer_csr(&mut self, name: &str, x: sprs::CsMatViewI<f32, u32>) -> Result<&mut Self, ScxError>;
    pub fn add_obsp(&mut self, name: &str, x: sprs::CsMatViewI<f32, u32>) -> Result<&mut Self, ScxError>;
    pub fn push_x_csr_chunk(
        &mut self,
        row_offset: usize,
        indptr: &[u64],      // length nrows+1
        indices: &[u32],
        data: &[f32],
    ) -> Result<&mut Self, ScxError>;
    pub fn finalize(self) -> Result<(), ScxError>;
}

pub struct BpcellsH5SeuratBuilder { /* same surface as H5AdBuilder, owns BpcellsH5Writer */ }
pub struct H5SeuratBuilder        { /* same surface, owns H5SeuratWriter */ }
```

## Internals

### Eager → streaming bridge

`write_h5ad_csr` is the canonical pattern; the rest are copies:

```rust
pub fn write_h5ad_csr(path, x, obs, var, opts) -> Result<(), ScxError> {
    let (n_obs, n_vars) = (x.rows(), x.cols());
    let chunk = opts.chunk_size.unwrap_or(5000);
    let mut b = H5AdBuilder::new(path, n_obs, n_vars, opts)?;
    b.obs(obs.clone()).var(var.clone());
    for row_off in (0..n_obs).step_by(chunk) {
        let row_end = (row_off + chunk).min(n_obs);
        let (sub_indptr, sub_indices, sub_data) = slice_csr(&x, row_off, row_end);
        b.push_x_csr_chunk(row_off, &sub_indptr, &sub_indices, &sub_data)?;
    }
    b.finalize()
}
```

`slice_csr` is ~10 LOC against `sprs::CsMatViewI`: read `indptr[row_off..=row_end]`,
re-base to start at 0, slice `indices`/`data` from `indptr[row_off]` to
`indptr[row_end]`. Returns owned `Vec`s (allocation per chunk; could later
optimize to `&[u32]` slices, but allocation is dwarfed by HDF5 write cost).

### Dense → CSR conversion

Lift the existing private helper `dense_array2_to_csr` from `h5ad.rs` into
`api::write` as a free fn `pub(crate) fn dense_to_csr(...)` (zero-drop). The
dense entry points just call this once then funnel into the CSR path.

### Sync facade over async traits

The `DatasetWriter` trait methods are `async fn`. Inside builder methods
we call `futures::executor::block_on(self.inner.write_x_chunk(...))`. This is
the same pattern used in `scx-cli/src/main.rs` and `picklerick`. No tokio
runtime, no executor setup.

### Orientation check on sprs input

`sprs::CsMat::storage()` returns `CSR` or `CSC`. We accept only CSR (cells on
rows). If caller hands us CSC we return `ScxError::WrongOrientation` — never
silently transpose, because the cost is O(nnz) and the caller probably wants
to make that decision explicit. Document `.to_csr()` in the error string.

### Error mapping

`ScxError::Hdf5(String)` wraps the existing `hdf5::Error`'s `Display` —
intentionally string-typed so consumers don't transitively depend on
`hdf5-metno`. Internal callsites do `.map_err(|e| ScxError::Hdf5(e.to_string()))`.
A small `From<anyhow::Error> for ScxError` lets the wrappers stay terse.

## Phases (sequential, each lands as one focused commit)

### Phase 1 — Scaffold (no behavior change)

- Create `crates/scx-core/src/api/mod.rs` and `crates/scx-core/src/api/write.rs`.
- Define `H5AdOptions`, `BpcellsOptions`, `ScxError`.
- Re-export `ObsTable`/`VarTable`/`NamedColumn`/`DataType` from `api`.
- Add `sprs = "0.11"` to `scx-core/Cargo.toml`.
- Stub all `write_*` and builder methods with `Err(ScxError::NotImplemented("…"))`.
- One smoke test that the module compiles and the stubs return NotImplemented.

**Acceptance:** `cargo build` green; `cargo test --workspace` green; nothing
exposed beyond what tests verify.

### Phase 2 — H5AD eager + builder

- Implement `H5AdBuilder` against `H5AdWriter`. Sync facade; `block_on` internally.
- Implement `write_h5ad_csr` via the slice+push pattern.
- Implement `write_h5ad_dense` via lifted `dense_to_csr` then CSR path.
- Tests (new file `crates/scx-core/tests/api_write.rs`):
  1. CSR round-trip: synthetic 100×50 CSR → `write_h5ad_csr` → reopen with
     `H5AdReader` → assert nnz, sample values, shape.
  2. Dense round-trip: 30×20 dense (~30% non-zero) → `write_h5ad_dense` →
     reopen → assert sparse layout, values, zeros dropped.
  3. Obsm preserved: builder adds 2 obsm entries → reopen → assert presence
     and shape.
  4. Layer preserved: builder adds 1 layer (CSR) → reopen → assert nnz match.
  5. Wrong orientation: pass CSC → `ScxError::WrongOrientation`.

**Acceptance:** all 5 tests pass; existing tests still pass.

### Phase 3 — BPCells h5seurat eager + builder

- Implement `BpcellsH5SeuratBuilder` against `BpcellsH5Writer`.
- Implement `write_bpcells_h5seurat_csr` / `_dense`.
- Tests in same `api_write.rs`:
  6. CSR round-trip via BPCells: synthetic counts (u32-castable f32) → write
     → reopen via `H5SeuratReader` (BPCells path) → assert.
  7. Dense round-trip via BPCells.

**Acceptance:** tests pass.

### Phase 4 — H5Seurat dgCMatrix eager + builder

- Implement `H5SeuratBuilder` against `H5SeuratWriter`.
- `write_h5seurat_dgcmatrix_*` (legacy path; matches existing CLI `--dgcmatrix`).
- 1–2 round-trip tests.

### Phase 5 — Example + docs

- `crates/scx-core/examples/write_from_ndarray.rs` — minimal, runnable.
- Doc comment at module head (`api/mod.rs`) showing the eager and streaming
  patterns.
- Update `docs/roadmap.md` with a "0.1.4 — Public Rust write API" entry
  describing what shipped (TBD section number; user picks).

### Phase 6 (deferred to 0.3.0) — Zarr stub

- Add `write_zarr_csr` returning `ScxError::NotImplemented("zarr writer in 0.3.0")`.
- Gated behind a future `net` feature; not in v1.

## Risks / open questions

- **`sprs` index type**: `CsMatViewI<f32, u32>` uses `u32` indices. h5ad
  on-disk indptr is `i64`/`u64`; indices are `i32`/`u32`. We promote `u32`
  inputs to `u64` for indptr at write time. Callers with `usize`-indexed
  sprs matrices (`CsMatView<f32>`) need to convert — document and provide
  a free helper `fn promote_csr(view: CsMatView<f32>) -> CsMatViewI<f32, u32>`
  *if and only if* the conversion is zero-copy. (It is not, in general —
  if usize=u64 on 64-bit, we need to downcast.) Skip the helper for v1;
  document the type requirement instead.

- **Cloning obs/var in eager fns**: `b.obs(obs.clone())` copies. Could take
  `ObsTable` by value (caller transfers ownership) — preferable. Update
  signature: `obs: ObsTable, var: VarTable` (owned) rather than `&ObsTable`.
  Decision: own the metadata; reference only the matrix.

- **What to do with `uns`**: skip for v1. Add `add_uns(key, value)` to
  builders in a follow-up; the eager fns don't take uns. Document.

- **API stability commitment**: `scx-core` is `0.1.0`. We can iterate the
  public API freely during 0.x. Pin via `version = "0.1"` in downstream
  Cargo.toml and we're fine until 0.2.

## Files touched

| File | Change |
|------|--------|
| `crates/scx-core/Cargo.toml` | Add `sprs = "0.11"` |
| `crates/scx-core/src/lib.rs` | `pub mod api;` |
| `crates/scx-core/src/api/mod.rs` | New — re-exports + `ScxError` |
| `crates/scx-core/src/api/write.rs` | New — eager fns + builders |
| `crates/scx-core/src/h5ad.rs` | Lift `dense_array2_to_csr` to `pub(crate)` (or move to `api::write`) |
| `crates/scx-core/tests/api_write.rs` | New — phase 2/3/4 round-trip tests |
| `crates/scx-core/examples/write_from_ndarray.rs` | New (phase 5) |
| `docs/roadmap.md` | New milestone entry (phase 5) |
