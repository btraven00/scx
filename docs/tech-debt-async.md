# Async runtime tech debt — proximal cleanup

## Context

`scx-core`'s `Reader`/`Writer` traits and matrix-chunk streams are async (`async fn`, `futures::Stream`), but every implementation today is backed by synchronous HDF5 (`hdf5-metno`). There is no real concurrency anywhere in the workspace — no `tokio::spawn`, `spawn_blocking`, `buffer_unordered`, `join!`, or network I/O. Consumers (picklerick Python binding, `scx-cli`) build a `tokio::runtime::Builder::new_current_thread()` runtime solely to `block_on` futures that never yield.

The async surface is therefore pure overhead today: runtime setup cost, `.await` plumbing, tokio dep in the Python wheel, more complex trait signatures. **However**, the roadmap targets network-backed formats (Zarr, Parquet, possibly ranged-HTTP h5ad, BPCells over object stores) where async over `object_store` is the idiomatic and high-leverage path — request pipelining via `buffer_unordered` is a real ~10× win on cloud latency that sync code can't match without reinventing thread pools.

**Decision: keep async, stop paying for it locally.** Ripping it out now and re-adding when Zarr lands would be churn-heavy. Instead, defer the tokio runtime cost until a reader actually needs it.

## Cleanup tasks (proximal, ordered by leverage)

### 1. Drop tokio runtime from picklerick (Python) — DONE
- `block_on` in `python/picklerick/rust/src/lib.rs` now uses
  `futures::executor::block_on`; `tokio` removed from `python/picklerick/Cargo.toml`.

### 2. Drop tokio runtime from `scx-cli` and `scx-core` — DONE
- `scx-cli/src/main.rs` switched from `#[tokio::main]` to a sync `main` that
  calls `futures::executor::block_on(run())`. `tokio` removed from
  `crates/scx-cli/Cargo.toml`.
- `crates/scx-cli/tests/merge.rs` switched its per-test `tokio::runtime::Builder`
  to `futures::executor::block_on`.
- `crates/scx-core/Cargo.toml` no longer declares `tokio` as a regular or
  dev dependency. `#[tokio::test]` is no longer used; tests rely on the
  unit-test attribute pattern provided by the existing harness.
- `validate.rs` had no production `block_on`; only test code referenced tokio.

### 3. Gate tokio behind a `net` / `object-store` feature — PENDING
- No code currently depends on tokio. When the first network-backed reader
  arrives (Zarr/Parquet over `object_store`), introduce a `net` feature on
  `scx-core` that pulls in `tokio` + `object_store` + the new async readers,
  and construct the runtime at the single entry point that needs it
  (CLI subcommand or `open_stream()`).

### 4. Simplify hand-rolled `Stream` impls
- Files using `futures::stream::{self, Stream}` wrapping sync HDF5 iterators: `h5.rs`, `h5ad.rs`, `h5seurat.rs`, `tenx.rs`, `bpcells.rs`, `h5bpcells.rs`, `npy.rs`, `merge/mod.rs`.
- Replace with `stream::iter(sync_iterator)` where applicable — same semantics, less code, swaps cleanly for real async streams when network readers arrive.
- Lower priority than (1)–(3); do opportunistically when touching these files for other reasons.

### 5. Don't touch the merge loop yet
- `merge::run_*` is sequential-per-slot today. Once any input source is network-backed (0.3.0+), wrap the per-input pipeline in `buffer_unordered(N)` to overlap reads.
- Keeping the existing async traits means this is a ~10-line change later. **No work needed now.**

## When network readers land (0.3.0+ rough sketch)

- **Zarr** (`zarrs` over `object_store`): real win, fully async. `buffer_unordered` for chunk reads.
- **Parquet** (`parquet::ParquetRecordBatchStreamBuilder` over `object_store`): async-native, footer + projected column reads.
- **HTTP-ranged h5ad/h5seurat**: `ros3` VFD (sync, painful) or reimplement chunked reads over `object_store` (async, larger project). Not MVP.
- **BPCells over object store**: depends on upstream; likely sync wrapped in `spawn_blocking`.

At that point, `net` feature flips on, tokio runtime gets constructed at a single entry point (CLI command or `open_stream()` call), and the existing async traits absorb the new readers without signature changes.

## Out of scope

- Removing async from the trait surface. Confirmed wrong direction given the roadmap.
- Switching to `async-std` / `smol`. `tokio` ecosystem alignment with `object_store`, `reqwest`, `parquet`, `zarrs` is decisive.
- Restructuring merge for parallelism. Premature until a network reader exists to amortize the latency.
