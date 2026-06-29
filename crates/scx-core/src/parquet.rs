//! Network-backed Parquet reader over `object_store`.
//!
//! This is the first network reader in scx — the vertical slice that turns the
//! inert `net` feature into a real feature: `object_store` (S3/GCS/Azure/local/
//! in-memory behind one trait) → parquet's async `RecordBatch` stream → the
//! existing async [`DatasetReader`](crate::stream::DatasetReader) surface.
//!
//! ## Expected schema
//!
//! One Parquet row per cell, sparse-encoded as two parallel list columns —
//! mirroring the `expression_data` subset of the Tahoe-100M perturbation atlas:
//!
//! | column        | arrow type           | meaning                              |
//! |---------------|----------------------|--------------------------------------|
//! | `genes`       | `List<Int64>`        | column indices of non-zero entries   |
//! | `expressions` | `List<Float32>`      | values aligned with `genes`          |
//! | *(other)*     | `Utf8`/`Int64`/`f64` | optional per-cell obs columns        |
//!
//! Each row maps directly onto one CSR row of a [`MatrixChunk`](crate::ir::MatrixChunk),
//! so a Parquet `RecordBatch` of `batch_size` rows becomes one chunk.
//!
//! ## Known gaps (follow-ups)
//!
//! - `n_vars` is not derivable from the expression file alone; it is passed to
//!   [`ParquetReader::open`] (Tahoe sources it from the `gene_metadata` subset,
//!   ~62.7k genes). Token-id → dense-index remapping via that subset, and
//!   Tahoe's leading marker token, are not handled yet — `genes` is treated as
//!   direct 0-based column indices.
//! - Only `x` + scalar obs columns are read; `var`/`obsm`/`uns`/layers are empty.

mod reader;

#[cfg(test)]
mod tests;

pub use reader::ParquetReader;
