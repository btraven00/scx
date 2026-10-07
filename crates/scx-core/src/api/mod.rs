//! Public Rust API for `scx-core`.
//!
//! Lets other Rust crates hand `scx-core` in-memory matrices + obs/var
//! metadata and get back written single-cell files (`.h5ad`, BPCells
//! `.h5seurat`, legacy dgCMatrix `.h5seurat`), without touching the
//! streaming `DatasetReader`/`DatasetWriter` traits.
//!
//! See [`write`] for the available writers.

pub mod write;

pub use crate::dtype::{DataType, TypedVec};
pub use crate::ir::{Column, ColumnData, ObsTable, VarTable};

/// The crate-wide error type, re-exported here for API users.
pub use crate::error::ScxError;
