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

#[derive(Debug, thiserror::Error)]
pub enum ScxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HDF5 error: {0}")]
    Hdf5(String),
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    WrongShape {
        expected: (usize, usize),
        got: (usize, usize),
    },
    #[error("expected CSR (cells × genes); got CSC. Call .to_csr() first.")]
    WrongOrientation,
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("{0}")]
    Other(String),
}
