use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScxError {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid format: {0}")]
    InvalidFormat(String),

    #[error("unsupported version: {0}")]
    UnsupportedVersion(String),

    #[error("dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },

    #[error("missing field: {0}")]
    MissingField(String),

    /// Network-backed reader failure (object_store / parquet / arrow). Carries a
    /// stringified cause so the variant stays free of the `net`-only crate types.
    #[error("network reader error: {0}")]
    Net(String),
}

pub type Result<T> = std::result::Result<T, ScxError>;
