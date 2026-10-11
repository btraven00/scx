use thiserror::Error;

/// Every scx-core error. New variants and fields may be added in minor
/// releases, so match with a wildcard arm.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ScxError {
    #[error("HDF5 error: {0}")]
    Hdf5(hdf5::Error),

    /// An HDF5 dataset compressed with a filter this build can't decode
    /// (blosc, zstd, bitshuffle, ... from h5py's hdf5plugin).
    #[error("{0}")]
    UnsupportedFilter(String),

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

    /// A matrix whose shape disagrees with the declared `(n_obs, n_vars)`.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    WrongShape {
        expected: (usize, usize),
        got: (usize, usize),
    },

    /// A CSC matrix where CSR (cells × genes) is required.
    #[error("expected CSR (cells × genes); got CSC. Call .to_csr() first.")]
    WrongOrientation,

    /// Network-backed reader failure (object_store / parquet / arrow). The
    /// source is boxed so the variant stays free of the `net`-only crate types.
    #[error("network reader error: {source}")]
    #[non_exhaustive]
    Net {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Zarr store failure at `path`. The source is boxed so the variant stays
    /// free of zarrs types (the crate is behind the `zarr` feature).
    #[error("zarr error: {path}: {source}")]
    #[non_exhaustive]
    Zarr {
        path: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl From<hdf5::Error> for ScxError {
    /// libhdf5 reports a filter it doesn't have as a failed search of its
    /// plugin directory, which is a path compiled into the build (a CI runner
    /// path in released wheels). Say what is unsupported instead.
    fn from(e: hdf5::Error) -> Self {
        let msg = e.to_string();
        if msg.contains("plugin") || msg.contains("is not registered") {
            ScxError::UnsupportedFilter(format!(
                "a dataset uses an HDF5 compression filter scx can't decode; supported: {SUPPORTED_FILTERS}"
            ))
        } else {
            ScxError::Hdf5(e)
        }
    }
}

/// The HDF5 filters scx decodes, for error messages.
pub(crate) const SUPPORTED_FILTERS: &str = "deflate (gzip), shuffle, lzf";

pub type Result<T> = std::result::Result<T, ScxError>;
