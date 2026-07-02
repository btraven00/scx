//! Unified reader factory: `open(input, opts) -> Box<dyn DatasetReader>`.
//!
//! One entry point that detects the format of a path (or, with the `net`
//! feature, a URL) and constructs the matching reader — replacing the
//! sniff-and-`match Format` blocks that were duplicated across the CLI
//! subcommands and the Python/R bindings.

use std::path::Path;

use crate::bpcells::BpcellsDatasetReader;
use crate::detect::{self, Format};
use crate::error::{Result, ScxError};
use crate::h5::ScxH5Reader;
use crate::h5ad::H5AdReader;
use crate::h5seurat::open_h5seurat;
use crate::npy::NpyIrReader;
use crate::stream::DatasetReader;
use crate::tenx::TenxH5Reader;

/// Options for [`open`]. Fields not relevant to the detected format are ignored,
/// so a caller can set only what it needs (e.g. `assay`/`layer` for H5Seurat,
/// `n_vars`/`genes` for Parquet).
#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Cells per streaming chunk.
    pub chunk_size: usize,
    /// H5Seurat assay (default "RNA" when `None`).
    pub assay: Option<String>,
    /// H5Seurat layer (default "counts" when `None`).
    pub layer: Option<String>,
    /// Open only enough to serve metadata (uses BPCells' metadata-only path).
    pub metadata_only: bool,
    /// Parquet gene axis, when no dictionary is supplied (`net` feature).
    pub n_vars: Option<usize>,
    /// Parquet gene dictionary — path or URL to `gene_metadata` (`net` feature).
    pub genes: Option<String>,
}

impl OpenOptions {
    /// Options with the given chunk size and everything else defaulted.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            assay: None,
            layer: None,
            metadata_only: false,
            n_vars: None,
            genes: None,
        }
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Detect the format of `input` and open the matching [`DatasetReader`].
///
/// `input` is a filesystem path, or — with the `net` feature — an object-store
/// URL (`s3://`, `https://`, …) which is routed to the Parquet reader.
pub async fn open(input: &str, opts: &OpenOptions) -> Result<Box<dyn DatasetReader + Send>> {
    #[cfg(feature = "net")]
    if crate::net::is_network_url(input) {
        return open_parquet(input, opts).await;
    }

    let path = Path::new(input);
    let fmt = detect::detect(path)
        .ok_or_else(|| ScxError::InvalidFormat(format!("could not detect format of '{input}'")))?;
    let cs = opts.chunk_size;

    let reader: Box<dyn DatasetReader + Send> = match fmt {
        Format::H5Ad => Box::new(H5AdReader::open(path, cs)?),
        Format::ScxH5 => Box::new(ScxH5Reader::open(path, cs)?),
        Format::TenxH5 => Box::new(TenxH5Reader::open(path, cs)?),
        Format::NpyDir => Box::new(NpyIrReader::open(path, cs)?),
        Format::BPCells if opts.metadata_only => {
            Box::new(BpcellsDatasetReader::open_metadata_only(path)?)
        }
        Format::BPCells => Box::new(BpcellsDatasetReader::open(path, cs)?),
        Format::H5Seurat => open_h5seurat(path, cs, opts.assay.as_deref(), opts.layer.as_deref())?,
        Format::Parquet => {
            #[cfg(feature = "net")]
            {
                return open_parquet(input, opts).await;
            }
            #[cfg(not(feature = "net"))]
            {
                return Err(ScxError::InvalidFormat(format!(
                    "'{input}' is Parquet, which requires building with the `net` feature"
                )));
            }
        }
        Format::PlainH5 => {
            return Err(ScxError::InvalidFormat(format!(
                "'{input}' is an unrecognized HDF5 file — use 'inspect' to explore it"
            )));
        }
    };
    Ok(reader)
}

/// Construct a Parquet reader over object storage, loading the gene dictionary
/// first when `opts.genes` is set. Both flow through the `net` transport.
#[cfg(feature = "net")]
async fn open_parquet(input: &str, opts: &OpenOptions) -> Result<Box<dyn DatasetReader + Send>> {
    use crate::net::resolve_store;
    use crate::parquet::{GeneDict, ParquetReader};

    let gene_dict = match opts.genes.as_deref() {
        Some(loc) => {
            let (store, path) = resolve_store(loc)?;
            Some(GeneDict::load(store, path, opts.chunk_size).await?)
        }
        None => None,
    };
    let (store, path) = resolve_store(input)?;
    Ok(Box::new(
        ParquetReader::open(store, path, opts.n_vars, gene_dict, opts.chunk_size).await?,
    ))
}
