//! SeuratDisk `.h5seurat` reader and writer.
//!
//! Split into `reader` and `writer` submodules. This root re-exports the public
//! items so the external paths `crate::h5seurat::{H5SeuratReader, H5SeuratWriter,
//! open_h5seurat}` stay stable.

mod reader;
mod writer;

pub use reader::{open_h5seurat, H5SeuratReader};
pub(crate) use writer::seurat_write_meta_cols;
pub use writer::H5SeuratWriter;
pub(crate) use writer::{reductions, seurat_write_uns, VARM_KEY_ATTR};

use crate::error::Result;
use crate::ir::SparseMatrixMeta;
use crate::stream::DatasetReader;

/// Where Seurat's `counts` slot is filled from when X is not the counts.
pub enum CountsSource {
    /// The source's `counts` layer.
    Layer,
    /// AnnData's `raw.X` (same cells and genes as X).
    Raw(SparseMatrixMeta),
}

/// The `--x-slot auto` rule for H5Seurat output, shared by the CLI and the
/// language bindings so they cannot drift apart.
///
/// `Some` means X is taken to be normalised and goes to the `data` slot, with
/// `counts` filled from the returned source. `None` means X is the counts.
pub async fn auto_counts_source(
    reader: &mut dyn DatasetReader,
    layer_metas: &[SparseMatrixMeta],
) -> Result<Option<CountsSource>> {
    if layer_metas.iter().any(|m| m.name == "counts") {
        return Ok(Some(CountsSource::Layer));
    }
    Ok(reader.raw_meta().await?.map(CountsSource::Raw))
}

#[cfg(test)]
mod tests;
