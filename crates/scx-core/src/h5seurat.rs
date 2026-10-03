//! SeuratDisk `.h5seurat` reader and writer.
//!
//! Split into `reader` and `writer` submodules. This root re-exports the public
//! items so the external paths `crate::h5seurat::{H5SeuratReader, H5SeuratWriter,
//! open_h5seurat}` stay stable.

mod reader;
mod writer;

pub use reader::{open_h5seurat, H5SeuratReader};
pub(crate) use writer::seurat_write_uns;
pub use writer::H5SeuratWriter;
pub(crate) use writer::{seurat_write_meta_cols, seurat_write_strings};

#[cfg(test)]
mod tests;
