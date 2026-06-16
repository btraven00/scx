//! AnnData `.h5ad` reader and writer.
//!
//! Split into `reader` and `writer` submodules. This root re-exports the public
//! types so the external paths `crate::h5ad::{H5AdReader, H5AdWriter,
//! WriterMode}` stay stable.

mod reader;
mod writer;

pub use reader::H5AdReader;
pub use writer::{H5AdWriter, WriterMode};

#[cfg(test)]
mod tests;
