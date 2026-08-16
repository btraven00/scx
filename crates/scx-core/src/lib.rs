pub mod api;
pub mod bpcells;
pub mod detect;
pub mod dtype;
pub mod error;
mod factory;
pub use factory::{open, OpenOptions};
pub mod h5;
pub mod h5_chunk;
pub mod h5_str;
pub mod h5ad;
pub mod h5bpcells;
pub mod h5seurat;
pub mod ir;
pub mod merge;
pub mod mtx;
#[cfg(feature = "net")]
pub mod net;
pub mod npy;
#[cfg(feature = "net")]
pub mod parquet;
pub mod provenance;
pub mod sparse;
pub mod stream;
pub mod tenx;
pub mod validate;
