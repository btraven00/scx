// The raw-byte paths (HDF5 chunk fast path, NPY) treat buffers as
// little-endian values in place.
#[cfg(target_endian = "big")]
compile_error!("scx-core assumes a little-endian target");

pub mod api;
pub mod bpcells;
pub mod concat;
pub mod detect;
pub mod dtype;
pub mod error;
mod factory;
pub use factory::{open, OpenOptions};
pub mod h5;
pub mod h5_chunk;
mod h5_json;
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
#[cfg(feature = "zarr")]
pub mod zarr_ad;

#[cfg(test)]
#[path = "../tests/common/golden.rs"]
mod golden;
