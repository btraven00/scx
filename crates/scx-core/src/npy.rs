//! NPY-backed IR snapshots.
//!
//! A snapshot is a directory of raw binary `.npy` files plus a `meta.json`
//! manifest.  The format is intentionally minimal — no compression, no
//! schema negotiation — so it can be read by any language with an NPY parser
//! and eliminates HDF5 overhead for benchmarking and debugging.
//!
//! ## File layout
//!
//! ```text
//! snapshot.scxd/
//!   meta.json               # full manifest: shapes, dtypes, slot keys
//!   obs_index.txt           # n_obs lines (cell barcodes)
//!   var_index.txt           # n_vars lines (gene names)
//!   uns.json                # unstructured metadata (absent if empty)
//!   X/
//!     data.npy              # (nnz,)      f32|f64|i32|u32
//!     indices.npy           # (nnz,)      u32
//!     indptr.npy            # (n_obs+1,)  u64
//!   obs/
//!     {col}.npy             # numeric / bool column
//!     {col}_strings.txt     # string column, one value per line
//!     {col}_codes.npy       # categorical codes (u32)
//!     {col}_levels.txt      # categorical levels, one per line
//!   var/                    # same layout as obs/
//!   obsm/
//!     {key}.npy             # (n_obs, k) f64 dense, C-contiguous
//!   varm/
//!     {key}.npy             # (n_vars, k) f64 dense, C-contiguous
//!   layers/{name}/
//!     data.npy
//!     indices.npy
//!     indptr.npy
//!   obsp/{name}/            # same layout as layers/{name}/
//!   varp/{name}/
//! ```

mod format;
mod meta;
mod reader;
mod writer;

pub use meta::SlotFilter;
pub use reader::NpyIrReader;
pub use writer::NpyIrWriter;

#[cfg(test)]
mod tests;
