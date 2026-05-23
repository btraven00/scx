//! Build an in-memory matrix and write it as `.h5ad` and a BPCells `.h5seurat`
//! using the public `scx_core::api::write` surface.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example write_from_ndarray -- /tmp/demo
//! ```
//!
//! Produces `/tmp/demo.h5ad` and `/tmp/demo.h5seurat`.

use std::path::PathBuf;

use ndarray::Array2;
use scx_core::api::write::{
    write_bpcells_h5seurat_csr, write_h5ad_dense, BpcellsOptions, H5AdOptions,
};
use scx_core::api::{ObsTable, VarTable};
use sprs::TriMatI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stem = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/scx_demo"));

    let n_obs = 6;
    let n_vars = 4;

    // Build a small dense matrix and write it as .h5ad. Exact zeros are dropped.
    let mut dense = Array2::<f32>::zeros((n_obs, n_vars));
    dense[[0, 1]] = 1.0;
    dense[[2, 0]] = 2.0;
    dense[[2, 3]] = 3.0;
    dense[[5, 2]] = 4.0;

    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
        columns: vec![],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
        columns: vec![],
    };

    let h5ad_path = stem.with_extension("h5ad");
    write_h5ad_dense(
        &h5ad_path,
        dense.view(),
        obs.clone(),
        var.clone(),
        &H5AdOptions::default(),
    )?;
    println!("wrote {}", h5ad_path.display());

    // Build the same matrix as sparse CSR via sprs and write it as a
    // BPCells-backed .h5seurat.
    let mut tri = TriMatI::<f32, u32>::new((n_obs, n_vars));
    tri.add_triplet(0, 1, 1.0);
    tri.add_triplet(2, 0, 2.0);
    tri.add_triplet(2, 3, 3.0);
    tri.add_triplet(5, 2, 4.0);
    let csr = tri.to_csr();

    let h5seurat_path = stem.with_extension("h5seurat");
    write_bpcells_h5seurat_csr(
        &h5seurat_path,
        csr.view(),
        obs,
        var,
        &BpcellsOptions::default(),
    )?;
    println!("wrote {}", h5seurat_path.display());

    Ok(())
}
