//! H5AD with the counts in `raw.X` (scanpy's usual layout) -> H5Seurat.
//!
//! Regression: scx ignored `raw`, so a normalised X went into Seurat's
//! `counts` slot and the real counts were dropped. `raw.X` now fills `counts`
//! and X goes to `data`, for both H5Seurat writers. A `raw` with a different
//! gene set than X can't share X's assay, so X stays the counts there.

use std::path::{Path, PathBuf};
use std::process::Command;

use futures::StreamExt;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn convert(input: &Path, out: &Path, extra: &[&str]) {
    let res = Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("convert")
        .arg(input)
        .arg(out)
        .args(extra)
        .output()
        .expect("failed to run scx convert");
    assert!(
        res.status.success(),
        "scx convert failed:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
}

/// The H5Seurat matrix in `slot`, dense, cells x genes.
fn slot(path: &Path, slot: &str) -> Vec<Vec<f64>> {
    futures::executor::block_on(async {
        let opts = scx_core::OpenOptions {
            layer: Some(slot.to_string()),
            ..scx_core::OpenOptions::new(100)
        };
        let mut r = scx_core::open(path.to_str().unwrap(), &opts).await.unwrap();
        let (n_obs, n_vars) = r.shape();
        let mut out = vec![vec![0.0; n_vars]; n_obs];
        let mut stream = r.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            let values = chunk.data.data.to_f64();
            for row in 0..chunk.nrows {
                let span = chunk.data.indptr[row] as usize..chunk.data.indptr[row + 1] as usize;
                for k in span {
                    out[chunk.row_offset + row][chunk.data.indices[k] as usize] = values[k];
                }
            }
        }
        out
    })
}

// tests/fixtures/tiny: raw.X holds COUNTS, X holds COUNTS * 0.5.
const COUNTS: [[f64; 5]; 6] = [
    [1.0, 0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 4.0],
    [5.0, 0.0, 6.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 7.0],
    [8.0, 9.0, 0.0, 0.0, 0.0],
];

fn counts(n_genes: usize, scale: f64) -> Vec<Vec<f64>> {
    COUNTS
        .iter()
        .map(|row| row[..n_genes].iter().map(|v| v * scale).collect())
        .collect()
}

#[test]
fn raw_x_fills_counts_and_x_goes_to_data() {
    for extra in [&[][..], &["--dgcmatrix"][..]] {
        let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        convert(&fixture("tiny_raw.h5ad"), out.path(), extra);
        assert_eq!(
            slot(out.path(), "counts"),
            counts(5, 1.0),
            "counts, {extra:?}"
        );
        assert_eq!(slot(out.path(), "data"), counts(5, 0.5), "data, {extra:?}");
    }
}

#[test]
fn raw_with_other_genes_leaves_x_as_counts() {
    let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    convert(
        &fixture("tiny_raw_subset.h5ad"),
        out.path(),
        &["--dgcmatrix"],
    );
    assert_eq!(slot(out.path(), "counts"), counts(4, 0.5));
}
