//! End-to-end: `scx convert <seurat.rds> out.h5ad`, from a local path and
//! (with the `net` feature) from a URL, which is downloaded first. Uses the
//! committed tiny fixture with split v5 layers.

use std::path::{Path, PathBuf};
use std::process::Command;

use futures::StreamExt;
use scx_core::dtype::TypedVec;
use scx_core::h5ad::H5AdReader;
use scx_core::stream::DatasetReader;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny/tiny_v5_split.rds")
        .canonicalize()
        .unwrap()
}

fn convert(input: &str, out: &Path) {
    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["convert", input, out.to_str().unwrap(), "--dtype", "f64"])
        .status()
        .unwrap();
    assert!(status.success(), "scx convert {input} failed");
}

/// The tiny counts sum to 45 over 6 cells x 5 genes.
fn assert_tiny(out: &Path) {
    let mut r = H5AdReader::open(out, 100).unwrap();
    assert_eq!(r.shape(), (6, 5));
    let total: f64 = futures::executor::block_on(r.x_stream().collect::<Vec<_>>())
        .into_iter()
        .map(|c| match c.unwrap().data.data {
            TypedVec::F64(v) => v.iter().sum::<f64>(),
            TypedVec::F32(v) => v.iter().map(|&x| x as f64).sum(),
            _ => panic!("unexpected dtype"),
        })
        .sum();
    assert_eq!(total, 45.0);
}

#[test]
fn converts_local_rds() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.h5ad");
    convert(fixture().to_str().unwrap(), &out);
    assert_tiny(&out);
}

#[cfg(feature = "net")]
#[test]
fn converts_rds_url() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.h5ad");
    let url = format!("file://{}", fixture().display());
    convert(&url, &out);
    assert_tiny(&out);
}

/// GEO's `x.rds.gz`: a gzip around saveRDS's own gzip, named with
/// percent-escapes in the URL (`x%2Erds%2Egz`).
#[cfg(feature = "net")]
#[test]
fn converts_double_gzipped_rds_url() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let gz = dir.path().join("obj.rds.gz");
    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(&gz).unwrap(),
        flate2::Compression::fast(),
    );
    enc.write_all(&std::fs::read(fixture()).unwrap()).unwrap();
    enc.finish().unwrap();
    let out = dir.path().join("out.h5ad");
    let url = format!("file://{}/obj%2Erds%2Egz", dir.path().display());
    convert(&url, &out);
    assert_tiny(&out);
}
