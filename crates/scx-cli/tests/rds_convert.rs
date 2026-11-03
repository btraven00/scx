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

/// A downloaded source is hashed as it arrives, so URL conversions record
/// the same source hash as local ones, in uns and in the sidecar.
#[cfg(feature = "net")]
#[test]
fn rds_url_records_source_hash() {
    let dir = tempfile::tempdir().unwrap();
    let hash = |input: &str, name: &str| {
        let out = dir.path().join(name);
        convert(input, &out);
        let side = std::fs::read_to_string(dir.path().join(format!("{name}.prov.json"))).unwrap();
        let side: serde_json::Value = serde_json::from_str(&side).unwrap();
        let uns = Command::new(env!("CARGO_BIN_EXE_scx"))
            .arg("inspect")
            .arg(&out)
            .output()
            .unwrap();
        let h = side["source"]["hash"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(
            String::from_utf8_lossy(&uns.stdout).contains(&h),
            "uns lacks {h}"
        );
        h
    };
    let local = hash(fixture().to_str().unwrap(), "local.h5ad");
    let remote = hash(&format!("file://{}", fixture().display()), "remote.h5ad");
    assert!(local.starts_with("sha256:"), "{local}");
    assert_eq!(remote, local);
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

/// `inspect` downloads an .rds URL first, as `convert` does.
#[cfg(feature = "net")]
#[test]
fn inspects_rds_url() {
    let url = format!("file://{}", fixture().display());
    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["inspect", &url])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("RDS (Seurat)") && stdout.contains("6 obs"),
        "{stdout}"
    );
}

/// inspect asks before downloading a URL over 1 GB (one HEAD request gives
/// the size), and stops when it can't ask: stdin is not a terminal here.
#[cfg(feature = "net")]
#[test]
fn inspect_stops_before_a_large_download() {
    let dir = tempfile::tempdir().unwrap();
    let big = dir.path().join("big.rds");
    // Sparse: 2 GB long, no blocks on disk.
    std::fs::File::create(&big)
        .unwrap()
        .set_len(2_000_000_000)
        .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["inspect", &format!("file://{}", big.display())])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("2.0 GB") && err.contains("--yes"), "{err}");
}

/// When HEAD can't give a size, inspect doesn't ask; the download itself
/// then reports what is wrong.
#[cfg(feature = "net")]
#[test]
fn inspect_without_a_size_goes_straight_to_the_download() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("file://{}", dir.path().join("missing.rds").display());
    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["inspect", &url])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("download") && !err.contains("--yes"), "{err}");
}
