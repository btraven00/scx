//! Regression: an output extension scx has no writer for used to fall through
//! to the h5ad writer (convert) or to CSV (export), so `out.zar` came out as
//! an HDF5 file under the wrong name. Both now refuse it and write nothing.

use std::path::PathBuf;
use std::process::Command;

fn scx(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(args)
        .output()
        .unwrap()
}

fn tiny_h5ad() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny/tiny.h5ad")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn convert_refuses_unknown_output_extension() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["out.zar", "out.loom", "out"] {
        let out = dir.path().join(name);
        let res = scx(&["convert", &tiny_h5ad(), out.to_str().unwrap()]);
        let stderr = String::from_utf8_lossy(&res.stderr);
        assert!(!res.status.success(), "{name}: should fail");
        assert!(
            stderr.contains("unsupported output extension"),
            "{name}: {stderr}"
        );
        assert!(
            stderr.contains(".zarr"),
            "{name}: should list the formats: {stderr}"
        );
    }
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "nothing written"
    );

    // A known extension still converts (h5ad: every build writes it).
    let out = dir.path().join("out.h5ad");
    let res = scx(&["convert", &tiny_h5ad(), out.to_str().unwrap()]);
    assert!(
        res.status.success(),
        "{}",
        String::from_utf8_lossy(&res.stderr)
    );
}

#[test]
fn export_refuses_unknown_output_extension() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("obs.tsv");
    let res = scx(&[
        "export",
        &tiny_h5ad(),
        "--slot",
        "obs",
        "--output",
        out.to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(!res.status.success());
    assert!(stderr.contains("use .csv or .parquet"), "{stderr}");
    assert!(!out.exists());
}
