//! Integration coverage for `scx convert --compress` (gzip h5ad output).
//!
//! Uses the committed `tests/fixtures/norman_subset.h5ad` (not the gitignored
//! golden fixtures) so these run in CI and cover the CLI compression wiring.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_scx"))
}

fn fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(filename)
}

fn convert(input: &Path, out: &Path, extra: &[&str]) -> Output {
    let mut cmd = Command::new(binary_path());
    cmd.arg("convert").arg(input).arg(out);
    for a in extra {
        cmd.arg(a);
    }
    cmd.output().expect("failed to run scx convert")
}

fn assert_ok(out: &Output, what: &str) {
    assert_eq!(
        out.status.code(),
        Some(0),
        "{what} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn convert_h5ad_with_compress_is_smaller_and_readable() {
    let input = fixture_path("norman_subset.h5ad");
    assert!(input.exists(), "committed fixture missing: {input:?}");

    let plain = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    let comp = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();

    assert_ok(&convert(&input, plain.path(), &[]), "plain convert");
    assert_ok(
        &convert(&input, comp.path(), &["--compress"]),
        "gzip convert",
    );

    let ps = std::fs::metadata(plain.path()).unwrap().len();
    let cs = std::fs::metadata(comp.path()).unwrap().len();
    assert!(
        cs < ps,
        "compressed ({cs}) should be smaller than plain ({ps})"
    );

    // The gzip output must read back through the normal reader path.
    let inspect = Command::new(binary_path())
        .arg("inspect")
        .arg(comp.path())
        .output()
        .expect("failed to run scx inspect");
    assert_ok(&inspect, "inspect on compressed output");
}

#[test]
fn convert_with_explicit_compress_level() {
    let input = fixture_path("norman_subset.h5ad");
    assert!(input.exists(), "committed fixture missing: {input:?}");

    let comp = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    assert_ok(
        &convert(&input, comp.path(), &["--compress", "9"]),
        "gzip convert at level 9",
    );
    assert!(std::fs::metadata(comp.path()).unwrap().len() > 0);
}

/// Covers the H5Seurat-source branch of the convert dispatch with `--compress`:
/// build an h5seurat from the committed h5ad fixture, then gzip it back to h5ad.
#[test]
fn convert_from_h5seurat_source_with_compress() {
    let input = fixture_path("norman_subset.h5ad");
    assert!(input.exists(), "committed fixture missing: {input:?}");

    let mid = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    assert_ok(&convert(&input, mid.path(), &[]), "h5ad -> h5seurat");

    let out = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    assert_ok(
        &convert(mid.path(), out.path(), &["--compress"]),
        "h5seurat -> h5ad --compress",
    );
    assert!(std::fs::metadata(out.path()).unwrap().len() > 0);
}

/// Covers the NpyDir-source branch of the convert dispatch with `--compress`:
/// snapshot the committed fixture to an npy directory, then gzip-convert it.
#[test]
fn convert_from_npy_snapshot_with_compress() {
    let input = fixture_path("norman_subset.h5ad");
    assert!(input.exists(), "committed fixture missing: {input:?}");

    let dir = tempfile::tempdir().unwrap();
    let snap = dir.path().join("snap");
    let snapshot = Command::new(binary_path())
        .arg("snapshot")
        .arg(&input)
        .arg(&snap)
        .output()
        .expect("failed to run scx snapshot");
    assert_ok(&snapshot, "snapshot to npy dir");

    let out = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    assert_ok(
        &convert(&snap, out.path(), &["--compress"]),
        "npy dir -> h5ad --compress",
    );
    assert!(std::fs::metadata(out.path()).unwrap().len() > 0);
}

#[test]
fn compress_is_ignored_and_warns_for_h5seurat_output() {
    let input = fixture_path("norman_subset.h5ad");
    assert!(input.exists(), "committed fixture missing: {input:?}");

    let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let result = convert(&input, out.path(), &["--compress"]);
    assert_ok(&result, "h5seurat convert with --compress");

    // The CLI logs via tracing's fmt subscriber, which writes to stdout.
    let logs = String::from_utf8_lossy(&result.stdout);
    assert!(
        logs.contains("--compress only applies to .h5ad"),
        "expected an ignore warning for .h5seurat output, got stdout:\n{logs}"
    );
}
