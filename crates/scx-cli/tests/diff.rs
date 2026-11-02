//! `scx diff`: real conversions compare SAME, different files exit 1, and
//! errors exit 2. Planted differences are covered in scx-core's diff tests.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn scx(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .env("NO_COLOR", "1")
        .env("RUST_LOG", "error")
        .args(args)
        .output()
        .expect("failed to run scx")
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// raw.X → counts and X → data, with both h5seurat writers: diff pairs them
/// the way convert did and finds nothing.
#[test]
fn raw_counts_conversions_diff_same() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny_raw.h5ad");
    for extra in [&[][..], &["--dgcmatrix"][..]] {
        let out = tmp.path().join("o.h5seurat");
        let _ = std::fs::remove_file(&out);
        let mut args = vec!["convert", s(&src), s(&out)];
        args.extend_from_slice(extra);
        assert!(scx(&args).status.success());
        let d = scx(&["diff", s(&src), s(&out)]);
        assert_eq!(d.status.code(), Some(0), "{extra:?}\n{}", text(&d));
        let t = text(&d);
        assert!(
            t.contains("X (vs layers/data)") && t.contains("raw (vs X)"),
            "{t}"
        );
    }
}

#[test]
fn a_file_against_itself_is_same() {
    let t = fixture("tiny.h5ad");
    let d = scx(&["diff", s(&t), s(&t)]);
    assert_eq!(d.status.code(), Some(0), "{}", text(&d));
    assert!(text(&d).contains("slots the same"), "{}", text(&d));
}

/// --map overrides the automatic pairing: X against X, normalised against
/// counts, differs.
#[test]
fn map_overrides_pairing() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny_raw.h5ad");
    let out = tmp.path().join("o.h5seurat");
    assert!(scx(&["convert", s(&src), s(&out), "--dgcmatrix"])
        .status
        .success());
    let d = scx(&["diff", s(&src), s(&out), "--map", "X=X", "--only", "X"]);
    assert_eq!(d.status.code(), Some(1), "{}", text(&d));
    assert!(text(&d).contains("DIFF    X "), "{}", text(&d));
}

#[test]
fn different_files_exit_1_with_a_json_report() {
    let d = scx(&[
        "diff",
        s(&fixture("tiny.h5ad")),
        s(&fixture("tiny_raw.h5ad")),
        "--json",
    ]);
    assert_eq!(d.status.code(), Some(1), "{}", text(&d));
    let stdout = String::from_utf8_lossy(&d.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout[stdout.find('{').unwrap()..]).unwrap();
    let slots = v["slots"].as_array().unwrap();
    assert!(slots.iter().any(|s| s["status"] != "SAME"), "{v}");
    assert!(slots
        .iter()
        .all(|s| s["slot"].is_string() && s["summary"].is_string()));
}

#[test]
fn errors_exit_2() {
    let t = fixture("tiny.h5ad");
    let missing = scx(&["diff", s(&t), "/nonexistent/x.h5ad"]);
    assert_eq!(missing.status.code(), Some(2), "{}", text(&missing));
    assert!(text(&missing).contains("cannot open"));
    let bad_map = scx(&["diff", s(&t), s(&t), "--map", "X"]);
    assert_eq!(bad_map.status.code(), Some(2), "{}", text(&bad_map));
    assert!(text(&bad_map).contains("expected A=B"));
}
