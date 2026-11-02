//! `scx validate`: exit 0 when every check passes, 1 when one fails, and 2
//! when validation can't run (unreadable file, bad or empty schema), so a
//! pipeline gate can tell a bad file from a broken gate.
//!
//! Regression: unknown schema keys were ignored, so a typo (`obps:`) or an
//! option that doesn't exist (`check_finite:`) passed with "All 0 checks
//! passed", and errors exited 1 like failed checks.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn tiny() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny/tiny.h5ad")
}

/// `scx validate <input> --schema <yaml>` with `extra` args.
fn validate(input: &Path, yaml: &str, extra: &[&str]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let schema = dir.path().join("schema.yaml");
    std::fs::write(&schema, yaml).unwrap();
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .env("NO_COLOR", "1")
        .arg("validate")
        .arg(input)
        .arg("--schema")
        .arg(&schema)
        .args(extra)
        .output()
        .expect("failed to run scx validate")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

const PASSES: &str = "obs: \">= 6\"\nvars: 5\nlayers: [counts]\nobsm: [X_pca]\nobsp: [connectivities]\nx_dtype: f32\nobs_index_unique: true\n";

#[test]
fn passing_schema_exits_0() {
    let out = validate(&tiny(), PASSES, &[]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    assert!(text(&out).contains("All 7 checks passed"), "{}", text(&out));
}

#[test]
fn failed_check_exits_1() {
    let out = validate(&tiny(), "obs: \">= 1000\"\nlayers: [normalized]\n", &[]);
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let t = text(&out);
    assert!(
        t.contains("FAIL  obs") && t.contains("missing: normalized"),
        "{t}"
    );
}

#[test]
fn schema_errors_exit_2_and_name_the_problem() {
    let cases = [
        ("obps: [connectivities]\n", "unknown field `obps`"),
        (
            "obs: 6\ncheck_finite: true\n",
            "unknown field `check_finite`",
        ),
        ("obs: [1, 2]\n", "invalid schema"),
        ("{}\n", "has no checks"),
    ];
    for (yaml, want) in cases {
        let out = validate(&tiny(), yaml, &[]);
        assert_eq!(out.status.code(), Some(2), "{yaml:?}: {}", text(&out));
        assert!(text(&out).contains(want), "{yaml:?}: {}", text(&out));
    }
}

#[test]
fn unreadable_input_exits_2() {
    let out = validate(Path::new("/nonexistent/x.h5ad"), PASSES, &[]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("cannot open"), "{}", text(&out));
}

#[test]
fn json_report() {
    let out = validate(&tiny(), "obs: 6\nvars: 4\n", &["--json"]);
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let start = stdout.find('{').expect("JSON on stdout");
    let v: serde_json::Value = serde_json::from_str(&stdout[start..]).unwrap();
    assert_eq!(
        (v["passed"].as_u64(), v["failed"].as_u64()),
        (Some(1), Some(1))
    );
}
