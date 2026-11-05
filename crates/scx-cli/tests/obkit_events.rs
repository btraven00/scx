//! `scx convert` with `OBKIT_DIR` set writes obkit phase anchors
//! (`obkit-events.jsonl`, spec 0.1): one start/end pair per phase, in order,
//! with millisecond UTC timestamps; without it, nothing is written.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

#[test]
fn convert_writes_paired_phase_events() {
    let dir = tempfile::tempdir().unwrap();
    let ob = dir.path().join("ob");
    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["convert"])
        .arg(fixture("tiny_raw.h5ad"))
        .arg(dir.path().join("out.h5seurat"))
        .arg("--dgcmatrix")
        .env("OBKIT_DIR", &ob)
        .status()
        .unwrap();
    assert!(status.success());

    let text = std::fs::read_to_string(ob.join("obkit-events.jsonl")).unwrap();
    let events: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let got: Vec<(String, String)> = events
        .iter()
        .map(|e| {
            (
                e["event"].as_str().unwrap().into(),
                e["phase"].as_str().unwrap().into(),
            )
        })
        .collect();
    let want: Vec<(String, String)> = [
        "open",
        "metadata",
        "layers",
        "obsp",
        "x",
        "finalize",
        "output_hash",
    ]
    .iter()
    .flat_map(|p| {
        [
            (p.to_string(), "start".into()),
            (p.to_string(), "end".into()),
        ]
    })
    .collect();
    assert_eq!(got, want);
    for e in &events {
        let ts = e["ts"].as_str().unwrap();
        // 2026-04-20T17:10:00.123Z
        assert_eq!(ts.len(), 24, "{ts}");
        assert!(ts.ends_with('Z') && ts.as_bytes()[19] == b'.', "{ts}");
        assert!(e["pid"].as_u64().is_some());
    }
    // Timestamps never go backwards.
    assert!(events
        .windows(2)
        .all(|w| w[0]["ts"].as_str() <= w[1]["ts"].as_str()));
}

#[test]
fn no_events_without_obkit_dir() {
    let dir = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["convert"])
        .arg(fixture("tiny.h5ad"))
        .arg(dir.path().join("out.h5ad"))
        .env_remove("OBKIT_DIR")
        .current_dir(dir.path())
        .status()
        .unwrap();
    assert!(status.success());
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(!names.iter().any(|n| n.contains("obkit")), "{names:?}");
}
