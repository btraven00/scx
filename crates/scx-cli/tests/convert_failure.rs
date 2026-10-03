//! A conversion that fails part-way must leave no output behind, and must
//! not destroy an output file that existed before the run.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A MatrixMarket dir whose entries aren't ordered by cell: it opens fine and
/// fails while streaming, after the writer has created its output.
fn unsorted_mtx(dir: &Path) -> PathBuf {
    let m = dir.join("mtx");
    std::fs::create_dir(&m).unwrap();
    std::fs::write(
        m.join("matrix.mtx"),
        "%%MatrixMarket matrix coordinate integer general\n3 3 3\n1 3 1\n1 1 2\n2 2 3\n",
    )
    .unwrap();
    std::fs::write(m.join("barcodes.tsv"), "c0\nc1\nc2\n").unwrap();
    std::fs::write(
        m.join("features.tsv"),
        "g0\tG0\tGene Expression\ng1\tG1\tGene Expression\ng2\tG2\tGene Expression\n",
    )
    .unwrap();
    m
}

fn convert(input: &Path, output: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["convert", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn failed_convert_leaves_no_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = unsorted_mtx(dir.path());
    for name in ["out.h5ad", "out.h5seurat"] {
        let out = dir.path().join(name);
        let res = convert(&input, &out);
        assert!(!res.status.success(), "{name}: conversion should fail");
        assert!(!out.exists(), "{name}: partial output left behind");
    }
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n != "mtx")
        .collect();
    assert!(leftovers.is_empty(), "stray files: {leftovers:?}");
}

#[test]
fn failed_convert_keeps_an_existing_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = unsorted_mtx(dir.path());
    let out = dir.path().join("out.h5ad");
    std::fs::write(&out, b"previous result").unwrap();
    assert!(!convert(&input, &out).status.success());
    assert_eq!(std::fs::read(&out).unwrap(), b"previous result");
}
