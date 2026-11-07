//! `scx inspect` and `scx convert` on a MuData file written by mudata
//! (tests/fixtures/tiny/tiny.h5mu: rna 6x5 CSR, prot 6x2 dense).

use std::path::PathBuf;
use std::process::Command;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny/tiny.h5mu")
}

fn scx(args: &[&str]) -> String {
    let res = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "{}",
        String::from_utf8_lossy(&res.stderr)
    );
    String::from_utf8_lossy(&res.stdout).into_owned()
}

#[test]
fn inspect_lists_modalities_and_reads_one() {
    let input = fixture();
    let input = input.to_str().unwrap();
    let out = scx(&["inspect", input]);
    for want in [
        "modalities: rna, prot",
        "reading /mod/rna/X",
        "H5MU (MuData)",
        "6 obs × 5 vars",
    ] {
        assert!(out.contains(want), "{want}:\n{out}");
    }
    let out = scx(&["inspect", input, "--assay", "ADT"]);
    assert!(
        out.contains("reading /mod/prot/X") && out.contains("6 obs × 2 vars"),
        "{out}"
    );
}
