//! A local input that does not exist is reported as missing, not as a file
//! whose format could not be detected.

use std::process::Command;

fn scx(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn missing_input_says_it_does_not_exist() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nonexistent.rds");
    let missing = missing.to_str().unwrap();
    let out = dir.path().join("out.h5ad");
    for args in [
        vec!["inspect", missing],
        vec!["convert", missing, out.to_str().unwrap()],
    ] {
        let res = scx(&args);
        let stderr = String::from_utf8_lossy(&res.stderr);
        assert!(!res.status.success(), "{}: should fail", args[0]);
        assert!(
            stderr.contains(&format!("no such file or directory: '{missing}'")),
            "{}: {stderr}",
            args[0]
        );
    }
}
