//! `scx --version`.

use std::process::Command;

/// `--version` names the version and whether `net` is compiled in.
#[test]
fn version_names_the_net_feature() {
    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("--version")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let flag = if cfg!(feature = "net") {
        "+net"
    } else {
        "-net"
    };
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")) && stdout.contains(flag),
        "{stdout}"
    );
}
