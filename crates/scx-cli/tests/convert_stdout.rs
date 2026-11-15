//! `scx convert IN -` writes the h5ad to stdout, so a subset can be piped
//! (`ssh host scx convert big.h5ad - --where ... | scx convert - out.h5ad`),
//! and `--only obs:COL` keeps only the named obs columns.
//!
//! tiny.h5ad's cell_type is B T NaN B NK T.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

fn tiny() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny/tiny.h5ad")
}

fn obs(path: &Path) -> scx_core::ir::ObsTable {
    let opts = scx_core::OpenOptions {
        source_hash: Default::default(),
        ..scx_core::OpenOptions::new(2)
    };
    futures::executor::block_on(async {
        let mut r = scx_core::open(path.to_str().unwrap(), &opts).await.unwrap();
        r.obs().await.unwrap()
    })
}

#[test]
fn stdout_output_pipes_a_subset_into_another_convert() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        obs(&tiny()).columns.len() > 1,
        "fixture needs columns to drop"
    );

    // The sending side: TMPDIR is ours, so the leftover check below is exact.
    let tmp = dir.path().join("tmp");
    std::fs::create_dir(&tmp).unwrap();
    let sent = scx()
        .env("RUST_LOG", "info")
        .env("TMPDIR", &tmp)
        .arg("convert")
        .arg(tiny())
        .args(["-", "--where", "cell_type=B", "--only", "obs:cell_type"])
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&sent.stderr);
    assert!(sent.status.success(), "{log}");
    // stdout is the file and nothing else; the log went to stderr.
    assert!(
        sent.stdout.starts_with(b"\x89HDF\r\n\x1a\n"),
        "not an HDF5 file"
    );
    assert!(log.contains("--where keeps 2 cells"), "{log}");
    assert_eq!(
        std::fs::read_dir(&tmp).unwrap().count(),
        0,
        "temporary file left behind"
    );

    // The receiving side reads it from stdin.
    let out = dir.path().join("out.h5ad");
    let mut recv = scx()
        .args(["convert", "-"])
        .arg(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(&mut recv.stdin.take().unwrap(), &sent.stdout).unwrap();
    let recv = recv.wait_with_output().unwrap();
    assert!(
        recv.status.success(),
        "{}",
        String::from_utf8_lossy(&recv.stderr)
    );

    let got = obs(&out);
    assert_eq!(got.index.len(), 2);
    let names: Vec<&str> = got.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["cell_type"]);
}

/// `--exclude obs:COL` drops that column alone; a filter with no column spec
/// (`--exclude obsm`) keeps every column.
#[test]
fn exclude_drops_one_obs_column() {
    let dir = tempfile::tempdir().unwrap();
    let all = obs(&tiny()).columns.len();
    for (spec, want) in [("obs:cell_type", all - 1), ("obsm", all)] {
        let out = dir.path().join(format!("{}.h5ad", spec.replace(':', "_")));
        let res = scx()
            .arg("convert")
            .arg(tiny())
            .arg(&out)
            .args(["--exclude", spec])
            .output()
            .unwrap();
        assert!(
            res.status.success(),
            "{}",
            String::from_utf8_lossy(&res.stderr)
        );
        assert_eq!(obs(&out).columns.len(), want, "--exclude {spec}");
    }
}
