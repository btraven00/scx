//! Integration coverage for `scx convert --only/--exclude` (auxiliary-slot
//! filtering). Uses the committed `norman_subset.h5ad` fixture: asserts an
//! excluded slot is dropped from the output while X survives, and that
//! `--only`/`--exclude` are mutually exclusive.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use scx_core::h5ad::H5AdReader;
use scx_core::stream::DatasetReader;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_scx"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn convert(input: &Path, out: &Path, extra: &[&str]) -> Output {
    let mut cmd = Command::new(binary());
    cmd.arg("convert").arg(input).arg(out);
    for a in extra {
        cmd.arg(a);
    }
    cmd.output().expect("failed to run scx convert")
}

fn layer_names(path: &Path) -> Vec<String> {
    let mut r = H5AdReader::open(path, 500).unwrap();
    futures::executor::block_on(r.layer_metas())
        .unwrap()
        .into_iter()
        .map(|m| m.name)
        .collect()
}

#[test]
fn exclude_drops_the_slot_but_keeps_x() {
    let input = fixture("norman_subset.h5ad");
    assert!(input.exists(), "committed fixture missing: {input:?}");

    // Control: a plain convert carries the source layer through, so the fixture
    // genuinely has a slot to drop.
    let plain = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    assert_eq!(convert(&input, plain.path(), &[]).status.code(), Some(0));
    assert!(
        !layer_names(plain.path()).is_empty(),
        "fixture must ship a layer for this test to mean anything"
    );

    // --exclude layers removes it; X still round-trips.
    let excl = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    let out = convert(&input, excl.path(), &["--exclude", "layers"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        layer_names(excl.path()).is_empty(),
        "layers should be dropped by --exclude layers"
    );
    let r = H5AdReader::open(excl.path(), 500).unwrap();
    assert!(r.shape().0 > 0, "X/obs must survive slot filtering");
}

#[test]
fn only_and_exclude_are_mutually_exclusive() {
    let out = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    let res = convert(
        &fixture("norman_subset.h5ad"),
        out.path(),
        &["--only", "obsm", "--exclude", "obsp"],
    );
    assert_ne!(
        res.status.code(),
        Some(0),
        "--only + --exclude must conflict"
    );
    assert!(String::from_utf8_lossy(&res.stderr).contains("cannot be used with"));
}

/// `--x-slot layers/<name>` on h5ad output writes the matrix there, leaves /X
/// absent, and drops a source layer of the same name instead of writing it twice.
#[test]
fn x_slot_layer_on_h5ad_output() {
    let input = fixture("norman_subset.h5ad");
    let (src_nnz, src_layers) = {
        let mut r = H5AdReader::open(&input, 500).unwrap();
        (count_nnz(&mut r), layer_names(&input))
    };
    assert!(
        src_layers.contains(&"counts".to_string()),
        "fixture must have a counts layer"
    );

    let out = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
    let res = convert(&input, out.path(), &["--x-slot", "layers/counts"]);
    assert_eq!(
        res.status.code(),
        Some(0),
        "stderr:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
    // The reader only falls back to a layer when /X is absent.
    let mut r = H5AdReader::open(out.path(), 500).unwrap();
    assert_eq!(r.x_source(), "layers/counts");
    assert_eq!(
        count_nnz(&mut r),
        src_nnz,
        "layers/counts must hold the source X"
    );
    assert!(!layer_names(out.path()).contains(&"counts".to_string()));

    let bad = convert(&input, out.path(), &["--x-slot", "obsm/counts"]);
    assert_ne!(
        bad.status.code(),
        Some(0),
        "non-layer slot must be rejected"
    );
}

fn count_nnz(r: &mut H5AdReader) -> usize {
    use futures::StreamExt;
    futures::executor::block_on(async {
        let mut n = 0;
        let mut s = r.x_stream();
        while let Some(c) = s.next().await {
            n += c.unwrap().data.indices.len();
        }
        n
    })
}
