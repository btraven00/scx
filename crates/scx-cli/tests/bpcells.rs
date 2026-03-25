use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/debug/scx")
}

fn golden_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(filename)
}

fn fixture_exists(filename: &str) -> bool {
    golden_path(filename).exists()
}

#[test]
fn test_convert_h5ad_to_h5seurat_bpcells_minimal() {
    if !fixture_exists("pbmc3k_reference.h5ad") {
        return;
    }

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let output = Command::new(binary_path())
        .arg("convert")
        .arg(golden_path("pbmc3k_reference.h5ad"))
        .arg(&out)
        .arg("--bpcells")
        .output()
        .expect("failed to run scx convert --bpcells");

    assert_eq!(
        output.status.code(),
        Some(0),
        "expected exit code 0\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let inspect = Command::new(binary_path())
        .arg("inspect")
        .arg(&out)
        .output()
        .expect("failed to run scx inspect on BPCells output");

    assert_eq!(
        inspect.status.code(),
        Some(0),
        "expected inspect exit code 0\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&inspect.stdout),
        String::from_utf8_lossy(&inspect.stderr)
    );

    let stdout = String::from_utf8_lossy(&inspect.stdout);

    assert!(
        stdout.contains("Format : H5Seurat"),
        "missing 'Format : H5Seurat'\n{}",
        stdout
    );
    assert!(
        stdout.contains("Shape  : 2700 × 13714  obs × vars"),
        "missing expected shape\n{}",
        stdout
    );
    assert!(
        stdout.contains("obs ("),
        "missing obs section\n{}",
        stdout
    );
    assert!(
        stdout.contains("var ("),
        "missing var section\n{}",
        stdout
    );
}

#[test]
fn test_convert_h5seurat_to_h5seurat_bpcells_minimal() {
    if !fixture_exists("pbmc3k.h5seurat") {
        return;
    }

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let output = Command::new(binary_path())
        .arg("convert")
        .arg(golden_path("pbmc3k.h5seurat"))
        .arg(&out)
        .arg("--bpcells")
        .output()
        .expect("failed to run scx convert h5seurat -> h5seurat --bpcells");

    assert_eq!(
        output.status.code(),
        Some(0),
        "expected exit code 0\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let inspect = Command::new(binary_path())
        .arg("inspect")
        .arg(&out)
        .output()
        .expect("failed to run scx inspect on re-encoded BPCells output");

    assert_eq!(
        inspect.status.code(),
        Some(0),
        "expected inspect exit code 0\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&inspect.stdout),
        String::from_utf8_lossy(&inspect.stderr)
    );

    let stdout = String::from_utf8_lossy(&inspect.stdout);

    assert!(
        stdout.contains("Format : H5Seurat"),
        "missing 'Format : H5Seurat'\n{}",
        stdout
    );
    assert!(
        stdout.contains("Shape  : 2700 × 13714  obs × vars"),
        "missing expected shape\n{}",
        stdout
    );
    assert!(
        stdout.contains("obs ("),
        "missing obs section\n{}",
        stdout
    );
    assert!(
        stdout.contains("var ("),
        "missing var section\n{}",
        stdout
    );
}