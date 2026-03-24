use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/scx")
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
fn test_inspect_h5seurat() {
    if !fixture_exists("pbmc3k.h5seurat") {
        return;
    }

    let output = Command::new(binary_path())
        .arg("inspect")
        .arg(golden_path("pbmc3k.h5seurat"))
        .output()
        .expect("failed to run scx inspect");

    assert_eq!(output.status.code(), Some(0), "expected exit code 0");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Format : H5Seurat"), "missing 'Format : H5Seurat'\n{}", stdout);
    assert!(stdout.contains("Shape  : 2700 obs × 13714 vars"), "missing shape\n{}", stdout);
    assert!(stdout.contains("nCount_RNA"), "missing 'nCount_RNA'\n{}", stdout);
    assert!(stdout.contains("orig.ident"), "missing 'orig.ident'\n{}", stdout);
    assert!(stdout.contains("categorical"), "missing 'categorical'\n{}", stdout);
    assert!(stdout.contains("X_pca"), "missing 'X_pca'\n{}", stdout);
    assert!(stdout.contains("(2700, 30)"), "missing '(2700, 30)'\n{}", stdout);
}

#[test]
fn test_inspect_h5ad() {
    if !fixture_exists("pbmc3k_reference.h5ad") {
        return;
    }

    let output = Command::new(binary_path())
        .arg("inspect")
        .arg(golden_path("pbmc3k_reference.h5ad"))
        .output()
        .expect("failed to run scx inspect");

    assert_eq!(output.status.code(), Some(0), "expected exit code 0");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Format : H5AD"), "missing 'Format : H5AD'\n{}", stdout);
    assert!(stdout.contains("Shape  : 2700 obs × 13714 vars"), "missing shape\n{}", stdout);
    assert!(stdout.contains("nCount_RNA"), "missing 'nCount_RNA'\n{}", stdout);
    assert!(stdout.contains("X_pca"), "missing 'X_pca'\n{}", stdout);
}

#[test]
fn test_inspect_hlca() {
    if !fixture_exists("hlca_core.h5ad") {
        return;
    }

    let output = Command::new(binary_path())
        .arg("inspect")
        .arg(golden_path("hlca_core.h5ad"))
        .output()
        .expect("failed to run scx inspect");

    assert_eq!(output.status.code(), Some(0), "expected exit code 0");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Format : H5AD"), "missing 'Format : H5AD'\n{}", stdout);
    assert!(stdout.contains("584944 obs"), "missing '584944 obs'\n{}", stdout);
}
