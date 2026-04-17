use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/scx")
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
fn test_convert_h5ad_to_h5seurat_bpcells_default() {
    if !fixture_exists("pbmc3k_reference.h5ad") {
        return;
    }

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let output = Command::new(binary_path())
        .arg("convert")
        .arg(golden_path("pbmc3k_reference.h5ad"))
        .arg(&out)
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
        stdout.contains("Shape  : 2700 obs × 13714 vars"),
        "missing expected shape\n{}",
        stdout
    );
    assert!(stdout.contains("obs ("), "missing obs section\n{}", stdout);
    assert!(stdout.contains("var ("), "missing var section\n{}", stdout);
    assert!(
        stdout.contains("obsm (2 keys):"),
        "missing expected obsm key count\n{}",
        stdout
    );
    assert!(
        stdout.contains("X_pca"),
        "missing X_pca after BPCells conversion\n{}",
        stdout
    );
    assert!(
        stdout.contains("X_umap"),
        "missing X_umap after BPCells conversion\n{}",
        stdout
    );
}

#[test]
fn test_convert_h5seurat_to_h5seurat_bpcells_default() {
    if !fixture_exists("pbmc3k.h5seurat") {
        return;
    }

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let output = Command::new(binary_path())
        .arg("convert")
        .arg(golden_path("pbmc3k.h5seurat"))
        .arg(&out)
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
        .expect("failed to run scx inspect on default BPCells re-encoded output");

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
        stdout.contains("Shape  : 2700 obs × 13714 vars"),
        "missing expected shape\n{}",
        stdout
    );
    assert!(stdout.contains("obs ("), "missing obs section\n{}", stdout);
    assert!(stdout.contains("var ("), "missing var section\n{}", stdout);
    assert!(
        stdout.contains("obsm (2 keys):"),
        "missing expected obsm key count\n{}",
        stdout
    );
    assert!(
        stdout.contains("X_pca"),
        "missing X_pca after BPCells re-encode\n{}",
        stdout
    );
    assert!(
        stdout.contains("X_umap"),
        "missing X_umap after BPCells re-encode\n{}",
        stdout
    );
}

#[test]
fn test_convert_h5ad_to_h5seurat_dgcmatrix_opt_out() {
    if !fixture_exists("pbmc3k_reference.h5ad") {
        return;
    }

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let output = Command::new(binary_path())
        .arg("convert")
        .arg(golden_path("pbmc3k_reference.h5ad"))
        .arg(&out)
        .arg("--dgcmatrix")
        .output()
        .expect("failed to run scx convert --dgcmatrix");

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
        .expect("failed to run scx inspect on dgCMatrix output");

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
        stdout.contains("Shape  : 2700 obs × 13714 vars"),
        "missing expected shape\n{}",
        stdout
    );
    assert!(stdout.contains("obs ("), "missing obs section\n{}", stdout);
    assert!(stdout.contains("var ("), "missing var section\n{}", stdout);
}
