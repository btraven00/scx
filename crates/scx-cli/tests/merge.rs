//! Integration tests for `scx merge` and `scx export`.
//!
//! Tests build synthetic h5ad fixtures in temp directories, invoke the
//! `scx` binary via `std::process::Command`, then read back the output with
//! `H5AdReader` to verify correctness.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

// HDF5 has global state that is not thread-safe without the --enable-threadsafe
// compile flag. Serialise all in-process HDF5 calls (fixture building and
// read-back verification) so parallel test threads don't corrupt each other.
// CLI subprocess calls don't need the lock (they're separate processes).
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn with_hdf5<T>(f: impl FnOnce() -> T) -> T {
    let _guard = HDF5_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    f()
}

use scx_core::{
    dtype::{DataType, TypedVec},
    h5ad::{H5AdReader, H5AdWriter},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
    },
    stream::{DatasetReader, DatasetWriter},
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Binary / path helpers
// ---------------------------------------------------------------------------

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/scx")
}

fn scx(args: &[&str]) -> std::process::Output {
    Command::new(binary_path())
        .args(args)
        .output()
        .expect("failed to run scx")
}

fn assert_success(out: &std::process::Output) {
    assert_eq!(
        out.status.code(),
        Some(0),
        "expected exit 0\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

fn assert_failure(out: &std::process::Output) {
    assert_ne!(
        out.status.code(),
        Some(0),
        "expected non-zero exit\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

// ---------------------------------------------------------------------------
// Synthetic h5ad builders — all run inside with_hdf5()
// ---------------------------------------------------------------------------

fn make_base_h5ad(path: &Path, n_obs: usize, n_vars: usize) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
            columns: vec![],
        };
        let mut writer = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();
        writer.write_varm(&Varm::default()).await.unwrap();
        let chunk = diag_chunk(n_obs, n_vars);
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
    });
}

fn make_obs_column_h5ad(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    extra_cols: Vec<(&str, ColumnData)>,
) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: extra_cols
                .into_iter()
                .map(|(name, data)| Column { name: name.to_string(), data })
                .collect(),
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
            columns: vec![],
        };
        let mut writer = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();
        writer.write_varm(&Varm::default()).await.unwrap();
        let chunk = diag_chunk(n_obs, n_vars);
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
    });
}

fn make_h5ad_with_layer(path: &Path, n_obs: usize, n_vars: usize, layer_name: &str) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
            columns: vec![],
        };
        let mut writer = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();
        writer.write_varm(&Varm::default()).await.unwrap();
        let chunk = diag_chunk(n_obs, n_vars);
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        // Reopen in append mode to add the layer.
        let mut writer = H5AdWriter::open_for_append(path).unwrap();
        let (indptr, indices, data) = diag_sparse(n_obs, n_vars);
        let meta = SparseMatrixMeta {
            name: layer_name.to_string(),
            shape: (n_obs, n_vars),
            indptr: indptr.clone(),
        };
        writer.begin_sparse("layers", layer_name, &meta).await.unwrap();
        let layer_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR { shape: (n_obs, n_vars), indptr, indices, data: TypedVec::F32(data) },
        };
        writer.write_sparse_chunk(&layer_chunk).await.unwrap();
        writer.end_sparse().await.unwrap();
    });
}

fn make_h5ad_with_obsm(path: &Path, n_obs: usize, n_vars: usize, obsm_name: &str, n_dims: usize) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
            columns: vec![],
        };
        let mut obsm_data = Embeddings::default();
        let mat_data: Vec<f64> = (0..n_obs * n_dims).map(|i| i as f64 * 0.1).collect();
        obsm_data.map.insert(
            obsm_name.to_string(),
            DenseMatrix { shape: (n_obs, n_dims), data: mat_data },
        );
        let mut writer = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&obsm_data).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();
        writer.write_varm(&Varm::default()).await.unwrap();
        let chunk = diag_chunk(n_obs, n_vars);
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
    });
}

/// Diagonal sparse matrix: cell_i has exactly one nonzero at var_i (for i < min(n,m)).
fn diag_sparse(n_obs: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let nnz = n_obs.min(n_vars);
    let mut indptr: Vec<u64> = vec![0];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for r in 0..n_obs {
        if r < nnz {
            indices.push(r as u32);
            data.push(1.0);
            indptr.push(indptr.last().unwrap() + 1);
        } else {
            indptr.push(*indptr.last().unwrap());
        }
    }
    (indptr, indices, data)
}

fn diag_chunk(n_obs: usize, n_vars: usize) -> MatrixChunk {
    let (indptr, indices, data) = diag_sparse(n_obs, n_vars);
    MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR { shape: (n_obs, n_vars), indptr, indices, data: TypedVec::F32(data) },
    }
}

// ---------------------------------------------------------------------------
// scx merge — create mode
// ---------------------------------------------------------------------------

#[test]
fn test_merge_create_layer() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 10, 8);
        make_h5ad_with_layer(&patch, 10, 8, "normalized");
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            assert_eq!(reader.shape(), (10, 8));
            let metas = reader.layer_metas().await.unwrap();
            assert!(
                metas.iter().any(|m| m.name == "normalized"),
                "layer 'normalized' missing from merged output"
            );
        });
    });
}

#[test]
fn test_merge_create_obs_column() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_obs_column_h5ad(
            &patch, 6, 4,
            vec![("leiden", ColumnData::Categorical {
                codes: vec![0, 1, 0, 1, 2, 0],
                levels: vec!["A".into(), "B".into(), "C".into()],
            })],
        );
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:obs/leiden", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let obs = reader.obs().await.unwrap();
            assert!(
                obs.columns.iter().any(|c| c.name == "leiden"),
                "obs column 'leiden' missing from merged output"
            );
        });
    });
}

#[test]
fn test_merge_create_obsm() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 8, 5);
        make_h5ad_with_obsm(&patch, 8, 5, "X_pca", 3);
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:obsm/X_pca", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let obsm = reader.obsm().await.unwrap();
            assert!(obsm.map.contains_key("X_pca"), "obsm 'X_pca' missing");
            assert_eq!(obsm.map["X_pca"].shape, (8, 3));
        });
    });
}

// ---------------------------------------------------------------------------
// scx merge — conflict policies
// ---------------------------------------------------------------------------

#[test]
fn test_merge_conflict_error_on_existing_layer() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_layer(&patch, 6, 4, "normalized");
    });

    // Create-mode: base → output (no layers) + patch "normalized" → success.
    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    // Append mode: output now has "normalized"; default policy=error → should fail.
    let second = scx(&[
        "merge",
        "--into", out.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
    ]);
    assert_failure(&second);
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(
        stderr.contains("already exists"),
        "expected 'already exists' in error: {stderr}"
    );
}

#[test]
fn test_merge_conflict_skip() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_layer(&patch, 6, 4, "normalized");
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    // Append with skip — output already has "normalized"; should succeed.
    assert_success(&scx(&[
        "merge",
        "--into", out.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--on-conflict", "skip",
    ]));
}

#[test]
fn test_merge_conflict_overwrite() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_layer(&patch, 6, 4, "normalized");
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    assert_success(&scx(&[
        "merge",
        "--into", out.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--on-conflict", "overwrite",
    ]));
}

// ---------------------------------------------------------------------------
// scx merge — append mode
// ---------------------------------------------------------------------------

#[test]
fn test_merge_append_adds_second_slot() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch1 = dir.path().join("patch1.h5ad");
    let patch2 = dir.path().join("patch2.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 8, 6);
        make_h5ad_with_layer(&patch1, 8, 6, "normalized");
        make_h5ad_with_layer(&patch2, 8, 6, "counts");
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch1.display()),
        "--output", out.to_str().unwrap(),
    ]));

    assert_success(&scx(&[
        "merge",
        "--into", out.to_str().unwrap(),
        "--patch", &format!("{}:layers/counts", patch2.display()),
    ]));

    with_hdf5(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let metas = reader.layer_metas().await.unwrap();
            let names: Vec<&str> = metas.iter().map(|m| m.name.as_str()).collect();
            assert!(names.contains(&"normalized"), "layer 'normalized' missing after append");
            assert!(names.contains(&"counts"), "layer 'counts' missing after append");
        });
    });
}

// ---------------------------------------------------------------------------
// scx merge — provenance
// ---------------------------------------------------------------------------

#[test]
fn test_merge_provenance_written_to_uns() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_layer(&patch, 6, 4, "normalized");
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--output", out.to_str().unwrap(),
        "--tag", "pipeline=test",
    ]));

    with_hdf5(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let uns = reader.uns().await.unwrap();
            let prov = uns
                .raw
                .as_object()
                .and_then(|o| o.get("scx_provenance"))
                .expect("scx_provenance missing from uns");
            assert!(prov.get("base").is_some(), "base anchor missing");
            assert!(prov.get("slots").is_some(), "slots map missing");
            let slots = prov["slots"].as_object().unwrap();
            assert!(
                slots.contains_key("layers/normalized"),
                "slot 'layers/normalized' missing from provenance"
            );
            let tags = prov.get("tags").and_then(|v| v.as_object()).unwrap();
            assert_eq!(
                tags.get("pipeline").and_then(|v| v.as_str()),
                Some("test"),
                "tag 'pipeline=test' not found"
            );
        });
    });
}

// ---------------------------------------------------------------------------
// scx inspect — shows provenance section
// ---------------------------------------------------------------------------

#[test]
fn test_inspect_shows_merge_provenance() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_layer(&patch, 6, 4, "normalized");
    });

    assert_success(&scx(&[
        "merge",
        "--base", base.to_str().unwrap(),
        "--patch", &format!("{}:layers/normalized", patch.display()),
        "--output", out.to_str().unwrap(),
    ]));

    let inspect_out = scx(&["inspect", out.to_str().unwrap()]);
    assert_success(&inspect_out);

    let stdout = String::from_utf8_lossy(&inspect_out.stdout);
    assert!(stdout.contains("base.path"), "inspect missing 'base.path'\n{stdout}");
    assert!(stdout.contains("layers/normalized"), "inspect missing slot\n{stdout}");
}

// ---------------------------------------------------------------------------
// scx export
// ---------------------------------------------------------------------------

#[test]
fn test_export_obs_csv() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.h5ad");
    let out_csv = dir.path().join("obs.csv");

    with_hdf5(|| {
        make_obs_column_h5ad(
            &input, 5, 3,
            vec![
                ("n_counts", ColumnData::Float(vec![100.0, 200.0, 150.0, 50.0, 300.0])),
                ("cell_type", ColumnData::Categorical {
                    codes: vec![0, 0, 1, 2, 1],
                    levels: vec!["T".into(), "B".into(), "NK".into()],
                }),
            ],
        );
    });

    assert_success(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot", "obs",
        "--output", out_csv.to_str().unwrap(),
    ]));

    assert!(out_csv.exists(), "CSV output not created");
    let content = std::fs::read_to_string(&out_csv).unwrap();
    assert!(content.contains("index"), "CSV missing 'index' column\n{content}");
    assert!(content.contains("n_counts"), "CSV missing 'n_counts'\n{content}");
    assert!(content.contains("cell_type"), "CSV missing 'cell_type'\n{content}");
    assert!(content.contains('T') || content.contains('B'), "CSV missing decoded categories\n{content}");
    let rows: Vec<&str> = content.lines().collect();
    assert_eq!(rows.len(), 6, "expected 6 lines (header + 5 rows)");
}

#[test]
fn test_export_var_csv() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.h5ad");
    let out_csv = dir.path().join("var.csv");

    with_hdf5(|| make_base_h5ad(&input, 5, 4));

    assert_success(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot", "var",
        "--output", out_csv.to_str().unwrap(),
    ]));

    let content = std::fs::read_to_string(&out_csv).unwrap();
    assert!(content.contains("index"), "var CSV missing 'index' column");
    let rows: Vec<&str> = content.lines().collect();
    assert_eq!(rows.len(), 5, "expected 5 lines (header + 4 vars)");
}

#[test]
fn test_export_obsm_csv() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.h5ad");
    let out_csv = dir.path().join("pca.csv");

    with_hdf5(|| make_h5ad_with_obsm(&input, 6, 3, "X_pca", 2));

    assert_success(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot", "obsm/X_pca",
        "--output", out_csv.to_str().unwrap(),
    ]));

    let content = std::fs::read_to_string(&out_csv).unwrap();
    assert!(content.contains("dim_0"), "obsm CSV missing 'dim_0'\n{content}");
    assert!(content.contains("dim_1"), "obsm CSV missing 'dim_1'\n{content}");
    let rows: Vec<&str> = content.lines().collect();
    assert_eq!(rows.len(), 7, "expected 7 lines (header + 6 cells)");
}

#[test]
fn test_export_parquet_creates_file() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.h5ad");
    let out_pq = dir.path().join("obs.parquet");

    with_hdf5(|| {
        make_obs_column_h5ad(
            &input, 4, 3,
            vec![("score", ColumnData::Float(vec![0.1, 0.2, 0.3, 0.4]))],
        );
    });

    assert_success(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot", "obs",
        "--output", out_pq.to_str().unwrap(),
    ]));

    assert!(out_pq.exists(), "Parquet output not created");
    let bytes = std::fs::read(&out_pq).unwrap();
    assert_eq!(&bytes[..4], b"PAR1", "output is not a valid Parquet file");
}

#[test]
fn test_export_unknown_slot_fails() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.h5ad");
    let out_csv = dir.path().join("out.csv");

    with_hdf5(|| make_base_h5ad(&input, 4, 3));

    assert_failure(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot", "obsp/nn",
        "--output", out_csv.to_str().unwrap(),
    ]));
}
