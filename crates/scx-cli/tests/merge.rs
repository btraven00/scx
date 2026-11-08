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
//
// We also serialise CLI subprocess calls under the same lock: HDF5 file
// locking on CI container filesystems intermittently returns EAGAIN when
// multiple processes touch HDF5 files concurrently, even on disjoint paths.
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
        SparseMatrixMeta, UnsArray, UnsData, UnsTable, VarTable, Varm, UNS_TYPED_MIN,
    },
    stream::{DatasetReader, DatasetWriter},
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Binary / path helpers
// ---------------------------------------------------------------------------

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_scx"))
}

fn scx(args: &[&str]) -> std::process::Output {
    with_hdf5(|| {
        Command::new(binary_path())
            .args(args)
            .output()
            .expect("failed to run scx")
    })
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
    futures::executor::block_on(async {
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
    futures::executor::block_on(async {
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: extra_cols
                .into_iter()
                .map(|(name, data)| Column::new(name.to_string(), data))
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
    futures::executor::block_on(async {
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
        writer
            .begin_sparse("layers", layer_name, &meta)
            .await
            .unwrap();
        let layer_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr,
                indices,
                data: TypedVec::F32(data),
            },
        };
        writer.write_sparse_chunk(&layer_chunk).await.unwrap();
        writer.end_sparse().await.unwrap();
    });
}

fn make_h5ad_with_obsm(path: &Path, n_obs: usize, n_vars: usize, obsm_name: &str, n_dims: usize) {
    futures::executor::block_on(async {
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
            DenseMatrix {
                shape: (n_obs, n_dims),
                data: mat_data,
            },
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

fn make_h5ad_with_obsp(path: &Path, n_obs: usize, n_vars: usize, obsp_name: &str) {
    futures::executor::block_on(async {
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
        writer
            .write_x_chunk(&diag_chunk(n_obs, n_vars))
            .await
            .unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        // Append a square obsp (n_obs × n_obs) with diagonal connectivity.
        let mut writer = H5AdWriter::open_for_append(path).unwrap();
        let (indptr, indices, data) = diag_sparse(n_obs, n_obs);
        let meta = SparseMatrixMeta {
            name: obsp_name.to_string(),
            shape: (n_obs, n_obs),
            indptr: indptr.clone(),
        };
        writer.begin_sparse("obsp", obsp_name, &meta).await.unwrap();
        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_obs),
                indptr,
                indices,
                data: TypedVec::F32(data),
            },
        };
        writer.write_sparse_chunk(&chunk).await.unwrap();
        writer.end_sparse().await.unwrap();
    });
}

fn make_h5ad_with_uns(path: &Path, n_obs: usize, n_vars: usize) {
    futures::executor::block_on(async {
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
            columns: vec![],
        };
        // Large arrays are carried typed: one under a JSON entry, one alone.
        let big = |v: f32| UnsArray {
            shape: vec![UNS_TYPED_MIN],
            data: UnsData::F32(vec![v; UNS_TYPED_MIN]),
        };
        let uns = UnsTable {
            raw: serde_json::json!({
                "neighbors": { "method": "umap", "n_neighbors": 15 }
            }),
            arrays: [
                ("neighbors/dist".to_string(), big(0.5)),
                ("image".to_string(), big(2.0)),
            ]
            .into(),
        };
        let mut writer = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&uns).await.unwrap();
        writer.write_varm(&Varm::default()).await.unwrap();
        writer
            .write_x_chunk(&diag_chunk(n_obs, n_vars))
            .await
            .unwrap();
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
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr,
            indices,
            data: TypedVec::F32(data),
        },
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
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
            &patch,
            6,
            4,
            vec![(
                "leiden",
                ColumnData::Categorical {
                    codes: vec![0, 1, 0, 1, 2, 0],
                    levels: vec!["A".into(), "B".into(), "C".into()],
                    ordered: false,
                },
            )],
        );
    });

    assert_success(&scx(&[
        "merge",
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:obs/leiden", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:obsm/X_pca", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let obsm = reader.obsm().await.unwrap();
            assert!(obsm.map.contains_key("X_pca"), "obsm 'X_pca' missing");
            assert_eq!(obsm.map["X_pca"].shape, (8, 3));
        });
    });
}

#[test]
fn test_merge_create_obsp() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 10, 8);
        make_h5ad_with_obsp(&patch, 10, 8, "connectivities");
    });

    assert_success(&scx(&[
        "merge",
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:obsp/connectivities", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let metas = reader.obsp_metas().await.unwrap();
            let m = metas
                .iter()
                .find(|m| m.name == "connectivities")
                .expect("obsp 'connectivities' missing from merged output");
            assert_eq!(m.shape, (10, 10));
        });
    });
}

#[test]
fn test_merge_create_uns() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_uns(&patch, 6, 4);
    });

    assert_success(&scx(&[
        "merge",
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:uns/neighbors", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let uns = reader.uns().await.unwrap();
            let neighbors = uns
                .raw
                .get("neighbors")
                .expect("uns 'neighbors' missing from merged output");
            assert!(
                neighbors.get("method").is_some(),
                "nested uns key not preserved: {neighbors:?}"
            );
            assert_eq!(uns.arrays["neighbors/dist"].data.at(0), 0.5);
            assert!(!uns.arrays.contains_key("image"), "unpatched key copied");
        });
    });

    // An entry that is only a typed array patches too.
    assert_success(&scx(&[
        "merge",
        "--into",
        out.to_str().unwrap(),
        "--patch",
        &format!("{}:uns/image", patch.display()),
    ]));
    with_hdf5(|| {
        futures::executor::block_on(async {
            let uns = H5AdReader::open(&out, 5).unwrap().uns().await.unwrap();
            assert_eq!(uns.arrays["image"].shape, [UNS_TYPED_MIN]);
            assert_eq!(uns.arrays["image"].data.at(7), 2.0);
        });
    });
}

/// A patch can come from any format scx reads, not only h5ad: here obsm
/// from an AnnData Zarr store (merge used to open every non-BPCells,
/// non-H5Seurat patch as h5ad).
#[cfg(feature = "zarr")]
#[test]
fn test_merge_patch_from_zarr() {
    let dir = TempDir::new().unwrap();
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/zarr");
    let out = dir.path().join("merged.h5ad");
    let base = fixtures.join("small.h5ad");
    let patch = fixtures.join("small_v2.zarr");

    assert_success(&scx(&[
        "merge",
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:obsm/X_pca", patch.display()),
        "--on-conflict",
        "overwrite",
        "--output",
        out.to_str().unwrap(),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut merged = H5AdReader::open(&out, 5).unwrap();
            let mut expect = H5AdReader::open(&base, 5).unwrap();
            assert_eq!(
                merged.obsm().await.unwrap().map["X_pca"].data,
                expect.obsm().await.unwrap().map["X_pca"].data
            );
        });
    });
}

#[test]
fn test_merge_varp_rejected() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let out = dir.path().join("merged.h5ad");

    with_hdf5(|| make_base_h5ad(&base, 5, 3));

    // varp is not carried by the streaming pipeline; the slot parse must reject it.
    assert_failure(&scx(&[
        "merge",
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:varp/foo", base.display()),
        "--output",
        out.to_str().unwrap(),
    ]));
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    // Append mode: output now has "normalized"; default policy=error → should fail.
    let second = scx(&[
        "merge",
        "--into",
        out.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    // Append with skip — output already has "normalized"; should succeed.
    assert_success(&scx(&[
        "merge",
        "--into",
        out.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--on-conflict",
        "skip",
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    assert_success(&scx(&[
        "merge",
        "--into",
        out.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--on-conflict",
        "overwrite",
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch1.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    assert_success(&scx(&[
        "merge",
        "--into",
        out.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/counts", patch2.display()),
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut reader = H5AdReader::open(&out, 5).unwrap();
            let metas = reader.layer_metas().await.unwrap();
            let names: Vec<&str> = metas.iter().map(|m| m.name.as_str()).collect();
            assert!(
                names.contains(&"normalized"),
                "layer 'normalized' missing after append"
            );
            assert!(
                names.contains(&"counts"),
                "layer 'counts' missing after append"
            );
        });
    });

    // Append keeps the first run's time in the sidecar and adds its own.
    let slots = &sidecar(&out)["slots"];
    for key in ["layers/normalized", "layers/counts"] {
        assert!(slots[key]["added_at"].is_string(), "{key} missing: {slots}");
    }
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--output",
        out.to_str().unwrap(),
        "--tag",
        "pipeline=test",
    ]));

    with_hdf5(|| {
        futures::executor::block_on(async {
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

fn sidecar(out: &Path) -> serde_json::Value {
    let p = format!("{}.prov.json", out.display());
    serde_json::from_slice(&std::fs::read(&p).expect("merge sidecar missing")).unwrap()
}

/// The same merge twice gives identical bytes: the wall-clock `added_at`
/// goes to the sidecar, not into uns.
#[test]
fn test_merge_output_is_byte_reproducible() {
    let dir = TempDir::new().unwrap();
    let base = dir.path().join("base.h5ad");
    let patch = dir.path().join("patch.h5ad");
    with_hdf5(|| {
        make_base_h5ad(&base, 6, 4);
        make_h5ad_with_layer(&patch, 6, 4, "normalized");
    });
    let merge = |name: &str| {
        let out = dir.path().join(name);
        assert_success(&scx(&[
            "merge",
            "--base",
            base.to_str().unwrap(),
            "--patch",
            &format!("{}:layers/normalized", patch.display()),
            "--output",
            out.to_str().unwrap(),
            "--tag",
            "b=2",
            "--tag",
            "a=1",
        ]));
        out
    };
    let (a, b) = (merge("a.h5ad"), merge("b.h5ad"));
    assert!(
        std::fs::read(&a).unwrap() == std::fs::read(&b).unwrap(),
        "two identical merges differ"
    );

    with_hdf5(|| {
        futures::executor::block_on(async {
            let uns = H5AdReader::open(&a, 5).unwrap().uns().await.unwrap();
            let slot = &uns.raw["scx_provenance"]["slots"]["layers/normalized"];
            assert!(slot.get("sha256").is_some(), "{slot}");
            assert!(slot.get("added_at").is_none(), "added_at leaked into uns");
        })
    });
    let at = &sidecar(&a)["slots"]["layers/normalized"]["added_at"];
    assert!(at.as_str().is_some_and(|s| s.ends_with('Z')), "{at}");
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
        "--base",
        base.to_str().unwrap(),
        "--patch",
        &format!("{}:layers/normalized", patch.display()),
        "--output",
        out.to_str().unwrap(),
    ]));

    let inspect_out = scx(&["inspect", out.to_str().unwrap()]);
    assert_success(&inspect_out);

    let stdout = String::from_utf8_lossy(&inspect_out.stdout);
    assert!(
        stdout.contains("base.path"),
        "inspect missing 'base.path'\n{stdout}"
    );
    assert!(
        stdout.contains("layers/normalized"),
        "inspect missing slot\n{stdout}"
    );
    let at = sidecar(&out)["slots"]["layers/normalized"]["added_at"].clone();
    assert!(
        stdout.contains(at.as_str().unwrap()),
        "inspect missing sidecar added_at\n{stdout}"
    );
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
            &input,
            5,
            3,
            vec![
                (
                    "n_counts",
                    ColumnData::Float(vec![100.0, 200.0, 150.0, 50.0, 300.0]),
                ),
                (
                    "cell_type",
                    ColumnData::Categorical {
                        codes: vec![0, 0, 1, 2, 1],
                        levels: vec!["T".into(), "B".into(), "NK".into()],
                        ordered: false,
                    },
                ),
            ],
        );
    });

    assert_success(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot",
        "obs",
        "--output",
        out_csv.to_str().unwrap(),
    ]));

    assert!(out_csv.exists(), "CSV output not created");
    let content = std::fs::read_to_string(&out_csv).unwrap();
    assert!(
        content.contains("index"),
        "CSV missing 'index' column\n{content}"
    );
    assert!(
        content.contains("n_counts"),
        "CSV missing 'n_counts'\n{content}"
    );
    assert!(
        content.contains("cell_type"),
        "CSV missing 'cell_type'\n{content}"
    );
    assert!(
        content.contains('T') || content.contains('B'),
        "CSV missing decoded categories\n{content}"
    );
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
        "--slot",
        "var",
        "--output",
        out_csv.to_str().unwrap(),
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
        "--slot",
        "obsm/X_pca",
        "--output",
        out_csv.to_str().unwrap(),
    ]));

    let content = std::fs::read_to_string(&out_csv).unwrap();
    assert!(
        content.contains("dim_0"),
        "obsm CSV missing 'dim_0'\n{content}"
    );
    assert!(
        content.contains("dim_1"),
        "obsm CSV missing 'dim_1'\n{content}"
    );
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
            &input,
            4,
            3,
            vec![("score", ColumnData::Float(vec![0.1, 0.2, 0.3, 0.4]))],
        );
    });

    assert_success(&scx(&[
        "export",
        input.to_str().unwrap(),
        "--slot",
        "obs",
        "--output",
        out_pq.to_str().unwrap(),
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
        "--slot",
        "obsp/nn",
        "--output",
        out_csv.to_str().unwrap(),
    ]));
}

// ---------------------------------------------------------------------------
// scx merge — every slot group, the YAML config, and the errors
// ---------------------------------------------------------------------------

/// Every slot group's patchable entry, as a merge selector.
const ALL_SLOTS: [&str; 7] = [
    "obs/score",
    "var/gene_score",
    "obsm/X_emb",
    "varm/PCs",
    "obsp/conn",
    "uns/note",
    "layers/norm",
];

/// An h5ad with one entry in every slot group; the obs/var/obsm/varm/uns
/// values are scaled by `k`, so two patches can be told apart.
fn make_full_patch(path: &Path, n_obs: usize, n_vars: usize, k: f64) {
    futures::executor::block_on(async {
        let col = |n: usize| ColumnData::Float((0..n).map(|i| i as f64 * k).collect());
        let dense = |rows: usize| DenseMatrix {
            shape: (rows, 2),
            data: (0..rows * 2).map(|i| i as f64 * k).collect(),
        };
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("cell_{i}")).collect(),
            columns: vec![Column::new("score", col(n_obs))],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("gene_{i}")).collect(),
            columns: vec![Column::new("gene_score", col(n_vars))],
        };
        let mut obsm = Embeddings::default();
        obsm.map.insert("X_emb".into(), dense(n_obs));
        let mut varm = Varm::default();
        varm.map.insert("PCs".into(), dense(n_vars));
        let mut w = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
        w.write_obs(&obs).await.unwrap();
        w.write_var(&var).await.unwrap();
        w.write_obsm(&obsm).await.unwrap();
        w.write_varm(&varm).await.unwrap();
        w.write_uns(&UnsTable {
            raw: serde_json::json!({ "note": k }),
            ..Default::default()
        })
        .await
        .unwrap();
        w.write_x_chunk(&diag_chunk(n_obs, n_vars)).await.unwrap();
        w.finalize().await.unwrap();
        drop(w);

        let mut w = H5AdWriter::open_for_append(path).unwrap();
        for (group, name, cols) in [("layers", "norm", n_vars), ("obsp", "conn", n_obs)] {
            let (indptr, indices, data) = diag_sparse(n_obs, cols);
            let meta = SparseMatrixMeta {
                name: name.into(),
                shape: (n_obs, cols),
                indptr: indptr.clone(),
            };
            w.begin_sparse(group, name, &meta).await.unwrap();
            w.write_sparse_chunk(&MatrixChunk {
                row_offset: 0,
                nrows: n_obs,
                data: SparseMatrixCSR {
                    shape: (n_obs, cols),
                    indptr,
                    indices,
                    data: TypedVec::F32(data),
                },
            })
            .await
            .unwrap();
            w.end_sparse().await.unwrap();
        }
    });
}

/// The scale `k` each patched slot of `out` carries (every slot must agree),
/// and that the layer and obsp are there.
fn patched_scale(out: &Path) -> f64 {
    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut r = H5AdReader::open(out, 5).unwrap();
            let float = |c: &ColumnData| match c {
                ColumnData::Float(v) => v[1],
                other => panic!("not a float column: {other:?}"),
            };
            let obs = r.obs().await.unwrap();
            let var = r.var().await.unwrap();
            let col =
                |cols: &[Column], n: &str| float(&cols.iter().find(|c| c.name == n).unwrap().data);
            let ks = [
                col(&obs.columns, "score"),
                col(&var.columns, "gene_score"),
                r.obsm().await.unwrap().map["X_emb"].data[1],
                r.varm().await.unwrap().map["PCs"].data[1],
                r.uns().await.unwrap().raw["note"].as_f64().unwrap(),
            ];
            assert!(ks.iter().all(|&k| k == ks[0]), "slots disagree: {ks:?}");
            let names =
                |m: Vec<SparseMatrixMeta>| m.into_iter().map(|m| m.name).collect::<Vec<_>>();
            assert_eq!(names(r.layer_metas().await.unwrap()), ["norm"]);
            assert_eq!(names(r.obsp_metas().await.unwrap()), ["conn"]);
            ks[0]
        })
    })
}

/// A YAML config drives a create-mode merge of every slot group, with tags;
/// appending the same slots then keeps them (skip) or replaces them with the
/// second patch's (overwrite), for every slot group.
#[test]
fn test_merge_every_slot_from_config_then_skip_and_overwrite() {
    let dir = TempDir::new().unwrap();
    let p = |n: &str| dir.path().join(n);
    with_hdf5(|| {
        make_base_h5ad(&p("base.h5ad"), 6, 4);
        make_full_patch(&p("one.h5ad"), 6, 4, 1.0);
        make_full_patch(&p("two.h5ad"), 6, 4, 2.0);
    });
    let slots = ALL_SLOTS.map(|s| format!("\"{s}\"")).join(", ");
    std::fs::write(
        p("merge.yaml"),
        format!(
            "base: {}\noutput: {}\non_conflict: error\npatches:\n  - source: {}\n    slots: [{slots}]\ntags:\n  run: config\n",
            p("base.h5ad").display(),
            p("out.h5ad").display(),
            p("one.h5ad").display(),
        ),
    )
    .unwrap();
    assert_success(&scx(&[
        "merge",
        "--config",
        p("merge.yaml").to_str().unwrap(),
    ]));
    assert_eq!(patched_scale(&p("out.h5ad")), 1.0);
    let tags = with_hdf5(|| {
        futures::executor::block_on(async {
            let uns = H5AdReader::open(p("out.h5ad"), 5)
                .unwrap()
                .uns()
                .await
                .unwrap();
            uns.raw["scx_provenance"]["tags"].clone()
        })
    });
    assert_eq!(tags, serde_json::json!({ "run": "config" }));

    let spec = format!("{}:{}", p("two.h5ad").display(), ALL_SLOTS.join(","));
    for (policy, want) in [("skip", 1.0), ("overwrite", 2.0)] {
        assert_success(&scx(&[
            "merge",
            "--into",
            p("out.h5ad").to_str().unwrap(),
            "--patch",
            &spec,
            "--on-conflict",
            policy,
        ]));
        assert_eq!(patched_scale(&p("out.h5ad")), want, "{policy}");
    }
    // The default policy refuses each slot group that already exists.
    for slot in ALL_SLOTS {
        let out = scx(&[
            "merge",
            "--into",
            p("out.h5ad").to_str().unwrap(),
            "--patch",
            &format!("{}:{slot}", p("two.h5ad").display()),
        ]);
        assert_failure(&out);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains(&format!("'{slot}' already exists")),
            "{slot}: {err}"
        );
    }
}

/// Requests that can't be merged fail with a message saying why.
#[test]
fn test_merge_rejects_bad_requests() {
    let dir = TempDir::new().unwrap();
    let p = |n: &str| dir.path().join(n).to_str().unwrap().to_string();
    with_hdf5(|| {
        make_base_h5ad(Path::new(&p("base.h5ad")), 6, 4);
        make_full_patch(Path::new(&p("patch.h5ad")), 6, 4, 1.0);
        make_full_patch(Path::new(&p("wide.h5ad")), 6, 5, 1.0);
        make_full_patch(Path::new(&p("tall.h5ad")), 7, 4, 1.0);
    });
    let yaml = |name: &str, body: &str| {
        std::fs::write(p(name), body).unwrap();
        p(name)
    };
    let create = |spec: &str| -> Vec<String> {
        [
            "merge",
            "--base",
            &p("base.h5ad"),
            "--output",
            &p("out.h5ad"),
            "--patch",
            spec,
        ]
        .map(String::from)
        .to_vec()
    };
    let with = |mut v: Vec<String>, extra: &[&str]| {
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let patch = p("patch.h5ad");
    let mut cases: Vec<(Vec<String>, &str)> = vec![
        (
            vec![
                "merge".into(),
                "--patch".into(),
                format!("{patch}:obs/score"),
            ],
            "specify either --base",
        ),
        (
            with(create(&format!("{patch}:obs/score")), &["--tag", "novalue"]),
            "--tag must be key=value",
        ),
        (
            with(
                create(&format!("{patch}:obs/score")),
                &["--on-conflict", "maybe"],
            ),
            "unknown --on-conflict value 'maybe'",
        ),
        (
            create(&format!("{patch}:bogus/x")),
            "unknown slot group 'bogus'",
        ),
        (create(&format!("{patch}:layers/")), "slot name is empty"),
        (create(&format!("{patch}:layers")), "must be 'group/name'"),
        (
            create(&format!("{}:layers/norm", p("wide.h5ad"))),
            "shape mismatch for layer 'norm'",
        ),
        (
            create(&format!("{}:obsp/conn", p("tall.h5ad"))),
            "shape mismatch for obsp 'conn'",
        ),
        (
            vec![
                "merge".into(),
                "--into".into(),
                p("base.h5ad"),
                "--patch".into(),
                format!("{patch}:obs/score"),
            ],
            "no scx_provenance",
        ),
        (
            vec!["merge".into(), "--config".into(), p("absent.yaml")],
            "cannot read config",
        ),
        (
            vec![
                "merge".into(),
                "--config".into(),
                yaml("bad.yaml", "patches: [unclosed"),
            ],
            "invalid config",
        ),
        (
            vec![
                "merge".into(),
                "--config".into(),
                yaml(
                    "slot.yaml",
                    &format!(
                "base: {}\noutput: {}\npatches:\n  - source: {patch}\n    slots: [\"nope\"]\n",
                p("base.h5ad"), p("out.h5ad")),
                ),
            ],
            "invalid slot 'nope'",
        ),
        (
            vec![
                "merge".into(),
                "--config".into(),
                yaml(
                    "both.yaml",
                    &format!(
                        "base: {}\noutput: {}\ninto: {}\n",
                        p("base.h5ad"),
                        p("out.h5ad"),
                        p("out.h5ad")
                    ),
                ),
            ],
            "mutually exclusive",
        ),
    ];
    for (slot, what) in [
        ("layers/x", "layer 'x' not found"),
        ("obs/x", "column 'x' not found in patch obs"),
        ("var/x", "column 'x' not found in patch var"),
        ("obsm/x", "obsm entry 'x' not found"),
        ("varm/x", "varm entry 'x' not found"),
        ("obsp/x", "obsp 'x' not found"),
        ("uns/x", "uns entry 'x' not found"),
    ] {
        cases.push((create(&format!("{patch}:{slot}")), what));
    }
    for (args, want) in cases {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let out = scx(&args);
        assert_failure(&out);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(want), "{args:?}: want '{want}' in\n{err}");
        let _ = std::fs::remove_file(p("out.h5ad"));
    }
}
