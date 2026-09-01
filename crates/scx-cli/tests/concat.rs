//! Integration tests for `scx concat`.
//!
//! Builds small synthetic h5ad fixtures with deliberately mismatched gene sets,
//! runs the `scx` binary, and reads the result back with `H5AdReader`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use futures::StreamExt;
use scx_core::{
    dtype::{DataType, TypedVec},
    h5ad::{H5AdReader, H5AdWriter},
    ir::{Column, ColumnData, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta, VarTable},
    stream::{DatasetReader, DatasetWriter},
};
use tempfile::TempDir;

// Same rationale as tests/merge.rs: HDF5 global state is not thread-safe, and
// CI container filesystems intermittently fail concurrent HDF5 file locking.
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn with_hdf5<T>(f: impl FnOnce() -> T) -> T {
    let _guard = HDF5_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    f()
}

fn scx(args: &[&str]) -> std::process::Output {
    with_hdf5(|| {
        Command::new(PathBuf::from(env!("CARGO_BIN_EXE_scx")))
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

/// Write an h5ad with the given cell names, gene names, dense rows (as
/// `(gene position, value)` pairs) and one string obs column.
fn make_h5ad(path: &Path, cells: &[&str], genes: &[&str], rows: &[Vec<(u32, f32)>], tag: &str) {
    with_hdf5(|| {
        futures::executor::block_on(async {
            let obs = ObsTable {
                index: cells.iter().map(|s| s.to_string()).collect(),
                columns: vec![Column {
                    name: "batch".into(),
                    data: ColumnData::String(vec![tag.to_string(); cells.len()]),
                }],
            };
            let var = VarTable {
                index: genes.iter().map(|s| s.to_string()).collect(),
                columns: vec![],
            };
            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            for row in rows {
                for &(g, v) in row {
                    indices.push(g);
                    data.push(v);
                }
                indptr.push(indices.len() as u64);
            }
            let mut w = H5AdWriter::create(path, cells.len(), genes.len(), DataType::F32).unwrap();
            w.write_obs(&obs).await.unwrap();
            w.write_var(&var).await.unwrap();
            w.write_x_chunk(&MatrixChunk {
                row_offset: 0,
                nrows: cells.len(),
                data: SparseMatrixCSR {
                    shape: (cells.len(), genes.len()),
                    indptr,
                    indices,
                    data: TypedVec::F32(data),
                },
            })
            .await
            .unwrap();
            w.finalize().await.unwrap();
        })
    });
}

/// Append a sparse layer holding `2 * X` to an existing h5ad.
fn add_doubled_layer(
    path: &Path,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    rows: &[Vec<(u32, f32)>],
) {
    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            for row in rows {
                for &(g, v) in row {
                    indices.push(g);
                    data.push(v * 2.0);
                }
                indptr.push(indices.len() as u64);
            }
            let meta = SparseMatrixMeta {
                name: name.to_string(),
                shape: (n_obs, n_vars),
                indptr: indptr.clone(),
            };
            let mut w = H5AdWriter::open_for_append(path).unwrap();
            w.begin_sparse("layers", name, &meta).await.unwrap();
            w.write_sparse_chunk(&MatrixChunk {
                row_offset: 0,
                nrows: n_obs,
                data: SparseMatrixCSR {
                    shape: (n_obs, n_vars),
                    indptr,
                    indices,
                    data: TypedVec::F32(data),
                },
            })
            .await
            .unwrap();
            w.end_sparse().await.unwrap();
        })
    });
}

/// Read an h5ad back as (obs, var, dense X).
fn read_back(path: &Path) -> (ObsTable, VarTable, Vec<Vec<f64>>) {
    read_back_slot(path, None)
}

/// Same, reading `layers/<name>` in place of X when `layer` is set.
fn read_back_slot(path: &Path, layer: Option<&str>) -> (ObsTable, VarTable, Vec<Vec<f64>>) {
    with_hdf5(|| {
        futures::executor::block_on(async {
            let mut r = H5AdReader::open_layer(path, 2, layer).unwrap();
            let (n_obs, n_vars) = r.shape();
            let obs = r.obs().await.unwrap();
            let var = r.var().await.unwrap();
            let mut dense = vec![vec![0.0; n_vars]; n_obs];
            let mut row = 0usize;
            let mut stream = r.x_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.unwrap();
                let vals = chunk.data.data.to_f64();
                for i in 0..chunk.nrows {
                    let (s, e) = (
                        chunk.data.indptr[i] as usize,
                        chunk.data.indptr[i + 1] as usize,
                    );
                    for p in s..e {
                        dense[row][chunk.data.indices[p] as usize] = vals[p];
                    }
                    row += 1;
                }
            }
            (obs, var, dense)
        })
    })
}

fn fixtures(dir: &Path) -> (PathBuf, PathBuf) {
    let a = dir.join("a.h5ad");
    let b = dir.join("b.h5ad");
    // a: genes g0,g1,g2 — cell_0 = [1,2,0], cell_1 = [0,0,3]
    make_h5ad(
        &a,
        &["cell_0", "cell_1"],
        &["g0", "g1", "g2"],
        &[vec![(0, 1.0), (1, 2.0)], vec![(2, 3.0)]],
        "a",
    );
    // b: genes g2,g1,g3 (reordered, g0 missing, g3 new) — cell_0 = g2:4, g3:5
    make_h5ad(
        &b,
        &["cell_0"],
        &["g2", "g1", "g3"],
        &[vec![(0, 4.0), (2, 5.0)]],
        "b",
    );
    (a, b)
}

#[test]
fn concat_outer_join_realigns_genes_and_labels_cells() {
    let dir = TempDir::new().unwrap();
    let (a, b) = fixtures(dir.path());
    let out = dir.path().join("out.h5ad");

    let res = scx(&[
        "concat",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--join",
        "outer",
        "--label",
        "sample",
        "--index-unique",
        "-",
    ]);
    assert_success(&res);

    let (obs, var, x) = read_back(&out);
    assert_eq!(var.index, vec!["g0", "g1", "g2", "g3"]);
    assert_eq!(obs.index, vec!["cell_0-a", "cell_1-a", "cell_0-b"]);
    assert_eq!(x[0], vec![1.0, 2.0, 0.0, 0.0]);
    assert_eq!(x[1], vec![0.0, 0.0, 3.0, 0.0]);
    // b's genes are reordered against the output axis: g2 -> col 2, g3 -> col 3.
    assert_eq!(x[2], vec![0.0, 0.0, 4.0, 5.0]);

    let label = obs.columns.iter().find(|c| c.name == "sample").unwrap();
    match &label.data {
        ColumnData::Categorical { codes, levels } => {
            assert_eq!(levels, &vec!["a".to_string(), "b".to_string()]);
            assert_eq!(codes, &vec![0, 0, 1]);
        }
        other => panic!("expected categorical label, got {other:?}"),
    }
    // obs columns present in every input are carried through.
    let batch = obs.columns.iter().find(|c| c.name == "batch").unwrap();
    assert_eq!(
        batch.data.len(),
        3,
        "batch column must span all concatenated cells"
    );
}

#[test]
fn concat_inner_join_keeps_only_shared_genes() {
    let dir = TempDir::new().unwrap();
    let (a, b) = fixtures(dir.path());
    let out = dir.path().join("inner.h5ad");

    let res = scx(&[
        "concat",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert_success(&res);

    let (obs, var, x) = read_back(&out);
    assert_eq!(var.index, vec!["g1", "g2"]);
    assert_eq!(obs.index.len(), 3);
    assert_eq!(x[0], vec![2.0, 0.0]); // g0=1.0 dropped by the inner join
    assert_eq!(x[1], vec![0.0, 3.0]);
    assert_eq!(x[2], vec![0.0, 4.0]); // g3=5.0 dropped
}

#[test]
fn concat_outer_zero_fills_a_layer_missing_from_one_input() {
    let dir = TempDir::new().unwrap();
    let (a, b) = fixtures(dir.path());
    // Only b carries the layer; a's rows must come out as explicit zeros.
    add_doubled_layer(&b, "norm", 1, 3, &[vec![(0, 4.0), (2, 5.0)]]);
    let out = dir.path().join("layer.h5ad");

    let res = scx(&[
        "concat",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--join",
        "outer",
    ]);
    assert_success(&res);

    let (_, var, norm) = read_back_slot(&out, Some("norm"));
    assert_eq!(var.index, vec!["g0", "g1", "g2", "g3"]);
    assert_eq!(norm[0], vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(norm[1], vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(norm[2], vec![0.0, 0.0, 8.0, 10.0]);
}
