//! raw.var columns survive h5ad -> h5ad, h5ad -> Zarr and Zarr -> h5ad.
//!
//! Its own test binary: the source is written in-process with HDF5, and a
//! test in the same binary spawning `scx` meanwhile would inherit the open
//! file and its lock.

use std::path::Path;
use std::process::Command;

fn convert(input: &Path, out: &Path, extra: &[&str]) {
    let res = Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("convert")
        .arg(input)
        .arg(out)
        .args(extra)
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "{}",
        String::from_utf8_lossy(&res.stderr)
    );
}

/// An h5ad whose raw.var has columns, at `path`: 2 cells x 2 genes.
async fn write_raw_with_columns(path: &Path) {
    use scx_core::dtype::{DataType, TypedVec};
    use scx_core::ir::{Column, ColumnData, MatrixChunk, ObsTable, SparseMatrixCSR, VarTable};
    use scx_core::stream::DatasetWriter;
    let genes = vec!["g0".to_string(), "g1".to_string()];
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: 2,
        data: SparseMatrixCSR {
            shape: (2, 2),
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: TypedVec::F32(vec![1.0, 2.0]),
        },
    };
    let mut w = scx_core::h5ad::H5AdWriter::create(path, 2, 2, DataType::F32).unwrap();
    w.write_obs(&ObsTable {
        index: vec!["c0".into(), "c1".into()],
        columns: vec![],
    })
    .await
    .unwrap();
    w.write_var(&VarTable {
        index: genes.clone(),
        columns: vec![],
    })
    .await
    .unwrap();
    w.write_x_chunk(&chunk).await.unwrap();
    let raw_var = VarTable {
        index: genes,
        columns: vec![
            Column::new(
                "gene_ids",
                ColumnData::String(vec!["E0".into(), "E1".into()]),
            ),
            Column::new("highly_variable", ColumnData::Bool(vec![true, false])),
        ],
    };
    let meta = scx_core::ir::SparseMatrixMeta {
        name: "raw".into(),
        shape: (2, 2),
        indptr: vec![0, 1, 2],
    };
    assert!(w.begin_raw(&meta, &raw_var).await.unwrap());
    w.write_sparse_chunk(&chunk).await.unwrap();
    w.end_sparse().await.unwrap();
    w.finalize().await.unwrap();
}

/// raw.var of the AnnData at `path`, in Debug form.
fn raw_var(path: &Path) -> String {
    futures::executor::block_on(async {
        let mut r = scx_core::open(path.to_str().unwrap(), &scx_core::OpenOptions::new(100))
            .await
            .unwrap();
        format!("{:?}", r.raw_var().await.unwrap())
    })
}

/// Regression: raw.var kept only its index; its columns (CELLxGENE's
/// feature_name, highly_variable, ...) were dropped.
#[test]
fn raw_var_keeps_its_columns() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.h5ad");
    futures::executor::block_on(write_raw_with_columns(&src));
    let want = raw_var(&src);
    assert!(want.contains("highly_variable"), "{want}");
    let out = dir.path().join("out.h5ad");
    convert(&src, &out, &[]);
    assert_eq!(raw_var(&out), want, "h5ad");
    if cfg!(feature = "zarr") {
        let zarr = dir.path().join("out.zarr");
        convert(&src, &zarr, &[]);
        assert_eq!(raw_var(&zarr), want, "zarr");
        let back = dir.path().join("back.h5ad");
        convert(&zarr, &back, &[]);
        assert_eq!(raw_var(&back), want, "zarr -> h5ad");
    }
}
