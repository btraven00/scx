//! h5ad -> h5seurat -> scx read, with both h5seurat writers.
//!
//! Regressions from a CELLxGENE corpus check:
//! - obsm keys came back as `X_` + the lowercased reduction name
//!   (`X_scANVI` -> `X_scanvi`, `spatial` -> `X_spatial`);
//! - an ordered categorical came back unordered.

use std::path::Path;
use std::process::Command;

use scx_core::dtype::{DataType, TypedVec};
use scx_core::h5ad::H5AdWriter;
use scx_core::ir::{
    Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, VarTable,
};
use scx_core::stream::DatasetWriter;

const KEYS: [&str; 3] = ["X_scANVI", "spatial", "X_umap"];

async fn write_source(path: &Path) {
    let (n_obs, n_vars) = (3, 2);
    let mut w = H5AdWriter::create(path, n_obs, n_vars, DataType::F32).unwrap();
    let ordered = ColumnData::Categorical {
        codes: vec![2, 0, 1],
        levels: vec!["low".into(), "mid".into(), "high".into()],
        ordered: true,
        level_type: scx_core::ir::LevelType::Str,
    };
    w.write_obs(&ObsTable {
        index: vec!["c0".into(), "c1".into(), "c2".into()],
        columns: vec![Column::new("grade", ordered)],
    })
    .await
    .unwrap();
    w.write_var(&VarTable {
        index: vec!["g0".into(), "g1".into()],
        columns: vec![],
    })
    .await
    .unwrap();
    let mut obsm = Embeddings::default();
    for (i, k) in KEYS.iter().enumerate() {
        let data = (0..n_obs * 2).map(|v| (v + 10 * i) as f64).collect();
        obsm.map.insert(
            k.to_string(),
            DenseMatrix {
                shape: (n_obs, 2),
                data,
            },
        );
    }
    w.write_obsm(&obsm).await.unwrap();
    w.write_x_chunk(&MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr: vec![0, 1, 2, 3],
            indices: vec![0, 1, 0],
            data: TypedVec::F32(vec![1.0, 2.0, 3.0]),
        },
    })
    .await
    .unwrap();
    w.finalize().await.unwrap();
}

#[test]
fn h5seurat_keeps_obsm_keys_and_ordered_categoricals() {
    futures::executor::block_on(check());
}

async fn check() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.h5ad");
    write_source(&src).await;
    for extra in [&[][..], &["--dgcmatrix"][..]] {
        let out = dir.path().join(format!("out{}.h5seurat", extra.len()));
        let res = Command::new(env!("CARGO_BIN_EXE_scx"))
            .arg("convert")
            .arg(&src)
            .arg(&out)
            .args(extra)
            .output()
            .unwrap();
        assert!(
            res.status.success(),
            "{}",
            String::from_utf8_lossy(&res.stderr)
        );

        let mut r = scx_core::open(out.to_str().unwrap(), &scx_core::OpenOptions::new(10))
            .await
            .unwrap();
        let mut keys: Vec<String> = r.obsm().await.unwrap().map.into_keys().collect();
        keys.sort();
        let mut want: Vec<String> = KEYS.iter().map(|k| k.to_string()).collect();
        want.sort();
        assert_eq!(keys, want, "obsm keys {extra:?}");

        let obs = r.obs().await.unwrap();
        let grade = obs.columns.iter().find(|c| c.name == "grade").unwrap();
        assert!(
            matches!(&grade.data, ColumnData::Categorical { ordered: true, levels, .. }
                if levels == &["low", "mid", "high"]),
            "grade {extra:?}: {:?}",
            grade.data
        );
    }
}
