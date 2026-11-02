//! Property-based round trips: random datasets through every writer, read
//! back with scx's readers, must come back as they went in.
//!
//! roundtrip.rs does this for the hand-made tiny fixtures; here proptest
//! generates the datasets (shapes down to 0 cells or 0 genes, X at any
//! sparsity, obs columns of every type, an embedding, any chunk size) and
//! shrinks a failure to a minimal case. Failing seeds are saved in
//! `roundtrip_prop.proptest-regressions` next to this file and replayed
//! first; commit it.
//!
//! `PROPTEST_CASES` sets the number of cases (default 32 per test).

#[path = "common/convert.rs"]
mod convert;

use convert::{dense_x, out_path, Out};
use futures::StreamExt;
use proptest::prelude::*;
use scx_core::dtype::{DataType, TypedVec};
use scx_core::ir::{
    Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, VarTable,
};
use scx_core::stream::{ChunkStream, DatasetReader};
use std::collections::BTreeSet;

/// A generated dataset; X is dense here, the reader streams it as CSR.
#[derive(Debug, Clone)]
struct Dataset {
    x: Vec<Vec<f32>>,
    obs: ObsTable,
    var: VarTable,
    obsm: Embeddings,
    chunk: usize,
}

impl Dataset {
    fn shape(&self) -> (usize, usize) {
        (self.obs.index.len(), self.var.index.len())
    }
}

/// An in-memory reader over a [`Dataset`].
struct Mem(Dataset);

#[async_trait::async_trait]
impl DatasetReader for Mem {
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }
    fn dtype(&self) -> DataType {
        DataType::F32
    }
    async fn obs(&mut self) -> scx_core::error::Result<ObsTable> {
        Ok(self.0.obs.clone())
    }
    async fn var(&mut self) -> scx_core::error::Result<VarTable> {
        Ok(self.0.var.clone())
    }
    async fn obsm(&mut self) -> scx_core::error::Result<Embeddings> {
        Ok(self.0.obsm.clone())
    }
    fn x_stream(&mut self) -> ChunkStream<'_> {
        let (n_obs, n_vars) = self.0.shape();
        let chunk = self.0.chunk.max(1);
        let chunks: Vec<_> = (0..n_obs)
            .step_by(chunk)
            .map(|start| {
                let rows = &self.0.x[start..(start + chunk).min(n_obs)];
                let (mut indptr, mut indices, mut data) = (vec![0u64], vec![], vec![]);
                for row in rows {
                    for (j, &v) in row.iter().enumerate() {
                        if v != 0.0 {
                            indices.push(j as u32);
                            data.push(v);
                        }
                    }
                    indptr.push(indices.len() as u64);
                }
                Ok(MatrixChunk {
                    row_offset: start,
                    nrows: rows.len(),
                    data: SparseMatrixCSR {
                        shape: (rows.len(), n_vars),
                        indptr,
                        indices,
                        data: TypedVec::F32(data),
                    },
                })
            })
            .collect();
        futures::stream::iter(chunks).boxed()
    }
}

/// `n` distinct names.
fn names(n: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::btree_set("[A-Za-z][A-Za-z0-9_.-]{0,7}", n)
        .prop_map(|s: BTreeSet<String>| s.into_iter().collect::<Vec<_>>())
        .prop_shuffle()
}

/// One obs column of `n` values, of any type.
fn column(n: usize) -> impl Strategy<Value = ColumnData> {
    prop_oneof![
        prop::collection::vec(-1e6f64..1e6, n).prop_map(ColumnData::Float),
        prop::collection::vec(any::<i32>(), n).prop_map(ColumnData::Int),
        prop::collection::vec(any::<i32>(), n)
            .prop_map(|v| ColumnData::Int64(v.into_iter().map(i64::from).collect())),
        prop::collection::vec("[a-z ]{0,6}", n).prop_map(ColumnData::String),
        prop::collection::vec(any::<bool>(), n).prop_map(ColumnData::Bool),
        (names(3), prop::collection::vec(0u32..3, n), any::<bool>()).prop_map(
            |(levels, codes, ordered)| ColumnData::Categorical {
                codes,
                levels,
                ordered,
            }
        ),
    ]
}

prop_compose! {
    fn dataset()(n_obs in 0usize..=25, n_vars in 0usize..=12)(
        x in prop::collection::vec(
            prop::collection::vec(prop_oneof![3 => Just(0.0f32), 1 => (1u16..1000).prop_map(f32::from)], n_vars),
            n_obs,
        ),
        obs_names in names(n_obs),
        var_names in names(n_vars),
        cols in prop::collection::vec(
            (column(n_obs), prop::option::of(prop::collection::vec(prop::bool::weighted(0.8), n_obs))),
            0..=3,
        ),
        pca in prop::option::of(prop::collection::vec(-100f64..100.0, n_obs * 2)),
        chunk in 1usize..=(n_obs + 1),
    ) -> Dataset {
        let columns = cols
            .into_iter()
            .enumerate()
            .map(|(i, (data, mask))| Column { name: format!("col{i}"), data, mask })
            .collect();
        let mut obsm = Embeddings::default();
        let n_obs = obs_names.len();
        if let Some(data) = pca.filter(|_| n_obs > 0) {
            obsm.map.insert("X_pca".into(), DenseMatrix { shape: (n_obs, 2), data });
        }
        Dataset {
            x,
            obs: ObsTable { index: obs_names, columns },
            var: VarTable { index: var_names, columns: vec![] },
            obsm,
            chunk,
        }
    }
}

/// Cell `i` of `col` as text, so storage types that hold the same value
/// (int32/int64, string/categorical) compare equal.
fn cell(col: &ColumnData, i: usize) -> String {
    match col {
        ColumnData::Int(v) => v[i].to_string(),
        ColumnData::Int64(v) => v[i].to_string(),
        ColumnData::Float(v) => format!("{:?}", v[i]),
        ColumnData::String(v) => v[i].clone(),
        ColumnData::Bool(v) => v[i].to_string(),
        ColumnData::Categorical { codes, levels, .. } => levels[codes[i] as usize].clone(),
    }
}

/// The column's cells as text, `na(column)` where missing.
fn column_text(
    obs: &ObsTable,
    name: &str,
    na: impl Fn(&ColumnData) -> String,
) -> Option<Vec<String>> {
    let col = obs.columns.iter().find(|c| c.name == name)?;
    let present = |i: usize| col.mask.as_ref().is_none_or(|m| m[i]);
    let na = na(&col.data);
    Some(
        (0..obs.index.len())
            .map(|i| {
                if present(i) {
                    cell(&col.data, i)
                } else {
                    na.clone()
                }
            })
            .collect(),
    )
}

/// How `out` stores a missing value of `col`'s type, as read back: float NA
/// is NaN in h5ad and h5seurat (no mask), and h5seurat writes a missing
/// string as "" (SeuratDisk has no NA for character columns).
fn stored_na(out: Out, col: &ColumnData) -> String {
    let seurat = matches!(out, Out::H5SeuratDgc | Out::H5SeuratBpcells);
    match col {
        ColumnData::Float(_) if !matches!(out, Out::Npy) => format!("{:?}", f64::NAN),
        ColumnData::String(_) if seurat => String::new(),
        _ => "NA".into(),
    }
}

/// What a writer is expected to keep besides shape, names and X.
fn keeps_metadata(out: Out) -> bool {
    !matches!(out, Out::BpcellsDir)
}

/// Write `ds` with `out`, read it back, and compare.
async fn check(ds: Dataset, out: Out) -> Result<(), TestCaseError> {
    let dir = tempfile::tempdir().unwrap();
    let path = out_path(dir.path(), out);
    let chunk = ds.chunk;
    let slot = convert::write(&mut Mem(ds.clone()), out, &path, chunk).await;
    let mut back = convert::open(&path, &slot, chunk).await;

    prop_assert_eq!(back.shape(), ds.shape(), "shape");
    let obs = back.obs().await.unwrap();
    prop_assert_eq!(&obs.index, &ds.obs.index, "obs names");
    prop_assert_eq!(
        back.var().await.unwrap().index,
        ds.var.index.clone(),
        "var names"
    );
    let want: Vec<Vec<f64>> =
        ds.x.iter()
            .map(|r| r.iter().map(|&v| f64::from(v)).collect())
            .collect();
    prop_assert_eq!(dense_x(&mut *back).await, want, "X");

    if keeps_metadata(out) {
        for col in &ds.obs.columns {
            let got = column_text(&obs, &col.name, |_| "NA".into());
            prop_assert!(got.is_some(), "obs column {} missing", col.name);
            prop_assert_eq!(
                got.unwrap(),
                column_text(&ds.obs, &col.name, |c| stored_na(out, c)).unwrap(),
                "obs column {} ({:?})",
                col.name,
                std::mem::discriminant(&col.data)
            );
        }
        let obsm = back.obsm().await.unwrap();
        for (key, m) in &ds.obsm.map {
            let got = obsm.map.get(key);
            prop_assert!(got.is_some(), "obsm {} missing", key);
            let got = got.unwrap();
            prop_assert_eq!(got.shape, m.shape, "obsm {} shape", key);
            prop_assert_eq!(&got.data, &m.data, "obsm {} values", key);
        }
    }
    Ok(())
}

fn config() -> ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    ProptestConfig::with_cases(cases)
}

macro_rules! roundtrip {
    ($($name:ident: $out:ident;)*) => {$(
        proptest! {
            #![proptest_config(config())]
            #[test]
            fn $name(ds in dataset()) {
                let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
                rt.block_on(check(ds, Out::$out))?;
            }
        }
    )*};
}

roundtrip! {
    h5ad: H5ad;
    h5seurat_dgcmatrix: H5SeuratDgc;
    h5seurat_bpcells: H5SeuratBpcells;
    bpcells_dir: BpcellsDir;
    npy_snapshot: Npy;
}
