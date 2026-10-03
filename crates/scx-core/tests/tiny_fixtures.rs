//! Every reader against `tests/fixtures/tiny/expected.json`: one 6-cell x
//! 5-gene dataset stored in each input format by scripts/make_tiny_fixtures.py.
//!
//! Tests marked `#[ignore = "bug: ..."]` document known reader bugs; they fail
//! today (run them with `cargo test -- --ignored`) and are un-ignored by the
//! fix.

use futures::StreamExt;
use scx_core::ir::{Column, ColumnData};
use scx_core::stream::DatasetReader;
use serde_json::Value;
use std::path::PathBuf;

fn tiny(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(tiny("expected.json")).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

async fn open(name: &str) -> Box<dyn DatasetReader + Send> {
    let opts = scx_core::OpenOptions::new(4); // < n_obs, so X spans two chunks
    scx_core::open(tiny(name).to_str().unwrap(), &opts)
        .await
        .unwrap_or_else(|e| panic!("open {name}: {e}"))
}

async fn dense_x(reader: &mut (dyn DatasetReader + Send)) -> Vec<Vec<f64>> {
    let (n_obs, n_vars) = reader.shape();
    let mut out = vec![vec![0.0; n_vars]; n_obs];
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let csr = &chunk.data;
        let values = csr.data.to_f64();
        for r in 0..chunk.nrows {
            for k in csr.indptr[r] as usize..csr.indptr[r + 1] as usize {
                out[chunk.row_offset + r][csr.indices[k] as usize] = values[k];
            }
        }
    }
    out
}

/// Shape, gene names and matrix; `slot` is "X" or "counts" depending on what
/// the format stores as its main matrix. Cell names are checked separately
/// (`check_obs_names`) because reading obs also decodes the metadata columns.
async fn check_matrix(name: &str, slot: &str) {
    let e = expected();
    let mut reader = open(name).await;
    assert_eq!(reader.shape(), (6, 5), "{name}: shape");
    assert_eq!(
        reader.var().await.unwrap().index,
        strings(&e["var_names"]),
        "{name}: var names"
    );
    assert_eq!(
        dense_x(&mut *reader).await,
        matrix(&e[slot]),
        "{name}: {slot}"
    );
}

async fn check_obs_names(name: &str) {
    let obs = open(name).await.obs().await.unwrap();
    assert_eq!(
        obs.index,
        strings(&expected()["obs_names"]),
        "{name}: obs names"
    );
}

#[tokio::test]
async fn obs_names() {
    for name in ["tiny.h5ad", "tiny_dense.h5ad", "tiny_10x.h5", "tiny_mtx"] {
        check_obs_names(name).await;
    }
}

#[tokio::test]
async fn h5seurat_v4_obs_names() {
    check_obs_names("tiny_v4.h5seurat").await;
}

#[tokio::test]
async fn h5ad_matrix() {
    check_matrix("tiny.h5ad", "X").await;
}

#[tokio::test]
async fn h5ad_dense_matrix() {
    check_matrix("tiny_dense.h5ad", "X").await;
}

#[tokio::test]
async fn tenx_matrix() {
    check_matrix("tiny_10x.h5", "counts").await;
}

#[tokio::test]
async fn mtx_matrix() {
    check_matrix("tiny_mtx", "counts").await;
}

#[tokio::test]
async fn h5seurat_v4_matrix() {
    check_matrix("tiny_v4.h5seurat", "counts").await;
}

#[tokio::test]
#[ignore = "bug: H5SeuratReader::open only probes assays/RNA/counts, not the v5 assays/RNA/layers/counts"]
async fn h5seurat_v5_bpcells_matrix() {
    check_matrix("tiny_v5_bpcells.h5seurat", "counts").await;
}

#[tokio::test]
async fn h5ad_counts_layer() {
    let e = expected();
    let mut reader = open("tiny.h5ad").await;
    let metas = reader.layer_metas().await.unwrap();
    let meta = metas
        .iter()
        .find(|m| m.name == "counts")
        .expect("counts layer");
    let mut dense = vec![vec![0.0; 5]; 6];
    let mut stream = reader.layer_stream(meta, 4);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let values = chunk.data.data.to_f64();
        for r in 0..chunk.nrows {
            for k in chunk.data.indptr[r] as usize..chunk.data.indptr[r + 1] as usize {
                dense[chunk.row_offset + r][chunk.data.indices[k] as usize] = values[k];
            }
        }
    }
    assert_eq!(dense, matrix(&e["counts"]));
}

/// Categorical values with NA as `None`.
fn categorical_values(col: &Column) -> Vec<Option<String>> {
    let ColumnData::Categorical { codes, levels } = &col.data else {
        panic!("{} is {}, not categorical", col.name, col.data.dtype_str());
    };
    (0..codes.len())
        .map(|i| (!col.is_na(i)).then(|| levels[codes[i] as usize].clone()))
        .collect()
}

fn expected_categorical(e: &Value, col: &str) -> Vec<Option<String>> {
    e["obs"][col]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().map(String::from))
        .collect()
}

async fn check_cell_type(name: &str) {
    let e = expected();
    let obs = open(name).await.obs().await.unwrap();
    let col = obs
        .columns
        .iter()
        .find(|c| c.name == "cell_type")
        .expect("cell_type");
    assert_eq!(
        categorical_values(col),
        expected_categorical(&e, "cell_type")
    );
}

#[tokio::test]
async fn h5ad_categorical_with_na() {
    check_cell_type("tiny.h5ad").await;
}

#[tokio::test]
async fn h5seurat_v4_categorical_with_na() {
    check_cell_type("tiny_v4.h5seurat").await;
}

#[tokio::test]
async fn h5ad_nullable_bool_and_int_keep_na() {
    let e = expected();
    let obs = open("tiny.h5ad").await.obs().await.unwrap();
    for name in ["flag", "batch"] {
        let col = obs.columns.iter().find(|c| c.name == name).expect(name);
        let na: Vec<bool> = (0..6).map(|i| col.is_na(i)).collect();
        let want: Vec<bool> = e["obs"][name]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(Value::is_null)
            .collect();
        assert_eq!(na, want, "{name}");
    }
}

#[tokio::test]
async fn h5ad_obsm_varm_obsp() {
    let e = expected();
    let mut reader = open("tiny.h5ad").await;
    let obsm = reader.obsm().await.unwrap();
    let pca = &obsm.map["X_pca"];
    assert_eq!(pca.shape, (6, 2));
    assert_eq!(pca.data, matrix(&e["obsm"]["X_pca"]).concat());
    let varm = reader.varm().await.unwrap();
    assert_eq!(varm.map["PCs"].shape, (5, 2));
    let obsp = reader.obsp_metas().await.unwrap();
    assert_eq!(
        obsp.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
        ["connectivities"]
    );
}

#[tokio::test]
async fn h5ad_uns() {
    let uns = open("tiny.h5ad").await.uns().await.unwrap();
    let e = expected();
    for key in ["title", "n_pcs", "params", "colors", "weights"] {
        assert!(uns.raw.get(key).is_some(), "uns.{key} missing: {}", uns.raw);
    }
    assert_eq!(uns.raw["title"], e["uns"]["title"]);
    assert_eq!(
        uns.raw["params"]["resolution"],
        e["uns"]["params"]["resolution"]
    );
}
