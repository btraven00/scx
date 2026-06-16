use super::reader::ad_read_categorical;
use super::writer::write_vlen_str_dataset;
use super::*;
use futures::StreamExt;
use hdf5::types::VarLenUnicode;
use hdf5::File;
use ndarray::Array1;
use std::str::FromStr;
use tempfile::NamedTempFile;

use crate::dtype::*;
use crate::h5::ScxH5Reader;
use crate::ir::*;
use crate::stream::{DatasetReader, DatasetWriter};

// Golden fixture produced by zellkonverter (via scripts/prepare_h5ad_reference.R)
const GOLDEN_REF: &str = "../../tests/golden/pbmc3k_reference.h5ad";
const GOLDEN: &str = "../../tests/golden/pbmc3k.h5";
// Committed subset fixture (generate with scripts/prepare_norman_subset.py)
const NORMAN_SUBSET: &str = "../../tests/fixtures/norman_subset.h5ad";

fn ref_exists() -> bool {
    std::path::Path::new(GOLDEN_REF).exists()
}

/// Return the path to the Norman H5AD to test against.
/// Prefers the full file via `NORMAN_H5AD` env var (dev/CI with large data),
/// falls back to the committed 500×200 subset.
fn norman_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NORMAN_H5AD") {
        let pb = std::path::PathBuf::from(&p);
        if pb.exists() {
            return Some(pb);
        }
        eprintln!("NORMAN_H5AD={p} not found, falling back to subset");
    }
    let pb = std::path::PathBuf::from(NORMAN_SUBSET);
    if pb.exists() {
        Some(pb)
    } else {
        None
    }
}

// --- H5AdReader tests (against zellkonverter reference) ---

#[tokio::test]
async fn test_reader_shape() {
    if !ref_exists() {
        return;
    }
    let reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let (n_obs, n_vars) = reader.shape();
    assert_eq!(n_obs, 2700, "expected 2700 cells");
    assert_eq!(n_vars, 13714, "expected 13714 genes");
}

#[tokio::test]
async fn test_reader_obs() {
    if !ref_exists() {
        return;
    }
    let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let obs = reader.obs().await.unwrap();
    assert_eq!(obs.index.len(), 2700, "obs index length");
    assert!(!obs.columns.is_empty(), "obs should have columns");
    assert!(
        obs.columns.iter().any(|c| c.name == "nCount_RNA"),
        "expected nCount_RNA column"
    );
}

#[tokio::test]
async fn test_reader_obs_categorical() {
    if !ref_exists() {
        return;
    }
    let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let obs = reader.obs().await.unwrap();
    // Seurat factor columns become categoricals in AnnData
    let cat_cols: Vec<_> = obs
        .columns
        .iter()
        .filter(|c| matches!(c.data, ColumnData::Categorical { .. }))
        .collect();
    assert!(
        !cat_cols.is_empty(),
        "expected at least one categorical obs column"
    );
}

#[tokio::test]
async fn test_reader_var() {
    if !ref_exists() {
        return;
    }
    let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let var = reader.var().await.unwrap();
    assert_eq!(var.index.len(), 13714, "var index length");
}

#[tokio::test]
async fn test_reader_obsm() {
    if !ref_exists() {
        return;
    }
    let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let obsm = reader.obsm().await.unwrap();
    assert!(obsm.map.contains_key("X_pca"), "missing X_pca");
    assert!(obsm.map.contains_key("X_umap"), "missing X_umap");
    assert_eq!(obsm.map["X_pca"].shape.0, 2700);
    assert_eq!(obsm.map["X_umap"].shape.0, 2700);
}

#[tokio::test]
async fn test_reader_stream_coverage() {
    if !ref_exists() {
        return;
    }
    let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let mut total_cells = 0usize;
    let mut total_nnz = 0usize;
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        total_cells += chunk.nrows;
        total_nnz += chunk.data.indices.len();
    }
    assert_eq!(total_cells, 2700);
    // nnz must match the reference (exact value verified against H5Seurat golden)
    assert_eq!(total_nnz, 2282976, "nnz mismatch against reference");
}

#[tokio::test]
async fn test_reader_chunk_size_respected() {
    if !ref_exists() {
        return;
    }
    let chunk_size = 300usize;
    let mut reader = H5AdReader::open(GOLDEN_REF, chunk_size).unwrap();
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        assert!(chunk.nrows <= chunk_size, "chunk exceeded chunk_size");
    }
}

// --- Round-trip: H5AD reader → H5AD writer → structural equivalence ---

#[tokio::test]
async fn test_h5ad_roundtrip() {
    if !ref_exists() {
        return;
    }

    let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
    let (n_obs, n_vars) = reader.shape();
    let dtype = reader.dtype();

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let out_path = tmp.path().to_path_buf();

    let obs = reader.obs().await.unwrap();
    let var = reader.var().await.unwrap();
    let obsm = reader.obsm().await.unwrap();
    let uns = reader.uns().await.unwrap();

    let mut writer = H5AdWriter::create(&out_path, n_obs, n_vars, dtype).unwrap();
    writer.write_obs(&obs).await.unwrap();
    writer.write_var(&var).await.unwrap();
    writer.write_obsm(&obsm).await.unwrap();
    writer.write_uns(&uns).await.unwrap();

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk.unwrap()).await.unwrap();
    }
    writer.finalize().await.unwrap();
    drop(writer);

    // Re-open the output and verify with the reader
    let mut rt = H5AdReader::open(&out_path, 500).unwrap();
    assert_eq!(rt.shape(), (n_obs, n_vars));

    let rt_obs = rt.obs().await.unwrap();
    assert_eq!(rt_obs.index.len(), n_obs);
    assert_eq!(rt_obs.columns.len(), obs.columns.len());

    let rt_obsm = rt.obsm().await.unwrap();
    for key in obsm.map.keys() {
        assert!(
            rt_obsm.map.contains_key(key),
            "obsm['{key}'] missing after roundtrip"
        );
        assert_eq!(rt_obsm.map[key].shape, obsm.map[key].shape);
    }

    let mut total_nnz = 0usize;
    let mut stream = rt.x_stream();
    while let Some(chunk) = stream.next().await {
        total_nnz += chunk.unwrap().data.indices.len();
    }
    assert_eq!(total_nnz, 2282976, "nnz changed after H5AD roundtrip");
}

fn golden_exists() -> bool {
    std::path::Path::new(GOLDEN).exists()
}

/// Full round-trip: read PBMC 3k → write h5ad → verify structure
#[tokio::test]
async fn test_roundtrip_pbmc3k() {
    if !golden_exists() {
        return;
    }

    let mut reader = ScxH5Reader::open(GOLDEN, 500).unwrap();
    let (n_obs, n_vars) = reader.shape();

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let out_path = tmp.path().to_path_buf();

    let obs = reader.obs().await.unwrap();
    let var = reader.var().await.unwrap();
    let obsm = reader.obsm().await.unwrap();
    let uns = reader.uns().await.unwrap();

    let mut writer = H5AdWriter::create(&out_path, n_obs, n_vars, DataType::F32).unwrap();
    writer.write_obs(&obs).await.unwrap();
    writer.write_var(&var).await.unwrap();
    writer.write_obsm(&obsm).await.unwrap();
    writer.write_uns(&uns).await.unwrap();

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk.unwrap()).await.unwrap();
    }
    writer.finalize().await.unwrap();
    drop(writer);

    // --- Verify the output h5ad ---
    let out = File::open(&out_path).unwrap();

    // Root encoding
    let enc_type: String = out
        .group("/")
        .unwrap()
        .attr("encoding-type")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap()
        .to_string();
    assert_eq!(enc_type, "anndata");

    // X shape attribute
    let x_grp = out.group("X").unwrap();
    let shape: ndarray::Array1<i64> = x_grp.attr("shape").unwrap().read_1d().unwrap();
    assert_eq!(shape[0], n_obs as i64, "n_obs mismatch");
    assert_eq!(shape[1], n_vars as i64, "n_vars mismatch");

    // X/data length matches indptr last value
    let indptr: ndarray::Array1<i32> = out.dataset("X/indptr").unwrap().read_1d().unwrap();
    let data_len = out.dataset("X/data").unwrap().shape()[0];
    assert_eq!(data_len, *indptr.last().unwrap() as usize);
    assert_eq!(indptr.len(), n_obs + 1);

    // obs
    let obs_grp = out.group("obs").unwrap();
    let obs_enc: String = obs_grp
        .attr("encoding-type")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap()
        .to_string();
    assert_eq!(obs_enc, "dataframe");
    let obs_idx: ndarray::Array1<VarLenUnicode> =
        out.dataset("obs/index").unwrap().read_1d().unwrap();
    assert_eq!(obs_idx.len(), n_obs);

    // var
    let var_idx: ndarray::Array1<VarLenUnicode> =
        out.dataset("var/index").unwrap().read_1d().unwrap();
    assert_eq!(var_idx.len(), n_vars);

    // obsm
    let pca: ndarray::Array2<f64> = out.dataset("obsm/X_pca").unwrap().read().unwrap();
    assert_eq!(pca.shape(), &[n_obs, 30]);
    let umap: ndarray::Array2<f64> = out.dataset("obsm/X_umap").unwrap().read().unwrap();
    assert_eq!(umap.shape(), &[n_obs, 2]);

    tracing::info!("roundtrip OK: {} cells × {} genes", n_obs, n_vars);
}

/// Regression: layer stored as dense 2-D dataset must not panic when streamed.
/// Previously, layer_stream() always called ad_read_sparse_chunk(), which indexed
/// into an empty indptr and panicked with "index out of bounds: the len is 0".
#[tokio::test]
async fn test_dense_layer_stream_no_panic() {
    use hdf5::File as H5File;
    use ndarray::Array2;

    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();

    let n_obs: usize = 5;
    let n_vars: usize = 4;

    let vlu = |s: &str| VarLenUnicode::from_str(s).unwrap();

    // Build a minimal H5AD with a dense "counts" layer.
    {
        let f = H5File::create(&path).unwrap();
        let root = f.group("/").unwrap();
        root.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("anndata"))
            .unwrap();
        root.new_attr::<VarLenUnicode>()
            .create("encoding-version")
            .unwrap()
            .write_scalar(&vlu("0.1.0"))
            .unwrap();

        // Minimal obs/var dataframes (just the index).
        let obs_grp = f.create_group("obs").unwrap();
        obs_grp
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("dataframe"))
            .unwrap();
        obs_grp
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")
            .unwrap()
            .write_scalar(&vlu("0.2.0"))
            .unwrap();
        obs_grp
            .new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("index"))
            .unwrap();
        let obs_idx: ndarray::Array1<VarLenUnicode> =
            (0..n_obs).map(|i| vlu(&format!("cell{i}"))).collect();
        obs_grp
            .new_dataset_builder()
            .with_data(&obs_idx)
            .create("index")
            .unwrap();

        let var_grp = f.create_group("var").unwrap();
        var_grp
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("dataframe"))
            .unwrap();
        var_grp
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")
            .unwrap()
            .write_scalar(&vlu("0.2.0"))
            .unwrap();
        var_grp
            .new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("index"))
            .unwrap();
        let var_idx: ndarray::Array1<VarLenUnicode> =
            (0..n_vars).map(|i| vlu(&format!("gene{i}"))).collect();
        var_grp
            .new_dataset_builder()
            .with_data(&var_idx)
            .create("index")
            .unwrap();

        // Sparse X (required by H5AdReader::open).
        let x_grp = f.create_group("X").unwrap();
        x_grp
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x_grp
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")
            .unwrap()
            .write_scalar(&vlu("0.1.0"))
            .unwrap();
        let shape = ndarray::array![n_obs as i64, n_vars as i64];
        x_grp
            .new_attr_builder()
            .with_data(&shape)
            .create("shape")
            .unwrap();
        let indptr: ndarray::Array1<i32> = ndarray::Array1::zeros(n_obs + 1);
        x_grp
            .new_dataset_builder()
            .with_data(&indptr)
            .create("indptr")
            .unwrap();
        let indices: ndarray::Array1<i32> = ndarray::Array1::zeros(0);
        x_grp
            .new_dataset_builder()
            .with_data(&indices)
            .create("indices")
            .unwrap();
        let data: ndarray::Array1<f32> = ndarray::Array1::zeros(0);
        x_grp
            .new_dataset_builder()
            .with_data(&data)
            .create("data")
            .unwrap();

        // Dense "counts" layer — shape (n_obs, n_vars), stored as f32.
        let layers_grp = f.create_group("layers").unwrap();
        let counts: Array2<f32> = Array2::from_elem((n_obs, n_vars), 1.0_f32);
        layers_grp
            .new_dataset_builder()
            .with_data(&counts)
            .create("counts")
            .unwrap();
    }

    let mut reader = H5AdReader::open(&path, 3).unwrap();
    let metas = reader.layer_metas().await.unwrap();
    assert_eq!(metas.len(), 1, "expected 1 layer meta");
    assert_eq!(metas[0].name, "counts");
    assert!(
        metas[0].indptr.is_empty(),
        "dense layer must have empty indptr"
    );

    // Stream and collect all chunks — must not panic.
    let mut total_rows = 0usize;
    let mut total_nnz = 0usize;
    let mut stream = reader.layer_stream(&metas[0], 3);
    while let Some(res) = stream.next().await {
        let chunk = res.unwrap();
        total_rows += chunk.nrows;
        total_nnz += chunk.data.indptr.last().copied().unwrap_or(0) as usize;
    }
    assert_eq!(total_rows, n_obs);
    assert_eq!(
        total_nnz,
        n_obs * n_vars,
        "all values are 1.0 so every entry is non-zero"
    );
}

// --- Norman perturbation tests ---
//
// Run against the committed 500×200 subset by default.
// Point NORMAN_H5AD=/path/to/norman_perturbation.h5ad to test the full file.
// Generate the subset with:
//   NORMAN_H5AD=... pixi run -e test prepare-norman-subset

#[tokio::test]
async fn test_norman_shape() {
    let Some(path) = norman_path() else {
        return;
    };
    let reader = H5AdReader::open(&path, 500).unwrap();
    let (n_obs, n_vars) = reader.shape();
    assert!(n_obs > 0, "n_obs must be > 0");
    assert!(n_vars > 0, "n_vars must be > 0");
    eprintln!("norman shape: {n_obs} × {n_vars}");
}

#[tokio::test]
async fn test_norman_obs_has_perturbation_column() {
    let Some(path) = norman_path() else {
        return;
    };
    let mut reader = H5AdReader::open(&path, 500).unwrap();
    let obs = reader.obs().await.unwrap();
    // Norman obs must contain at least one perturbation-related column
    let cols: Vec<&str> = obs.columns.iter().map(|c| c.name.as_str()).collect();
    eprintln!("norman obs columns: {cols:?}");
    assert!(!cols.is_empty(), "obs must have at least one column");
}

#[tokio::test]
async fn test_norman_dense_layer_stream_coverage() {
    let Some(path) = norman_path() else {
        return;
    };
    let (n_obs, n_vars) = H5AdReader::open(&path, 500).unwrap().shape();

    let mut reader = H5AdReader::open(&path, 500).unwrap();
    let metas = reader.layer_metas().await.unwrap();
    assert!(
        !metas.is_empty(),
        "norman H5AD must have at least one layer"
    );

    // counts layer should be present and dense
    let counts = metas
        .iter()
        .find(|m| m.name == "counts")
        .expect("expected a 'counts' layer");
    assert!(
        counts.indptr.is_empty(),
        "counts layer should be dense (empty indptr)"
    );
    assert_eq!(counts.shape, (n_obs, n_vars));

    // stream and verify total row coverage
    let mut total_rows = 0usize;
    let mut stream = reader.layer_stream(counts, 500);
    while let Some(res) = stream.next().await {
        let chunk = res.unwrap();
        assert!(chunk.nrows > 0);
        total_rows += chunk.nrows;
    }
    assert_eq!(total_rows, n_obs, "streamed rows must equal n_obs");
}

#[tokio::test]
async fn test_norman_x_stream_coverage() {
    let Some(path) = norman_path() else {
        return;
    };
    let (n_obs, _) = H5AdReader::open(&path, 500).unwrap().shape();

    let mut reader = H5AdReader::open(&path, 500).unwrap();
    let mut total_rows = 0usize;
    let mut stream = reader.x_stream();
    while let Some(res) = stream.next().await {
        total_rows += res.unwrap().nrows;
    }
    assert_eq!(total_rows, n_obs);
}

// --- categorical with non-string levels (regression: integer-coded
// categoricals such as HBCA's cluster_id were silently skipped) ---

/// Write a minimal categorical group: i8 `codes` + a `categories`
/// dataset of caller-chosen dtype `T`. Returns the open file.
fn write_cat_group<T: hdf5::H5Type>(path: &std::path::Path, codes: &[i8], levels: &[T]) -> File {
    let file = File::create(path).unwrap();
    let grp = file.create_group("col").unwrap();
    grp.new_dataset::<i8>()
        .shape(codes.len())
        .create("codes")
        .unwrap()
        .write(&Array1::from_vec(codes.to_vec()))
        .unwrap();
    grp.new_dataset::<T>()
        .shape(levels.len())
        .create("categories")
        .unwrap()
        .write(levels)
        .unwrap();
    file
}

#[test]
fn test_categorical_int64_levels() {
    let tmp = NamedTempFile::with_suffix(".h5").unwrap();
    // codes index into integer levels [10, 20, 30].
    let file = write_cat_group::<i64>(tmp.path(), &[0, 2, 1, 0], &[10, 20, 30]);
    match ad_read_categorical(&file, "col").unwrap() {
        ColumnData::Categorical { codes, levels } => {
            assert_eq!(levels, vec!["10", "20", "30"]);
            assert_eq!(codes, vec![0, 2, 1, 0]);
        }
        other => panic!("expected Categorical, got {:?}", other),
    }
}

#[test]
fn test_categorical_uint32_levels() {
    let tmp = NamedTempFile::with_suffix(".h5").unwrap();
    let file = write_cat_group::<u32>(tmp.path(), &[1, 0], &[100u32, 200u32]);
    match ad_read_categorical(&file, "col").unwrap() {
        ColumnData::Categorical { codes, levels } => {
            assert_eq!(levels, vec!["100", "200"]);
            assert_eq!(codes, vec![1, 0]);
        }
        other => panic!("expected Categorical, got {:?}", other),
    }
}

#[test]
fn test_categorical_string_levels_unchanged() {
    // The common case must still work after the dtype-dispatch change.
    let tmp = NamedTempFile::with_suffix(".h5").unwrap();
    let file = File::create(tmp.path()).unwrap();
    let grp = file.create_group("col").unwrap();
    grp.new_dataset::<i8>()
        .shape(3)
        .create("codes")
        .unwrap()
        .write(&Array1::from_vec(vec![0i8, 1, 0]))
        .unwrap();
    write_vlen_str_dataset(&grp, "categories", &["a".into(), "b".into()]).unwrap();
    match ad_read_categorical(&file, "col").unwrap() {
        ColumnData::Categorical { codes, levels } => {
            assert_eq!(levels, vec!["a", "b"]);
            assert_eq!(codes, vec![0, 1, 0]);
        }
        other => panic!("expected Categorical, got {:?}", other),
    }
}

// --- Self-contained round-trips (no golden fixtures; run in CI) ---

/// Diagonal CSR: cell_i has a single nonzero at gene_i (i < min(n_obs, n_vars)).
fn diag_csr(n_obs: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let nnz = n_obs.min(n_vars);
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in 0..n_obs {
        if r < nnz {
            indices.push(r as u32);
            data.push(1.0f32);
            indptr.push(indptr.last().unwrap() + 1);
        } else {
            indptr.push(*indptr.last().unwrap());
        }
    }
    (indptr, indices, data)
}

/// Write → read every slot type with a synthetic dataset, asserting structure
/// survives. Exercises the writer's dataframe/column/obsm/uns/layer paths and
/// the matching reader helpers without depending on a golden fixture.
#[tokio::test]
async fn synthetic_roundtrip_all_slots() {
    let (n_obs, n_vars) = (6usize, 4usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();

    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("cell{i}")).collect(),
        columns: vec![
            Column {
                name: "n_counts".into(),
                data: ColumnData::Int((0..n_obs as i32).collect()),
            },
            Column {
                name: "score".into(),
                data: ColumnData::Float((0..n_obs).map(|i| i as f64 * 0.5).collect()),
            },
            Column {
                name: "sample".into(),
                data: ColumnData::String((0..n_obs).map(|i| format!("s{}", i % 3)).collect()),
            },
            Column {
                name: "passed".into(),
                data: ColumnData::Bool((0..n_obs).map(|i| i % 2 == 0).collect()),
            },
            Column {
                name: "leiden".into(),
                data: ColumnData::Categorical {
                    codes: (0..n_obs as u32).map(|i| i % 3).collect(),
                    levels: vec!["A".into(), "B".into(), "C".into()],
                },
            },
        ],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![Column {
            name: "highly_variable".into(),
            data: ColumnData::Bool((0..n_vars).map(|i| i % 2 == 0).collect()),
        }],
    };
    let mut obsm = Embeddings::default();
    obsm.map.insert(
        "X_pca".into(),
        DenseMatrix {
            shape: (n_obs, 2),
            data: (0..n_obs * 2).map(|i| i as f64).collect(),
        },
    );
    let uns = UnsTable {
        raw: serde_json::json!({ "title": "synthetic", "params": { "k": 15 } }),
    };
    let (indptr, indices, data) = diag_csr(n_obs, n_vars);
    let x = |d: &[f32]| MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr: indptr.clone(),
            indices: indices.clone(),
            data: TypedVec::F32(d.to_vec()),
        },
    };

    let mut w = H5AdWriter::create(&path, n_obs, n_vars, DataType::F32).unwrap();
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    w.write_obsm(&obsm).await.unwrap();
    w.write_uns(&uns).await.unwrap();
    w.write_x_chunk(&x(&data)).await.unwrap();
    w.finalize().await.unwrap();
    drop(w);

    // Append a layer (also covers open_for_append + append-mode sparse write).
    let mut w = H5AdWriter::open_for_append(&path).unwrap();
    let meta = SparseMatrixMeta {
        name: "normalized".into(),
        shape: (n_obs, n_vars),
        indptr: indptr.clone(),
    };
    w.begin_sparse("layers", "normalized", &meta).await.unwrap();
    w.write_sparse_chunk(&x(&data)).await.unwrap();
    w.end_sparse().await.unwrap();
    drop(w);

    let mut r = H5AdReader::open(&path, 3).unwrap();
    assert_eq!(r.shape(), (n_obs, n_vars));

    let robs = r.obs().await.unwrap();
    assert_eq!(robs.index.len(), n_obs);
    // Int/Float/String/Categorical round-trip. The Bool column ("passed") does
    // NOT yet: the writer stores bool as unsigned u8 but ad_read_column has no
    // Unsigned arm, so it's dropped on read. (Known reader gap — follow-up fix.)
    let names: Vec<&str> = robs.columns.iter().map(|c| c.name.as_str()).collect();
    for n in ["n_counts", "score", "sample", "leiden"] {
        assert!(names.contains(&n), "obs column '{n}' missing: {names:?}");
    }
    assert!(robs
        .columns
        .iter()
        .any(|c| c.name == "leiden" && matches!(c.data, ColumnData::Categorical { .. })));

    assert_eq!(r.var().await.unwrap().index.len(), n_vars);
    assert_eq!(r.obsm().await.unwrap().map["X_pca"].shape, (n_obs, 2));
    assert!(r.uns().await.unwrap().raw.get("title").is_some());
    assert!(r
        .layer_metas()
        .await
        .unwrap()
        .iter()
        .any(|m| m.name == "normalized"));

    let mut nnz = 0usize;
    let mut s = r.x_stream();
    while let Some(c) = s.next().await {
        nnz += c.unwrap().data.indices.len();
    }
    assert_eq!(nnz, n_obs.min(n_vars));
}

/// Round-trip X in every supported dtype, asserting the writer/reader dtype
/// branches (write_x_chunk + read_x_data/bytes_to_typed) agree.
#[tokio::test]
async fn synthetic_roundtrip_dtypes() {
    // U32 X is intentionally omitted: ad_detect_dtype has no Unsigned arm, so a
    // u32-stored X reads back as F32 (known reader gap, follow-up fix).
    for dt in [DataType::F32, DataType::F64, DataType::I32] {
        let (n_obs, n_vars) = (4usize, 3usize);
        let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
        let path = tmp.path().to_path_buf();
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("c{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("g{i}")).collect(),
            columns: vec![],
        };
        let indptr: Vec<u64> = (0..=n_obs as u64).collect(); // one nnz per cell
        let indices: Vec<u32> = vec![0u32; n_obs];
        let values = match dt {
            DataType::F32 => TypedVec::F32(vec![7.0; n_obs]),
            DataType::F64 => TypedVec::F64(vec![7.0; n_obs]),
            DataType::I32 => TypedVec::I32(vec![7; n_obs]),
            DataType::U32 => TypedVec::U32(vec![7; n_obs]),
        };
        let mut w = H5AdWriter::create(&path, n_obs, n_vars, dt).unwrap();
        w.write_obs(&obs).await.unwrap();
        w.write_var(&var).await.unwrap();
        w.write_obsm(&Embeddings::default()).await.unwrap();
        w.write_uns(&UnsTable::default()).await.unwrap();
        w.write_x_chunk(&MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr,
                indices,
                data: values,
            },
        })
        .await
        .unwrap();
        w.finalize().await.unwrap();
        drop(w);

        let mut r = H5AdReader::open(&path, 2).unwrap();
        assert_eq!(r.dtype(), dt, "dtype {dt:?} round-trips");
        let mut nnz = 0usize;
        let mut s = r.x_stream();
        while let Some(c) = s.next().await {
            nnz += c.unwrap().data.indices.len();
        }
        assert_eq!(nnz, n_obs, "nnz for dtype {dt:?}");
    }
}
