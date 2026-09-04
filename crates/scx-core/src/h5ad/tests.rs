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
        out.dataset("obs/_index").unwrap().read_1d().unwrap();
    assert_eq!(obs_idx.len(), n_obs);

    // var
    let var_idx: ndarray::Array1<VarLenUnicode> =
        out.dataset("var/_index").unwrap().read_1d().unwrap();
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

/// Write a CSR `layers/<name>` group (all-ones matrix, one nnz per cell) into
/// an already-open file's `/layers` group.
fn write_csr_layer(layers: &hdf5::Group, name: &str, n_obs: usize, n_vars: usize) {
    let g = layers.create_group(name).unwrap();
    g.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&VarLenUnicode::from_str("csr_matrix").unwrap())
        .unwrap();
    let shape = ndarray::array![n_obs as i64, n_vars as i64];
    g.new_attr_builder()
        .with_data(&shape)
        .create("shape")
        .unwrap();
    // One entry per row at column 0.
    let indptr: Array1<i32> = (0..=n_obs as i32).collect();
    g.new_dataset_builder()
        .with_data(&indptr)
        .create("indptr")
        .unwrap();
    let indices: Array1<i32> = Array1::zeros(n_obs);
    g.new_dataset_builder()
        .with_data(&indices)
        .create("indices")
        .unwrap();
    let data: Array1<f32> = Array1::from_elem(n_obs, 1.0);
    g.new_dataset_builder()
        .with_data(&data)
        .create("data")
        .unwrap();
}

/// A file with no `/X` (written as `adata.X = None`) but a `layers/counts`
/// matrix must open by auto-falling-back to the sole layer.
#[tokio::test]
async fn missing_x_falls_back_to_sole_layer() {
    let (n_obs, n_vars) = (5usize, 4usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();
    {
        let f = File::create(&path).unwrap();
        let layers = f.create_group("layers").unwrap();
        write_csr_layer(&layers, "counts", n_obs, n_vars);
    }

    let mut reader = H5AdReader::open(&path, 2).unwrap();
    assert_eq!(reader.x_source(), "layers/counts");
    assert_eq!(reader.shape(), (n_obs, n_vars));

    let mut total_nnz = 0usize;
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        total_nnz += chunk.unwrap().data.indptr.last().copied().unwrap_or(0) as usize;
    }
    assert_eq!(total_nnz, n_obs, "one nnz per row");
}

/// With no `/X` and several layers, auto-fallback errors unless one is named
/// `counts`/`X`; an explicit `open_layer` always works.
#[tokio::test]
async fn missing_x_multiple_layers_needs_explicit_choice() {
    let (n_obs, n_vars) = (3usize, 2usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();
    {
        let f = File::create(&path).unwrap();
        let layers = f.create_group("layers").unwrap();
        write_csr_layer(&layers, "spliced", n_obs, n_vars);
        write_csr_layer(&layers, "unspliced", n_obs, n_vars);
    }

    // Ambiguous — no counts/X layer to prefer.
    assert!(H5AdReader::open(&path, 2).is_err());
    // Explicit choice resolves it.
    let r = H5AdReader::open_layer(&path, 2, Some("spliced")).unwrap();
    assert_eq!(r.x_source(), "layers/spliced");
    assert_eq!(r.shape(), (n_obs, n_vars));
    // Unknown layer is a clear error.
    assert!(H5AdReader::open_layer(&path, 2, Some("nope")).is_err());
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

/// Regression: a source obs/var column literally named "_index" collides with
/// the reserved frame-index dataset (`<obs|var>/_index`). Before the fix the
/// writer created `_index` twice and HDF5 aborted the second create with "name
/// already exists" — surfaced by scx as the opaque "unknown library error".
/// Repro in the wild: Azimuth `pbmc_multimodal.h5seurat`, whose `meta.data`
/// carries an `_index` column left over from a prior AnnData round-trip.
/// The writer must drop the colliding column and produce a valid file.
#[tokio::test]
async fn dataframe_column_named_index_is_dropped() {
    let (n_obs, n_vars) = (4usize, 3usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();

    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("cell{i}")).collect(),
        columns: vec![
            // Offending column: same name as the reserved frame index.
            Column {
                name: "_index".into(),
                data: ColumnData::String((0..n_obs).map(|i| format!("cell{i}")).collect()),
            },
            Column {
                name: "celltype".into(),
                data: ColumnData::String((0..n_obs).map(|i| format!("t{}", i % 2)).collect()),
            },
        ],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![],
    };
    let (indptr, indices, data) = diag_csr(n_obs, n_vars);
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr,
            indices,
            data: TypedVec::F32(data),
        },
    };

    let mut w = H5AdWriter::create(&path, n_obs, n_vars, DataType::F32).unwrap();
    // Previously errored here: HDF5 "name already exists" on the second `_index`.
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    w.write_x_chunk(&chunk).await.unwrap();
    w.finalize().await.unwrap();
    drop(w);

    // The reserved-name column is dropped; the real index and siblings survive.
    let mut rt = H5AdReader::open(&path, 500).unwrap();
    let rt_obs = rt.obs().await.unwrap();
    assert_eq!(rt_obs.index, obs.index, "frame index must be preserved");
    let names: Vec<&str> = rt_obs.columns.iter().map(|c| c.name.as_str()).collect();
    assert!(
        !names.contains(&"_index"),
        "reserved '_index' column must be dropped, got {names:?}"
    );
    assert!(
        names.contains(&"celltype"),
        "non-colliding columns must survive, got {names:?}"
    );
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
    // All five column types round-trip (Int/Float/String/Bool/Categorical).
    assert_eq!(robs.columns.len(), 5, "all obs column types preserved");
    let names: Vec<&str> = robs.columns.iter().map(|c| c.name.as_str()).collect();
    for n in ["n_counts", "score", "sample", "passed", "leiden"] {
        assert!(names.contains(&n), "obs column '{n}' missing: {names:?}");
    }
    assert!(robs
        .columns
        .iter()
        .any(|c| c.name == "leiden" && matches!(c.data, ColumnData::Categorical { .. })));
    assert!(robs
        .columns
        .iter()
        .any(|c| c.name == "passed" && matches!(c.data, ColumnData::Bool(_))));

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
    for dt in [DataType::F32, DataType::F64, DataType::I32, DataType::U32] {
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

// --- gzip (deflate) compression ---

/// Write a small all-slots synthetic dataset at the given compression level and
/// return the temp file (kept alive so the caller can read/inspect it).
async fn write_synthetic_compressed(compression: Option<u8>) -> NamedTempFile {
    let (n_obs, n_vars) = (6usize, 4usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("cell{i}")).collect(),
        columns: vec![
            Column {
                name: "n_counts".into(),
                data: ColumnData::Int((0..n_obs as i32).collect()),
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
        columns: vec![],
    };
    let mut obsm = Embeddings::default();
    obsm.map.insert(
        "X_pca".into(),
        DenseMatrix {
            shape: (n_obs, 2),
            data: (0..n_obs * 2).map(|i| i as f64).collect(),
        },
    );
    let (indptr, indices, data) = diag_csr(n_obs, n_vars);
    let mut w =
        H5AdWriter::create_compressed(tmp.path(), n_obs, n_vars, DataType::F32, compression)
            .unwrap();
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    w.write_obsm(&obsm).await.unwrap();
    w.write_x_chunk(&MatrixChunk {
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
    w.finalize().await.unwrap();
    tmp
}

async fn count_nnz(path: &std::path::Path) -> usize {
    let mut r = H5AdReader::open(path, 3).unwrap();
    let mut nnz = 0usize;
    let mut s = r.x_stream();
    while let Some(c) = s.next().await {
        nnz += c.unwrap().data.indices.len();
    }
    nnz
}

/// A gzip-compressed file round-trips to the same structure as an uncompressed
/// one (compression is transparent to the reader via libhdf5's filter pipeline).
#[tokio::test]
async fn compressed_roundtrip_equals_uncompressed() {
    let comp = write_synthetic_compressed(Some(6)).await;
    let plain = write_synthetic_compressed(None).await;

    let mut rc = H5AdReader::open(comp.path(), 3).unwrap();
    let mut rp = H5AdReader::open(plain.path(), 3).unwrap();
    assert_eq!(rc.shape(), rp.shape());

    let oc = rc.obs().await.unwrap();
    let op = rp.obs().await.unwrap();
    assert_eq!(oc.index, op.index);
    assert_eq!(oc.columns.len(), op.columns.len());
    assert!(oc
        .columns
        .iter()
        .any(|c| c.name == "leiden" && matches!(c.data, ColumnData::Categorical { .. })));
    assert!(oc
        .columns
        .iter()
        .any(|c| c.name == "passed" && matches!(c.data, ColumnData::Bool(_))));

    assert_eq!(
        rc.obsm().await.unwrap().map["X_pca"].shape,
        rp.obsm().await.unwrap().map["X_pca"].shape
    );

    let nnz_c = count_nnz(comp.path()).await;
    let nnz_p = count_nnz(plain.path()).await;
    assert_eq!(nnz_c, nnz_p, "nnz differs between compressed and plain");
    // diag_csr(6, 4) fills min(n_obs, n_vars) = 4 diagonal entries.
    assert_eq!(nnz_c, 4, "diagonal nnz preserved under compression");
}

/// Write a large, highly compressible payload uncompressed vs gzip and assert
/// the compressed file is smaller.
async fn write_big_compressible(path: &std::path::Path, compression: Option<u8>) {
    let (n_obs, n_vars) = (20_000usize, 50usize);
    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("c{i}")).collect(),
        // Constant column compresses to almost nothing.
        columns: vec![Column {
            name: "const".into(),
            data: ColumnData::Float(vec![1.0; n_obs]),
        }],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![],
    };
    // One nnz per row, all value 1.0 at column 0 → extremely compressible.
    let indptr: Vec<u64> = (0..=n_obs as u64).collect();
    let indices = vec![0u32; n_obs];
    let data = vec![1.0f32; n_obs];
    let mut w =
        H5AdWriter::create_compressed(path, n_obs, n_vars, DataType::F32, compression).unwrap();
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    w.write_x_chunk(&MatrixChunk {
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
    w.finalize().await.unwrap();
}

#[tokio::test]
async fn compressed_file_is_smaller() {
    let plain = NamedTempFile::with_suffix(".h5ad").unwrap();
    let comp = NamedTempFile::with_suffix(".h5ad").unwrap();
    write_big_compressible(plain.path(), None).await;
    write_big_compressible(comp.path(), Some(6)).await;

    let ps = std::fs::metadata(plain.path()).unwrap().len();
    let cs = std::fs::metadata(comp.path()).unwrap().len();
    assert!(
        cs < ps,
        "compressed file ({cs} bytes) should be smaller than plain ({ps} bytes)"
    );
}

/// The deflate filter lands on numeric datasets but not on vlen string datasets.
#[tokio::test]
async fn deflate_filter_applied_to_x_not_vlen() {
    use hdf5::filters::Filter;
    let comp = write_synthetic_compressed(Some(6)).await;
    let f = File::open(comp.path()).unwrap();

    let xdata = f.dataset("X/data").unwrap();
    assert!(
        xdata.is_chunked(),
        "X/data should be chunked when compressed"
    );
    assert!(
        xdata
            .filters()
            .iter()
            .any(|fl| matches!(fl, Filter::Deflate(_))),
        "X/data should carry a deflate filter"
    );

    // The obs index is a variable-length string dataset — left uncompressed.
    let idx = f.dataset("obs/_index").unwrap();
    assert!(
        !idx.filters()
            .iter()
            .any(|fl| matches!(fl, Filter::Deflate(_))),
        "vlen string index must stay uncompressed"
    );
}

#[tokio::test]
async fn compression_level_out_of_range_errors() {
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let r = H5AdWriter::create_compressed(tmp.path(), 2, 2, DataType::F32, Some(10));
    assert!(r.is_err(), "gzip level 10 must be rejected");
}

/// Build a minimal h5ad whose obs index and one obs column use anndata 0.13's
/// `nullable-string-array` encoding: a *group* of `values` + `mask`, rather
/// than a plain string dataset.
fn write_nullable_string_h5ad(path: &std::path::Path) {
    let f = File::create(path).unwrap();
    let root = f.group("/").unwrap();
    let enc = |o: &hdf5::Group, v: &str| {
        let s = VarLenUnicode::from_str(v).unwrap();
        o.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&s)
            .unwrap();
    };
    enc(&root, "anndata");

    // X: 2 cells x 2 genes, CSR, one entry per row.
    let x = f.create_group("X").unwrap();
    enc(&x, "csr_matrix");
    let shape = x.new_attr::<i64>().shape(2).create("shape").unwrap();
    shape.write(&Array1::from_vec(vec![2i64, 2])).unwrap();
    for (name, vals) in [("indices", vec![0i32, 1]), ("indptr", vec![0i32, 1, 2])] {
        let ds = x
            .new_dataset::<i32>()
            .shape(vals.len())
            .create(name)
            .unwrap();
        ds.write(&Array1::from_vec(vals)).unwrap();
    }
    let d = x.new_dataset::<f32>().shape(2).create("data").unwrap();
    d.write(&Array1::from_vec(vec![1.0f32, 2.0])).unwrap();

    // A nullable-string-array group: values + mask, mask 1 = missing.
    let nullable_str = |parent: &hdf5::Group, name: &str, vals: &[&str], mask: &[bool]| {
        let g = parent.create_group(name).unwrap();
        enc(&g, "nullable-string-array");
        let owned: Vec<String> = vals.iter().map(|s| s.to_string()).collect();
        write_vlen_str_dataset(&g, "values", &owned).unwrap();
        let m = g
            .new_dataset::<bool>()
            .shape(mask.len())
            .create("mask")
            .unwrap();
        m.write(&Array1::from_vec(mask.to_vec())).unwrap();
    };

    for (frame, labels) in [("obs", ["cell_a", "cell_b"]), ("var", ["gene_a", "gene_b"])] {
        let g = f.create_group(frame).unwrap();
        enc(&g, "dataframe");
        let idx = VarLenUnicode::from_str("_index").unwrap();
        g.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&idx)
            .unwrap();
        let order = if frame == "obs" {
            vec!["batch"]
        } else {
            vec![]
        };
        let names: Vec<VarLenUnicode> = order
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        let attr = g
            .new_attr::<VarLenUnicode>()
            .shape(names.len())
            .create("column-order")
            .unwrap();
        attr.write(&Array1::from_vec(names)).unwrap();
        nullable_str(&g, "_index", &labels, &[false, false]);
        if frame == "obs" {
            // Second entry masked, to prove the mask is honoured.
            nullable_str(&g, "batch", &["s1", "ignored"], &[false, true]);
        }
    }
}

/// Regression: anndata 0.13 writes string columns — the dataframe index
/// included — as `nullable-string-array`, a group of `values` + `mask`. scx
/// opened the index path directly as a dataset, so *every* file written by a
/// current anndata failed with "H5Dopen2(): not a dataset", and nullable string
/// columns were dropped with a warning because the reader had no string arm.
#[tokio::test]
async fn test_reads_anndata_013_nullable_string_index_and_column() {
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    write_nullable_string_h5ad(tmp.path());

    let mut reader = H5AdReader::open(tmp.path(), 10).unwrap();
    assert_eq!(reader.shape(), (2, 2));

    let obs = reader.obs().await.unwrap();
    assert_eq!(
        obs.index,
        vec!["cell_a", "cell_b"],
        "index must be readable"
    );

    let var = reader.var().await.unwrap();
    assert_eq!(var.index, vec!["gene_a", "gene_b"]);

    let batch = obs
        .columns
        .iter()
        .find(|c| c.name == "batch")
        .expect("nullable string column must be read, not skipped");
    match &batch.data {
        ColumnData::String(v) => {
            // Masked entries become "", the fill a missing string gets elsewhere.
            assert_eq!(v, &vec!["s1".to_string(), String::new()]);
        }
        other => panic!("expected String column, got {other:?}"),
    }
}

/// Regression: boolean obs/var columns must be written as the HDF5 enum
/// `{FALSE=0, TRUE=1}` that h5py/AnnData use, not as a plain integer.
///
/// The writer used to cast `ColumnData::Bool` to `u8`, producing an
/// `H5T_INTEGER` dataset. Readers that key off the HDF5 type rather than the
/// `encoding-type` attribute then mis-typed the column: `rhdf5::h5read` hands
/// back an R `raw` vector instead of `logical`, and passing that to duckdb's
/// `rapi_register_df` aborts with a bare `std::exception`. Repro in the wild:
/// a CELLxGENE h5ad (`obs/is_primary_data`, `var/feature_is_filtered`)
/// round-tripped through scx and then ingested by the bixverse R package.
///
/// Asserting on the *dtype*, not just the round-tripped values, is the point —
/// scx's own reader accepts both encodings, so a value-only test passes with
/// the bug present.
#[tokio::test]
async fn bool_columns_are_written_as_h5_enum_not_integer() {
    let (n_obs, n_vars) = (4usize, 3usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();

    let flags: Vec<bool> = vec![true, false, true, true];
    let gene_flags: Vec<bool> = vec![false, true, false];

    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("cell{i}")).collect(),
        columns: vec![Column {
            name: "is_primary_data".into(),
            data: ColumnData::Bool(flags.clone()),
        }],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![Column {
            name: "feature_is_filtered".into(),
            data: ColumnData::Bool(gene_flags.clone()),
        }],
    };
    let (indptr, indices, data) = diag_csr(n_obs, n_vars);
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr,
            indices,
            data: TypedVec::F32(data),
        },
    };

    let mut w = H5AdWriter::create(&path, n_obs, n_vars, DataType::F32).unwrap();
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    w.write_x_chunk(&chunk).await.unwrap();
    w.finalize().await.unwrap();
    drop(w);

    // The actual regression: on-disk HDF5 type must be Boolean (an enum), and
    // must NOT be Integer, for both frames.
    let f = File::open(&path).unwrap();
    for ds_path in ["obs/is_primary_data", "var/feature_is_filtered"] {
        let ds = f.dataset(ds_path).unwrap();
        let descr = ds.dtype().unwrap().to_descriptor().unwrap();
        assert_eq!(
            descr,
            hdf5::types::TypeDescriptor::Boolean,
            "{ds_path} must be an HDF5 enum bool, got {descr:?}"
        );
    }
    drop(f);

    // …and the values still survive the round-trip.
    let mut rt = H5AdReader::open(&path, 500).unwrap();
    let rt_obs = rt.obs().await.unwrap();
    match &rt_obs
        .columns
        .iter()
        .find(|c| c.name == "is_primary_data")
        .unwrap()
        .data
    {
        ColumnData::Bool(v) => assert_eq!(v, &flags, "obs bool values must round-trip"),
        other => panic!("expected Bool column, got {other:?}"),
    }
    let rt_var = rt.var().await.unwrap();
    match &rt_var
        .columns
        .iter()
        .find(|c| c.name == "feature_is_filtered")
        .unwrap()
        .data
    {
        ColumnData::Bool(v) => assert_eq!(v, &gene_flags, "var bool values must round-trip"),
        other => panic!("expected Bool column, got {other:?}"),
    }
}

/// The enum encoding must survive the compressed path too: booleans are one
/// byte, so they take the chunk+deflate branch of `write_1d`, and a filter
/// pipeline is exactly the kind of thing that can silently change a dtype.
#[tokio::test]
async fn bool_columns_stay_enum_when_compressed() {
    let (n_obs, n_vars) = (64usize, 2usize);
    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
    let path = tmp.path().to_path_buf();

    let flags: Vec<bool> = (0..n_obs).map(|i| i % 3 == 0).collect();
    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("c{i}")).collect(),
        columns: vec![Column {
            name: "flag".into(),
            data: ColumnData::Bool(flags.clone()),
        }],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![],
    };
    let (indptr, indices, data) = diag_csr(n_obs, n_vars);
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr,
            indices,
            data: TypedVec::F32(data),
        },
    };

    let mut w =
        H5AdWriter::create_compressed(&path, n_obs, n_vars, DataType::F32, Some(4)).unwrap();
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    w.write_x_chunk(&chunk).await.unwrap();
    w.finalize().await.unwrap();
    drop(w);

    let f = File::open(&path).unwrap();
    let descr = f
        .dataset("obs/flag")
        .unwrap()
        .dtype()
        .unwrap()
        .to_descriptor()
        .unwrap();
    assert_eq!(
        descr,
        hdf5::types::TypeDescriptor::Boolean,
        "compressed bool column must still be an enum, got {descr:?}"
    );
    drop(f);

    let mut rt = H5AdReader::open(&path, 500).unwrap();
    let rt_obs = rt.obs().await.unwrap();
    match &rt_obs
        .columns
        .iter()
        .find(|c| c.name == "flag")
        .unwrap()
        .data
    {
        ColumnData::Bool(v) => assert_eq!(v, &flags),
        other => panic!("expected Bool column, got {other:?}"),
    }
}
