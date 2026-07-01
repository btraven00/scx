use super::reader::{detect_sparse_group_kind, SparseGroupKind};
use super::*;
use futures::StreamExt;
use hdf5::types::VarLenUnicode;
use hdf5::File;
use ndarray::Array1;
use std::str::FromStr;

use crate::dtype::*;
use crate::ir::*;
use crate::stream::{DatasetReader, DatasetWriter};

const GOLDEN: &str = "../../tests/golden/pbmc3k.h5seurat";
const NORMAN_FIXTURE: &str = "../../tests/fixtures/norman_subset.h5ad";

fn golden_exists() -> bool {
    std::path::Path::new(GOLDEN).exists()
}

fn norman_exists() -> bool {
    std::path::Path::new(NORMAN_FIXTURE).exists()
}

#[test]
fn test_detect_sparse_group_kind_bpcells_layer() {
    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let path = tmp.path();

    let file = File::create(path).unwrap();
    file.create_group("assays").unwrap();
    file.create_group("assays/RNA").unwrap();
    let grp = file.create_group("assays/RNA/data").unwrap();

    let version = VarLenUnicode::from_str("packed-uint-matrix-v2").unwrap_or_default();
    let attr = grp
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create("version")
        .unwrap();
    attr.write_scalar(&version).unwrap();

    let shape = Array1::from_vec(vec![4u32, 3u32]);
    grp.new_dataset::<u32>()
        .shape(shape.len())
        .create("shape")
        .unwrap()
        .write(&shape)
        .unwrap();

    let idxptr = Array1::from_vec(vec![0u64, 1, 3, 4, 4]);
    grp.new_dataset::<u64>()
        .shape(idxptr.len())
        .create("idxptr")
        .unwrap()
        .write(&idxptr)
        .unwrap();

    drop(file);

    let file = File::open(path).unwrap();
    assert_eq!(
        detect_sparse_group_kind(&file, "assays/RNA/data"),
        Some(SparseGroupKind::BpCells)
    );
}

#[tokio::test]
async fn test_open_shape() {
    if !golden_exists() {
        return;
    }
    let reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
    let (n_obs, n_vars) = reader.shape();
    assert_eq!(n_obs, 2700, "expected 2700 cells");
    assert_eq!(n_vars, 13714, "expected 13714 genes");
}

#[tokio::test]
async fn test_obs() {
    if !golden_exists() {
        return;
    }
    let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
    let obs = reader.obs().await.unwrap();
    assert_eq!(obs.index.len(), 2700);
    assert!(!obs.columns.is_empty());
    assert!(obs.columns.iter().any(|c| c.name == "nCount_RNA"));
}

#[tokio::test]
async fn test_var() {
    if !golden_exists() {
        return;
    }
    let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
    let var = reader.var().await.unwrap();
    assert_eq!(var.index.len(), 13714);
    assert!(!var.columns.is_empty(), "expected meta.features columns");
    assert!(var.columns.iter().any(|c| c.name == "vf.vst.mean"));
    assert!(var.columns.iter().any(|c| c.name == "vf.vst.variable"));
    // vf.vst.variable should be Bool
    let hvg_col = var
        .columns
        .iter()
        .find(|c| c.name == "vf.vst.variable")
        .unwrap();
    assert!(matches!(hvg_col.data, crate::ir::ColumnData::Bool(_)));
}

#[tokio::test]
async fn test_obsm() {
    if !golden_exists() {
        return;
    }
    let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
    let obsm = reader.obsm().await.unwrap();
    assert!(obsm.map.contains_key("X_pca"), "missing X_pca");
    assert!(obsm.map.contains_key("X_umap"), "missing X_umap");
    assert_eq!(obsm.map["X_pca"].shape, (2700, 30));
    assert_eq!(obsm.map["X_umap"].shape, (2700, 2));
}

#[tokio::test]
async fn test_stream_coverage() {
    if !golden_exists() {
        return;
    }
    let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
    let mut total_cells = 0usize;
    let mut total_nnz = 0usize;
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        total_cells += chunk.nrows;
        total_nnz += chunk.data.indices.len();
    }
    assert_eq!(total_cells, 2700);
    assert_eq!(total_nnz, 2282976);
}

#[tokio::test]
async fn test_h5seurat_roundtrip() {
    if !golden_exists() {
        return;
    }

    let mut reader = H5SeuratReader::open(GOLDEN, 500, None, None).unwrap();
    let (n_obs, n_vars) = reader.shape();

    let obs = reader.obs().await.unwrap();
    let var = reader.var().await.unwrap();
    let obsm = reader.obsm().await.unwrap();
    let uns = reader.uns().await.unwrap();

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let mut writer =
        H5SeuratWriter::create(&out, n_obs, n_vars, DataType::F32, None, None, None, false)
            .unwrap();
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

    // Re-open and verify with H5SeuratReader
    let mut rt = H5SeuratReader::open(&out, 500, None, None).unwrap();
    assert_eq!(rt.shape(), (n_obs, n_vars));

    let rt_obs = rt.obs().await.unwrap();
    assert_eq!(rt_obs.index.len(), n_obs);
    assert_eq!(rt_obs.index[0], obs.index[0]);
    assert_eq!(rt_obs.columns.len(), obs.columns.len());

    let rt_var = rt.var().await.unwrap();
    assert_eq!(rt_var.index.len(), n_vars);
    assert_eq!(rt_var.index[0], var.index[0]);

    let rt_obsm = rt.obsm().await.unwrap();
    assert!(
        rt_obsm.map.contains_key("X_pca"),
        "X_pca missing after roundtrip"
    );
    assert_eq!(rt_obsm.map["X_pca"].shape, obsm.map["X_pca"].shape);

    let mut total_nnz = 0usize;
    let mut stream = rt.x_stream();
    while let Some(chunk) = stream.next().await {
        total_nnz += chunk.unwrap().data.indices.len();
    }
    assert_eq!(total_nnz, 2282976, "nnz changed after H5Seurat roundtrip");
}

// -----------------------------------------------------------------------
// data-layer test (synthetic 3×4 matrix written with layer="data")
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_data_layer() {
    // Write a 3-cell × 4-gene matrix under the "data" layer and read it back.
    let n_obs = 3usize;
    let n_vars = 4usize;

    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("c{i}")).collect(),
        columns: vec![],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![],
    };
    // Non-zeros at (0,0),(0,2),(1,1),(1,3),(2,0),(2,3) → nnz = 6
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr: vec![0, 2, 4, 6],
            indices: vec![0, 2, 1, 3, 0, 3],
            data: TypedVec::F32(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]),
        },
    };

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let mut writer = H5SeuratWriter::create(
        &out,
        n_obs,
        n_vars,
        DataType::F32,
        None,
        Some("data"),
        None,
        false,
    )
    .unwrap();
    writer.write_obs(&obs).await.unwrap();
    writer.write_var(&var).await.unwrap();
    writer.write_obsm(&Embeddings::default()).await.unwrap();
    writer.write_uns(&UnsTable::default()).await.unwrap();
    writer.write_x_chunk(&chunk).await.unwrap();
    writer.finalize().await.unwrap();
    drop(writer);

    let mut reader = H5SeuratReader::open(&out, 100, None, Some("data")).unwrap();
    assert_eq!(reader.shape(), (n_obs, n_vars));

    let mut total_nnz = 0usize;
    let mut stream = reader.x_stream();
    while let Some(c) = stream.next().await {
        total_nnz += c.unwrap().data.indices.len();
    }
    assert_eq!(total_nnz, 6, "nnz mismatch for data layer");
}

// -----------------------------------------------------------------------
// assay-resolution regression: a file whose only assay is "SCT" (as in the
// Azimuth pbmc_multimodal reference) must resolve rather than crash on the
// hardcoded default "RNA" (missing assays/RNA → H5Gopen2 component not found).
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_missing_assay_falls_back() {
    let n_obs = 3usize;
    let n_vars = 4usize;
    let obs = ObsTable {
        index: (0..n_obs).map(|i| format!("c{i}")).collect(),
        columns: vec![],
    };
    let var = VarTable {
        index: (0..n_vars).map(|i| format!("g{i}")).collect(),
        columns: vec![],
    };
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr: vec![0, 2, 4, 6],
            indices: vec![0, 2, 1, 3, 0, 3],
            data: TypedVec::F32(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]),
        },
    };

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    // Write the matrix under an "SCT" assay only — there is no "RNA".
    let mut writer = H5SeuratWriter::create(
        &out,
        n_obs,
        n_vars,
        DataType::F32,
        Some("SCT"),
        None,
        None,
        false,
    )
    .unwrap();
    writer.write_obs(&obs).await.unwrap();
    writer.write_var(&var).await.unwrap();
    writer.write_obsm(&Embeddings::default()).await.unwrap();
    writer.write_uns(&UnsTable::default()).await.unwrap();
    writer.write_x_chunk(&chunk).await.unwrap();
    writer.finalize().await.unwrap();
    drop(writer);

    // Default assay "RNA" is absent — the reader must resolve to "SCT" instead
    // of failing. (Pre-fix this panicked on H5Gopen2 "component not found".)
    let mut reader = H5SeuratReader::open(&out, 100, None, None).unwrap();
    assert_eq!(reader.shape(), (n_obs, n_vars));

    let mut total_nnz = 0usize;
    let mut stream = reader.x_stream();
    while let Some(c) = stream.next().await {
        total_nnz += c.unwrap().data.indices.len();
    }
    assert_eq!(total_nnz, 6, "should stream the SCT assay's matrix");
}

// -----------------------------------------------------------------------
// uns / misc pass-through test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_uns_misc_passthrough() {
    // Create a minimal valid H5Seurat with H5SeuratWriter, then inject a
    // misc/ group.  Verify that uns() surfaces it as JSON.
    let obs = ObsTable {
        index: vec!["c0".into()],
        columns: vec![],
    };
    let var = VarTable {
        index: vec!["g0".into()],
        columns: vec![],
    };
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: 1,
        data: SparseMatrixCSR {
            shape: (1, 1),
            indptr: vec![0, 1],
            indices: vec![0],
            data: TypedVec::F32(vec![1.0]),
        },
    };

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let path = tmp.path().to_path_buf();

    let mut writer =
        H5SeuratWriter::create(&path, 1, 1, DataType::F32, None, None, None, false).unwrap();
    writer.write_obs(&obs).await.unwrap();
    writer.write_var(&var).await.unwrap();
    writer.write_obsm(&Embeddings::default()).await.unwrap();
    writer.write_uns(&UnsTable::default()).await.unwrap();
    writer.write_x_chunk(&chunk).await.unwrap();
    writer.finalize().await.unwrap();
    drop(writer);

    // Inject misc/ with a scalar dataset and a numeric array
    {
        let file = File::open_rw(&path).unwrap();
        let misc = file
            .group("misc")
            .or_else(|_| file.create_group("misc"))
            .unwrap();
        let ds = misc
            .new_dataset::<f64>()
            .shape(3)
            .create("weights")
            .unwrap();
        ds.write(&Array1::from_vec(vec![0.1f64, 0.2, 0.3])).unwrap();
    }

    let mut reader = H5SeuratReader::open(&path, 100, None, None).unwrap();
    let uns = reader.uns().await.unwrap();

    assert!(uns.raw.is_object(), "uns.raw should be a JSON object");
    assert!(
        uns.raw.get("weights").is_some(),
        "misc/weights missing from uns"
    );
    assert_eq!(uns.raw["weights"], serde_json::json!([0.1, 0.2, 0.3]));
}

// -----------------------------------------------------------------------
// Slot parity: layers, obsp, varm write → read roundtrip (synthetic)
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_slot_parity_roundtrip() {
    // 3 cells × 4 genes synthetic dataset.
    let n_obs = 3usize;
    let n_vars = 4usize;

    let obs = ObsTable {
        index: vec!["c0".into(), "c1".into(), "c2".into()],
        columns: vec![],
    };
    let var = VarTable {
        index: vec!["g0".into(), "g1".into(), "g2".into(), "g3".into()],
        columns: vec![],
    };

    // X: sparse 3×4, 4 non-zeros
    let x_chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_vars),
            indptr: vec![0, 2, 3, 4],
            indices: vec![0, 2, 1, 3],
            data: TypedVec::F32(vec![1.0, 2.0, 3.0, 4.0]),
        },
    };

    // layers["data"]: sparse chunk (n_vars × n_obs stored as CSR in H5Seurat convention)
    let layer_chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_vars,
        data: SparseMatrixCSR {
            shape: (n_vars, n_obs),
            indptr: vec![0, 1, 2, 3, 3],
            indices: vec![1, 0, 2],
            data: TypedVec::F32(vec![10.0, 20.0, 30.0]),
        },
    };
    let layer_meta = SparseMatrixMeta {
        name: "data".into(),
        shape: (n_vars, n_obs),
        indptr: vec![0, 1, 2, 3, 3],
    };

    // obsp["knn"]: 3×3 cell-cell graph
    let obsp_chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: SparseMatrixCSR {
            shape: (n_obs, n_obs),
            indptr: vec![0, 1, 2, 3],
            indices: vec![1, 2, 0],
            data: TypedVec::F32(vec![0.5, 0.6, 0.7]),
        },
    };
    let obsp_meta = SparseMatrixMeta {
        name: "knn".into(),
        shape: (n_obs, n_obs),
        indptr: vec![0, 1, 2, 3],
    };

    // varm["X_pca"]: 4 genes × 2 PCs
    let varm_mat = DenseMatrix {
        shape: (n_vars, 2),
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    };
    let mut varm = Varm::default();
    varm.map.insert("X_pca".into(), varm_mat.clone());

    let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let path = tmp.path().to_path_buf();

    let mut writer =
        H5SeuratWriter::create(&path, n_obs, n_vars, DataType::F32, None, None, None, false)
            .unwrap();
    writer.write_obs(&obs).await.unwrap();
    writer.write_var(&var).await.unwrap();
    writer.write_obsm(&Embeddings::default()).await.unwrap();
    writer.write_uns(&UnsTable::default()).await.unwrap();

    // Stream layer "data"
    writer
        .begin_sparse("layers", "data", &layer_meta)
        .await
        .unwrap();
    writer.write_sparse_chunk(&layer_chunk).await.unwrap();
    writer.end_sparse().await.unwrap();

    // Stream obsp "knn"
    writer
        .begin_sparse("obsp", "knn", &obsp_meta)
        .await
        .unwrap();
    writer.write_sparse_chunk(&obsp_chunk).await.unwrap();
    writer.end_sparse().await.unwrap();

    writer.write_varm(&varm).await.unwrap();
    writer.write_x_chunk(&x_chunk).await.unwrap();
    writer.finalize().await.unwrap();
    drop(writer);

    // Re-read and verify
    let mut reader = H5SeuratReader::open(&path, 100, None, None).unwrap();
    assert_eq!(reader.shape(), (n_obs, n_vars));

    let layer_metas = reader.layer_metas().await.unwrap();
    assert!(
        layer_metas.iter().any(|m| m.name == "data"),
        "layers['data'] missing"
    );
    // Stream the layer in its own block so the borrow on `reader` ends before
    // the next `&mut self` call.
    let all_indices: Vec<u32> = {
        let lm = layer_metas.iter().find(|m| m.name == "data").unwrap();
        assert_eq!(lm.shape, layer_meta.shape);
        assert_eq!(lm.indptr, layer_meta.indptr);
        let mut indices = Vec::new();
        let mut stream = reader.layer_stream(lm, 100);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            indices.extend_from_slice(&chunk.data.indices);
        }
        indices
    };
    assert_eq!(all_indices, layer_chunk.data.indices);

    let obsp_metas = reader.obsp_metas().await.unwrap();
    assert!(
        obsp_metas.iter().any(|m| m.name == "knn"),
        "obsp['knn'] missing"
    );
    let om = obsp_metas.iter().find(|m| m.name == "knn").unwrap();
    assert_eq!(om.shape, obsp_meta.shape);

    let rt_varm = reader.varm().await.unwrap();
    assert!(rt_varm.map.contains_key("X_pca"), "varm['X_pca'] missing");
    let rt_pca = &rt_varm.map["X_pca"];
    assert_eq!(rt_pca.shape, varm_mat.shape);
    for (a, b) in rt_pca.data.iter().zip(varm_mat.data.iter()) {
        assert!((a - b).abs() < 1e-10, "varm data mismatch: {a} vs {b}");
    }
}

// --- Norman obs round-trip: H5AD → H5Seurat → read back ---

#[tokio::test]
async fn test_norman_obs_roundtrip() {
    use crate::h5ad::H5AdReader;
    use tempfile::NamedTempFile;

    if !norman_exists() {
        return;
    }

    // Read the Norman subset H5AD.
    let fixture = std::path::Path::new(NORMAN_FIXTURE);
    let mut src = H5AdReader::open(fixture, 500).unwrap();
    let (n_obs, n_vars) = src.shape();
    let src_obs = src.obs().await.unwrap();
    let src_var = src.var().await.unwrap();
    let src_obsm = src.obsm().await.unwrap();
    let src_uns = src.uns().await.unwrap();

    let src_col_names: Vec<&str> = src_obs.columns.iter().map(|c| c.name.as_str()).collect();
    eprintln!("source obs columns: {src_col_names:?}");

    // Convert to H5Seurat.
    let tmp = NamedTempFile::with_suffix(".h5seurat").unwrap();
    let out = tmp.path().to_path_buf();

    let mut writer =
        H5SeuratWriter::create(&out, n_obs, n_vars, src.dtype(), None, None, None, false).unwrap();
    writer.write_obs(&src_obs).await.unwrap();
    writer.write_var(&src_var).await.unwrap();
    writer.write_obsm(&src_obsm).await.unwrap();
    writer.write_uns(&src_uns).await.unwrap();
    {
        let mut stream = src.x_stream();
        while let Some(chunk) = stream.next().await {
            writer.write_x_chunk(&chunk.unwrap()).await.unwrap();
        }
    }
    // Skip layers — the "counts" layer would collide with the X path in H5Seurat.
    // Obs fidelity is what this test exercises.
    writer.finalize().await.unwrap();
    drop(writer);

    // Read back via H5SeuratReader and check obs fidelity.
    let mut rt = H5SeuratReader::open(&out, 500, None, None).unwrap();
    assert_eq!(
        rt.shape(),
        (n_obs, n_vars),
        "shape mismatch after round-trip"
    );

    let rt_obs = rt.obs().await.unwrap();
    assert_eq!(rt_obs.index.len(), n_obs, "obs index length mismatch");

    // Every source column must survive the round-trip.
    for src_col in &src_obs.columns {
        let rt_col = rt_obs
            .columns
            .iter()
            .find(|c| c.name == src_col.name)
            .unwrap_or_else(|| panic!("obs column '{}' missing after round-trip", src_col.name));

        // Dtype class must be preserved.
        let src_kind = std::mem::discriminant(&src_col.data);
        let rt_kind = std::mem::discriminant(&rt_col.data);
        assert_eq!(
            src_kind, rt_kind,
            "obs column '{}' changed dtype after round-trip",
            src_col.name
        );
    }

    eprintln!(
        "norman obs round-trip OK: {n_obs} cells, {} columns",
        rt_obs.columns.len()
    );
}
