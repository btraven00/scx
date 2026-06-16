use super::format::*;
use super::meta::*;
use super::*;
use crate::dtype::*;
use crate::ir::*;
use crate::stream::DatasetReader;
use std::fs;

fn synthetic_dataset() -> SingleCellDataset {
    let x = SparseMatrixCSR {
        shape: (3, 4),
        indptr: vec![0, 2, 3, 5],
        indices: vec![0, 2, 1, 0, 3],
        data: TypedVec::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    };
    let obs = ObsTable {
        index: vec!["cell1".into(), "cell2".into(), "cell3".into()],
        columns: vec![
            Column {
                name: "count".into(),
                data: ColumnData::Int(vec![10, 20, 30]),
            },
            Column {
                name: "score".into(),
                data: ColumnData::Float(vec![1.1, 2.2, 3.3]),
            },
            Column {
                name: "active".into(),
                data: ColumnData::Bool(vec![true, false, true]),
            },
            Column {
                name: "label".into(),
                data: ColumnData::Categorical {
                    codes: vec![0, 1, 0],
                    levels: vec!["A".into(), "B".into()],
                },
            },
            Column {
                name: "notes".into(),
                data: ColumnData::String(vec!["x".into(), "y".into(), "z".into()]),
            },
        ],
    };
    let var = VarTable {
        index: vec!["g1".into(), "g2".into(), "g3".into(), "g4".into()],
        columns: vec![Column {
            name: "highly_variable".into(),
            data: ColumnData::Bool(vec![true, false, true, false]),
        }],
    };
    let obsm = Embeddings {
        map: [(
            "X_pca".to_string(),
            DenseMatrix {
                shape: (3, 2),
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        )]
        .into_iter()
        .collect(),
    };
    let varm = Varm {
        map: [(
            "PCs".to_string(),
            DenseMatrix {
                shape: (4, 2),
                data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            },
        )]
        .into_iter()
        .collect(),
    };
    let layers = Layers {
        map: [(
            "spliced".to_string(),
            SparseMatrixCSR {
                shape: (3, 4),
                indptr: vec![0, 1, 2, 3],
                indices: vec![1, 2, 3],
                data: TypedVec::F32(vec![9.0, 8.0, 7.0]),
            },
        )]
        .into_iter()
        .collect(),
    };
    SingleCellDataset {
        x,
        x_dtype: DataType::F32,
        obs,
        var,
        obsm,
        uns: UnsTable::default(),
        layers,
        obsp: Obsp::default(),
        varp: Varp::default(),
        varm,
    }
}

#[test]
fn test_npy_header_alignment() {
    for &n in &[1usize, 100, 1_000_000, 5_000_000_000] {
        let mut buf = Vec::new();
        write_npy_header(&mut buf, "<f4", &[n]).unwrap();
        assert_eq!(buf.len() % 64, 0, "header not multiple of 64 for n={n}");
    }
    let mut buf = Vec::new();
    write_npy_header(&mut buf, "<f8", &[2638, 50]).unwrap();
    assert_eq!(buf.len() % 64, 0);
}

#[test]
fn test_full_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let ds = synthetic_dataset();
    NpyIrWriter::write(dir.path(), &ds, &SlotFilter::all()).unwrap();

    // Verify nested layout
    assert!(dir.path().join("X/data.npy").exists());
    assert!(dir.path().join("X/indices.npy").exists());
    assert!(dir.path().join("X/indptr.npy").exists());
    assert!(dir.path().join("obs/count.npy").exists());
    assert!(dir.path().join("obs/label_codes.npy").exists());
    assert!(dir.path().join("obs/label_levels.txt").exists());
    assert!(dir.path().join("obs/notes_strings.txt").exists());
    assert!(dir.path().join("obsm/X_pca.npy").exists());
    assert!(dir.path().join("varm/PCs.npy").exists());
    assert!(dir.path().join("layers/spliced/data.npy").exists());

    let reader = NpyIrReader::open(dir.path(), 1000).unwrap();
    let got = reader.into_dataset();

    assert_eq!(got.x.shape, ds.x.shape);
    match (&got.x.data, &ds.x.data) {
        (TypedVec::F32(a), TypedVec::F32(b)) => assert_eq!(a, b),
        _ => panic!("dtype mismatch"),
    }
    assert_eq!(got.x.indices, ds.x.indices);
    assert_eq!(got.x.indptr, ds.x.indptr);
    assert_eq!(got.obs.index, ds.obs.index);
    assert_eq!(got.var.index, ds.var.index);
    assert_eq!(got.obs.columns.len(), ds.obs.columns.len());

    for (g, e) in got.obs.columns.iter().zip(ds.obs.columns.iter()) {
        assert_eq!(g.name, e.name);
        match (&g.data, &e.data) {
            (ColumnData::Int(a), ColumnData::Int(b)) => assert_eq!(a, b),
            (ColumnData::Float(a), ColumnData::Float(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    assert!((x - y).abs() < 1e-10);
                }
            }
            (ColumnData::Bool(a), ColumnData::Bool(b)) => assert_eq!(a, b),
            (ColumnData::String(a), ColumnData::String(b)) => assert_eq!(a, b),
            (
                ColumnData::Categorical {
                    codes: ca,
                    levels: la,
                },
                ColumnData::Categorical {
                    codes: cb,
                    levels: lb,
                },
            ) => {
                assert_eq!(ca, cb);
                assert_eq!(la, lb);
            }
            _ => panic!("column kind mismatch for '{}'", g.name),
        }
    }
    assert_eq!(got.obsm.map["X_pca"].data, ds.obsm.map["X_pca"].data);
    assert_eq!(got.varm.map["PCs"].data, ds.varm.map["PCs"].data);
    assert_eq!(
        got.layers.map["spliced"].indices,
        ds.layers.map["spliced"].indices
    );
}

#[test]
fn test_meta_json_is_rich() {
    let dir = tempfile::tempdir().unwrap();
    let ds = synthetic_dataset();
    NpyIrWriter::write(dir.path(), &ds, &SlotFilter::all()).unwrap();
    let meta: Meta = read_json(&dir.path().join("meta.json")).unwrap();

    // X carries shape + nnz + dtype
    let xm = meta.x.as_ref().unwrap();
    assert_eq!(xm.shape, [3, 4]);
    assert_eq!(xm.nnz, 5);
    assert_eq!(xm.dtype, "f32");

    // obs columns carry kind + shape
    let count_meta = meta.obs.iter().find(|c| c.name == "count").unwrap();
    assert_eq!(count_meta.kind, "int");
    assert_eq!(count_meta.shape, [3]);

    // categorical has n_levels
    let label_meta = meta.obs.iter().find(|c| c.name == "label").unwrap();
    assert_eq!(label_meta.kind, "categorical");
    assert_eq!(label_meta.n_levels, Some(2));

    // obsm carries shape + dtype
    let pca_meta = meta.obsm.get("X_pca").unwrap();
    assert_eq!(pca_meta.shape, [3, 2]);
    assert_eq!(pca_meta.dtype, "f64");

    // layers carry shape + nnz + dtype
    let spliced_meta = meta.layers.get("spliced").unwrap();
    assert_eq!(spliced_meta.shape, [3, 4]);
    assert_eq!(spliced_meta.nnz, 3);
}

#[test]
fn test_selective_only_x() {
    let dir = tempfile::tempdir().unwrap();
    let ds = synthetic_dataset();
    NpyIrWriter::write(dir.path(), &ds, &SlotFilter::from_only("X,obs_index")).unwrap();

    assert!(dir.path().join("X/data.npy").exists());
    assert!(dir.path().join("obs_index.txt").exists());
    assert!(!dir.path().join("var_index.txt").exists());
    assert!(!dir.path().join("obsm/X_pca.npy").exists());
    assert!(!dir.path().join("obs").exists());

    let reader = NpyIrReader::open(dir.path(), 5000).unwrap();
    let got = reader.into_dataset();
    assert_eq!(got.obs.index, ds.obs.index);
    assert!(got.obs.columns.is_empty());
    assert!(got.obsm.map.is_empty());
}

#[test]
fn test_selective_exclude() {
    let dir = tempfile::tempdir().unwrap();
    let ds = synthetic_dataset();
    NpyIrWriter::write(
        dir.path(),
        &ds,
        &SlotFilter::from_exclude("layers,obsp,varp"),
    )
    .unwrap();

    assert!(dir.path().join("X/data.npy").exists());
    assert!(dir.path().join("obsm/X_pca.npy").exists());
    assert!(!dir.path().join("layers").exists());
}

#[tokio::test]
async fn test_stream_writer_matches_materialised() {
    // Materialise a synthetic dataset to NPY via the all-in-memory writer,
    // re-open it as a reader, then re-write via the streaming writer.
    // Both outputs must be byte-identical (and round-trip back to the
    // same SingleCellDataset).
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let ds = synthetic_dataset();

    NpyIrWriter::write(src.path(), &ds, &SlotFilter::all()).unwrap();
    let mut reader = NpyIrReader::open(src.path(), 2).unwrap();
    NpyIrWriter::stream(dst.path(), &mut reader, &SlotFilter::all(), 2)
        .await
        .unwrap();

    // Compare X arrays byte-for-byte — single-pass path runs here because
    // NpyIrReader exposes a full x_indptr.
    for sub in &["X/data.npy", "X/indices.npy", "X/indptr.npy"] {
        let a = fs::read(src.path().join(sub)).unwrap();
        let b = fs::read(dst.path().join(sub)).unwrap();
        assert_eq!(a, b, "byte mismatch in {sub}");
    }

    // Layer parity.
    for sub in &[
        "layers/spliced/data.npy",
        "layers/spliced/indices.npy",
        "layers/spliced/indptr.npy",
    ] {
        let a = fs::read(src.path().join(sub)).unwrap();
        let b = fs::read(dst.path().join(sub)).unwrap();
        assert_eq!(a, b, "byte mismatch in {sub}");
    }

    // Round-trip the streamed snapshot back to a dataset.
    let got = NpyIrReader::open(dst.path(), 1000).unwrap().into_dataset();
    assert_eq!(got.x.shape, ds.x.shape);
    assert_eq!(got.x.indices, ds.x.indices);
    assert_eq!(got.x.indptr, ds.x.indptr);
    assert_eq!(got.obs.index, ds.obs.index);
    assert_eq!(got.var.index, ds.var.index);
    assert_eq!(got.obs.columns.len(), ds.obs.columns.len());
    assert_eq!(got.obsm.map["X_pca"].data, ds.obsm.map["X_pca"].data);
}

#[tokio::test]
async fn test_dataset_reader_stream() {
    use futures::StreamExt;
    let dir = tempfile::tempdir().unwrap();
    let ds = synthetic_dataset();
    NpyIrWriter::write(dir.path(), &ds, &SlotFilter::all()).unwrap();

    let mut reader = NpyIrReader::open(dir.path(), 2).unwrap();
    let mut chunks = Vec::new();
    let mut stream = reader.x_stream();
    while let Some(c) = stream.next().await {
        chunks.push(c.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].row_offset, 0);
    assert_eq!(chunks[0].nrows, 2);
    assert_eq!(chunks[1].row_offset, 2);
    assert_eq!(chunks[1].nrows, 1);
    let total_nnz: usize = chunks.iter().map(|c| c.data.indices.len()).sum();
    assert_eq!(total_nnz, ds.x.indices.len());
}
