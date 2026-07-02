//! Self-contained tests: generate a small Tahoe-like Parquet fixture in
//! memory, serve it from `object_store::InMemory`, read it back through
//! [`ParquetReader`]. No network and no committed golden file — coverage is
//! Codecov-gated and golden fixtures are gitignored, so the fixture is built
//! from scratch each run.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float32Builder, Int64Array, Int64Builder, ListBuilder, StringArray,
};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use object_store::memory::InMemory;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ArrowWriter;

use super::ParquetReader;
use crate::dtype::{DataType, TypedVec};
use crate::ir::ColumnData;
use crate::stream::DatasetReader;

const N_VARS: usize = 6;

/// The fixture: 4 cells × 6 genes, sparse, plus two obs string columns and a
/// barcode column. Rows mirror the Tahoe `expression_data` layout (parallel
/// `genes`/`expressions` lists), but gene ids are direct 0-based indices.
fn expected_rows() -> Vec<(Vec<i64>, Vec<f32>)> {
    vec![
        (vec![0, 2, 5], vec![1.0, 3.0, 9.0]),
        (vec![], vec![]), // empty cell (all-zero row)
        (vec![1, 4], vec![2.0, 5.0]),
        (vec![0, 1, 2, 3, 4, 5], vec![7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
    ]
}

fn fixture_bytes() -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "genes",
            ArrowType::List(Arc::new(Field::new("item", ArrowType::Int64, true))),
            true,
        ),
        Field::new(
            "expressions",
            ArrowType::List(Arc::new(Field::new("item", ArrowType::Float32, true))),
            true,
        ),
        Field::new("BARCODE_SUB_LIB_ID", ArrowType::Utf8, false),
        Field::new("drug", ArrowType::Utf8, false),
    ]));

    let rows = expected_rows();

    let mut genes_b = ListBuilder::new(Int64Builder::new());
    let mut expr_b = ListBuilder::new(Float32Builder::new());
    for (g, e) in &rows {
        for &gi in g {
            genes_b.values().append_value(gi);
        }
        genes_b.append(true);
        for &ev in e {
            expr_b.values().append_value(ev);
        }
        expr_b.append(true);
    }
    let genes: ArrayRef = Arc::new(genes_b.finish());
    let exprs: ArrayRef = Arc::new(expr_b.finish());
    let barcodes: ArrayRef = Arc::new(StringArray::from(vec![
        "cell_0", "cell_1", "cell_2", "cell_3",
    ]));
    let drugs: ArrayRef = Arc::new(StringArray::from(vec![
        "DMSO_TF", "DMSO_TF", "drugA", "drugB",
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![genes, exprs, barcodes, drugs])
        .expect("build record batch");

    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, None).expect("arrow writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
    buf
}

async fn fixture_store() -> (Arc<dyn ObjectStore>, StorePath) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = StorePath::from("tahoe/expression_data.parquet");
    store
        .put(&path, fixture_bytes().into())
        .await
        .expect("put fixture");
    (store, path)
}

#[tokio::test]
async fn streams_csr_matching_fixture() {
    let (store, path) = fixture_store().await;
    // chunk_size = 3 forces more than one batch over 4 rows, exercising the
    // row_offset accumulation across chunks.
    let mut reader = ParquetReader::open(store, path, Some(N_VARS), None, 3)
        .await
        .expect("open");

    assert_eq!(reader.shape(), (4, N_VARS));
    assert_eq!(reader.dtype(), DataType::F32);

    // Reassemble the streamed chunks into dense rows and compare to the fixture.
    let mut dense = vec![vec![0.0f32; N_VARS]; 4];
    let mut seen_rows = 0;
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk");
        assert_eq!(chunk.data.shape.1, N_VARS);
        assert_eq!(chunk.row_offset, seen_rows);
        let TypedVec::F32(values) = &chunk.data.data else {
            panic!("expected f32 CSR data");
        };
        for local_row in 0..chunk.nrows {
            let global = chunk.row_offset + local_row;
            let start = chunk.data.indptr[local_row] as usize;
            let end = chunk.data.indptr[local_row + 1] as usize;
            for k in start..end {
                dense[global][chunk.data.indices[k] as usize] = values[k];
            }
        }
        seen_rows += chunk.nrows;
    }
    assert_eq!(seen_rows, 4);

    for (row, (genes, exprs)) in expected_rows().iter().enumerate() {
        let mut want = vec![0.0f32; N_VARS];
        for (&g, &v) in genes.iter().zip(exprs) {
            want[g as usize] = v;
        }
        assert_eq!(dense[row], want, "row {row} mismatch");
    }
}

#[tokio::test]
async fn reads_scalar_obs_columns() {
    let (store, path) = fixture_store().await;
    let mut reader = ParquetReader::open(store, path, Some(N_VARS), None, 1024)
        .await
        .expect("open");

    let obs = reader.obs().await.expect("obs");

    // The barcode column is promoted to the index, not kept as a column.
    assert_eq!(obs.index, vec!["cell_0", "cell_1", "cell_2", "cell_3"]);

    let drug = obs
        .columns
        .iter()
        .find(|c| c.name == "drug")
        .expect("drug column present");
    match &drug.data {
        ColumnData::String(v) => {
            assert_eq!(v, &vec!["DMSO_TF", "DMSO_TF", "drugA", "drugB"]);
        }
        other => panic!("drug should be String, got {other:?}"),
    }

    // List columns (genes/expressions) must NOT leak into obs.
    assert!(obs.columns.iter().all(|c| c.name != "genes"));
    assert!(obs.columns.iter().all(|c| c.name != "expressions"));
}

#[tokio::test]
async fn rejects_gene_index_out_of_range() {
    // n_vars deliberately too small: gene id 5 exceeds it -> Net error.
    let (store, path) = fixture_store().await;
    let mut reader = ParquetReader::open(store, path, Some(3), None, 1024)
        .await
        .expect("open");

    let mut stream = reader.x_stream();
    let mut hit_err = false;
    while let Some(chunk) = stream.next().await {
        if chunk.is_err() {
            hit_err = true;
            break;
        }
    }
    assert!(hit_err, "expected an out-of-range gene index error");
}

// --- Dense layout (one float column per gene) ---

/// 3 cells × 4 genes, dense float columns + a string id column.
fn dense_fixture_bytes() -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("cell_id", ArrowType::Utf8, false),
        Field::new("g0", ArrowType::Float32, false),
        Field::new("g1", ArrowType::Float32, false),
        Field::new("g2", ArrowType::Float32, false),
        Field::new("g3", ArrowType::Float32, false),
    ]));
    let cell_id: ArrayRef = Arc::new(StringArray::from(vec!["c0", "c1", "c2"]));
    let g0: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 0.0, 4.0]));
    let g1: ArrayRef = Arc::new(Float32Array::from(vec![0.0, 2.0, 0.0]));
    let g2: ArrayRef = Arc::new(Float32Array::from(vec![0.0, 0.0, 5.0]));
    let g3: ArrayRef = Arc::new(Float32Array::from(vec![3.0, 0.0, 0.0]));

    let batch = RecordBatch::try_new(schema.clone(), vec![cell_id, g0, g1, g2, g3]).expect("batch");
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
    buf
}

#[tokio::test]
async fn dense_layout_round_trips_and_derives_n_vars() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = StorePath::from("dense.parquet");
    store
        .put(&path, dense_fixture_bytes().into())
        .await
        .expect("put");

    // n_vars = None: the dense layout derives it from the 4 float columns.
    let mut reader = ParquetReader::open(store, path, None, None, 1024)
        .await
        .expect("open");
    assert_eq!(reader.shape(), (3, 4));
    assert_eq!(reader.dtype(), DataType::F32);

    // obs: the string column survives; gene columns are excluded.
    let obs = reader.obs().await.expect("obs");
    assert!(obs.columns.iter().any(|c| c.name == "cell_id"));
    assert!(obs.columns.iter().all(|c| !c.name.starts_with('g')));

    let expected = [
        [1.0f32, 0.0, 0.0, 3.0],
        [0.0, 2.0, 0.0, 0.0],
        [4.0, 0.0, 5.0, 0.0],
    ];
    let mut dense = vec![vec![0.0f32; 4]; 3];
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk");
        let TypedVec::F32(values) = &chunk.data.data else {
            panic!("expected f32 CSR data");
        };
        for local in 0..chunk.nrows {
            let g = chunk.row_offset + local;
            let start = chunk.data.indptr[local] as usize;
            let end = chunk.data.indptr[local + 1] as usize;
            for k in start..end {
                dense[g][chunk.data.indices[k] as usize] = values[k];
            }
        }
    }
    for (r, want) in expected.iter().enumerate() {
        assert_eq!(dense[r], want.to_vec(), "dense row {r} mismatch");
    }
}

// --- Schema sniffing ---

use super::layout::ParquetLayout;

fn field(name: &str, dt: ArrowType) -> Field {
    Field::new(name, dt, true)
}

fn list_field(name: &str, item: ArrowType) -> Field {
    Field::new(
        name,
        ArrowType::List(Arc::new(Field::new("item", item, true))),
        true,
    )
}

#[test]
fn sniff_detects_per_cell_lists() {
    let schema = Schema::new(vec![
        list_field("genes", ArrowType::Int64),
        list_field("expressions", ArrowType::Float32),
        field("drug", ArrowType::Utf8),
    ]);
    let layout = ParquetLayout::sniff(&schema).expect("sniff");
    assert!(matches!(layout, ParquetLayout::PerCellLists { .. }));
    assert_eq!(layout.intrinsic_n_vars(), None);
}

#[test]
fn sniff_detects_dense() {
    let schema = Schema::new(vec![
        field("cell_id", ArrowType::Utf8),
        field("g0", ArrowType::Float32),
        field("g1", ArrowType::Float64),
    ]);
    let layout = ParquetLayout::sniff(&schema).expect("sniff");
    assert_eq!(layout.intrinsic_n_vars(), Some(2));
    assert!(layout.is_matrix_column("g0"));
    assert!(!layout.is_matrix_column("cell_id"));
}

#[test]
fn sniff_rejects_unrecognized_schema() {
    // No list pair and no float columns → unknown encoding (e.g. long-format).
    let schema = Schema::new(vec![
        field("cell_id", ArrowType::Utf8),
        field("gene_id", ArrowType::Int64),
        field("count", ArrowType::Int64),
    ]);
    assert!(ParquetLayout::sniff(&schema).is_err());
}

// --- Gene dictionary (token → column remap) ---

/// gene_metadata fixture: 3 genes with non-positional token ids (3, 7, 9).
fn gene_dict_bytes() -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("token_id", ArrowType::Int64, false),
        Field::new("ensembl_id", ArrowType::Utf8, false),
        Field::new("gene_symbol", ArrowType::Utf8, false),
    ]));
    let token_id: ArrayRef = Arc::new(Int64Array::from(vec![3i64, 7, 9]));
    let ensembl: ArrayRef = Arc::new(StringArray::from(vec!["ENSG_A", "ENSG_B", "ENSG_C"]));
    let symbol: ArrayRef = Arc::new(StringArray::from(vec!["GENEA", "GENEB", "GENEC"]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![token_id, ensembl, symbol]).expect("batch");
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
    buf
}

#[tokio::test]
async fn gene_dict_load_builds_dense_map_and_var() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = StorePath::from("gene_metadata.parquet");
    store
        .put(&path, gene_dict_bytes().into())
        .await
        .expect("put");

    let dict = super::GeneDict::load(store, path, 1024)
        .await
        .expect("load");

    assert_eq!(dict.n_vars(), 3);
    assert_eq!(dict.var.index, vec!["ENSG_A", "ENSG_B", "ENSG_C"]);
    // Dense map sized to max(token)=9 → 10 slots; tokens map to row index.
    assert_eq!(dict.token_to_col.len(), 10);
    assert_eq!(dict.token_to_col[3], 0);
    assert_eq!(dict.token_to_col[7], 1);
    assert_eq!(dict.token_to_col[9], 2);
    // Reserved / marker / gap tokens are sentinels.
    assert_eq!(dict.token_to_col[0], -1);
    assert_eq!(dict.token_to_col[1], -1);
    assert_eq!(dict.token_to_col[5], -1);
}

/// Build a one-cell per-cell-list batch (genes + expressions lists).
fn single_cell_batch(genes: &[i64], exprs: &[f32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "genes",
            ArrowType::List(Arc::new(Field::new("item", ArrowType::Int64, true))),
            true,
        ),
        Field::new(
            "expressions",
            ArrowType::List(Arc::new(Field::new("item", ArrowType::Float32, true))),
            true,
        ),
    ]));
    let mut gb = ListBuilder::new(Int64Builder::new());
    let mut eb = ListBuilder::new(Float32Builder::new());
    for &g in genes {
        gb.values().append_value(g);
    }
    gb.append(true);
    for &e in exprs {
        eb.values().append_value(e);
    }
    eb.append(true);
    let g: ArrayRef = Arc::new(gb.finish());
    let e: ArrayRef = Arc::new(eb.finish());
    RecordBatch::try_new(schema, vec![g, e]).expect("batch")
}

#[test]
fn per_cell_remap_maps_tokens_and_drops_marker() {
    // tokens 3→0, 7→1, 9→2; everything else sentinel (incl. marker token 1).
    let map: Vec<i32> = vec![-1, -1, -1, 0, -1, -1, -1, 1, -1, 2];
    let layout = ParquetLayout::PerCellLists {
        genes: "genes".to_string(),
        exprs: "expressions".to_string(),
    };
    // cell: [marker, gene3, gene9]
    let batch = single_cell_batch(&[1, 3, 9], &[-2.0, 5.0, 2.0]);
    let chunk = layout
        .batch_to_chunk(&batch, 3, 0, Some(&map))
        .expect("chunk");

    assert_eq!(chunk.nrows, 1);
    assert_eq!(chunk.data.indptr, vec![0, 2]); // marker dropped → 2 entries
    assert_eq!(chunk.data.indices, vec![0, 2]); // token3→col0, token9→col2
    match &chunk.data.data {
        TypedVec::F32(v) => assert_eq!(v, &vec![5.0, 2.0]),
        other => panic!("expected f32, got {other:?}"),
    }
}

#[test]
fn per_cell_without_map_uses_direct_index() {
    // No token map → integers are direct column indices (legacy behavior).
    let layout = ParquetLayout::PerCellLists {
        genes: "genes".to_string(),
        exprs: "expressions".to_string(),
    };
    let batch = single_cell_batch(&[0, 2], &[1.0, 3.0]);
    let chunk = layout.batch_to_chunk(&batch, 3, 0, None).expect("chunk");
    assert_eq!(chunk.data.indices, vec![0, 2]);
    match &chunk.data.data {
        TypedVec::F32(v) => assert_eq!(v, &vec![1.0, 3.0]),
        other => panic!("expected f32, got {other:?}"),
    }
}
