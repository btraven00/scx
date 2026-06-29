//! Self-contained tests: generate a small Tahoe-like Parquet fixture in
//! memory, serve it from `object_store::InMemory`, read it back through
//! [`ParquetReader`]. No network and no committed golden file — coverage is
//! Codecov-gated and golden fixtures are gitignored, so the fixture is built
//! from scratch each run.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Builder, Int64Builder, ListBuilder, StringArray};
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
    let barcodes: ArrayRef =
        Arc::new(StringArray::from(vec!["cell_0", "cell_1", "cell_2", "cell_3"]));
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
    let mut reader = ParquetReader::open(store, path, N_VARS, 3)
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
    let mut reader = ParquetReader::open(store, path, N_VARS, 1024)
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
    let mut reader = ParquetReader::open(store, path, 3, 1024)
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
