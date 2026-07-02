//! End-to-end: `scx convert <local.parquet> out.h5ad --n-vars N`.
//!
//! Self-contained — builds a small Tahoe-shaped Parquet fixture in a temp dir,
//! runs the real `scx` binary (which constructs the tokio runtime and routes
//! through the object-store transport's LocalFileSystem branch), then reads the
//! output `.h5ad` back through scx-core to assert the matrix round-trips. No
//! network, no committed golden file.
//!
//! Gated on the `net` feature: without it the binary has no Parquet support and
//! the fixture deps aren't relevant.
#![cfg(feature = "net")]

use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Builder, Int64Array, Int64Builder, ListBuilder, StringArray};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use futures::StreamExt;
use scx_core::h5ad::H5AdReader;
use scx_core::ir::ColumnData;
use scx_core::stream::DatasetReader;

const N_VARS: usize = 6;

/// 4 cells × 6 genes, sparse — must match the dense expectation below.
fn rows() -> Vec<(Vec<i64>, Vec<f32>)> {
    vec![
        (vec![0, 2, 5], vec![1.0, 3.0, 9.0]),
        (vec![], vec![]),
        (vec![1, 4], vec![2.0, 5.0]),
        (vec![0, 1, 2, 3, 4, 5], vec![7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
    ]
}

fn write_fixture(path: &std::path::Path) {
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

    let mut genes_b = ListBuilder::new(Int64Builder::new());
    let mut expr_b = ListBuilder::new(Float32Builder::new());
    for (g, e) in rows() {
        for gi in g {
            genes_b.values().append_value(gi);
        }
        genes_b.append(true);
        for ev in e {
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

    let batch = RecordBatch::try_new(schema.clone(), vec![genes, exprs, barcodes, drugs]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn convert_local_parquet_to_h5ad() {
    let dir = tempfile::tempdir().unwrap();
    let parquet = dir.path().join("expression_data.parquet");
    let h5ad = dir.path().join("out.h5ad");
    write_fixture(&parquet);

    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("convert")
        .arg(&parquet)
        .arg(&h5ad)
        .arg("--n-vars")
        .arg(N_VARS.to_string())
        .status()
        .expect("run scx convert");
    assert!(status.success(), "scx convert exited with {status}");
    assert!(h5ad.exists(), "output h5ad was not written");

    // Read the output back and reconstruct the dense matrix.
    let mut reader = H5AdReader::open(&h5ad, 5000).expect("open output h5ad");
    assert_eq!(reader.shape(), (4, N_VARS));

    let dense = futures::executor::block_on(async {
        let mut dense = vec![vec![0.0f64; N_VARS]; 4];
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("chunk");
            let values = chunk.data.data.to_f64();
            for local in 0..chunk.nrows {
                let g = chunk.row_offset + local;
                let start = chunk.data.indptr[local] as usize;
                let end = chunk.data.indptr[local + 1] as usize;
                for k in start..end {
                    dense[g][chunk.data.indices[k] as usize] = values[k];
                }
            }
        }
        dense
    });

    for (r, (genes, exprs)) in rows().iter().enumerate() {
        let mut want = vec![0.0f64; N_VARS];
        for (&g, &v) in genes.iter().zip(exprs) {
            want[g as usize] = v as f64;
        }
        assert_eq!(dense[r], want, "row {r} mismatch after round-trip");
    }
}

// ===========================================================================
// Acceptance test for the gene-token remap story.
//
// Two fixtures, deliberately constructed so that the CURRENT direct-index
// behavior would be WRONG: the token ids (3, 7, 9) are not their column
// positions (0, 1, 2), and tokens 7/9 exceed the 3-gene axis — so without
// remapping the conversion would mis-column or error. The test only passes if
// the dictionary is actually used: tokens remapped, marker dropped, var filled,
// n_vars derived (note: NO --n-vars is passed).
// ===========================================================================

/// gene_metadata fixture: 3 genes with non-positional token ids.
fn write_gene_dict(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("token_id", ArrowType::Int64, false),
        Field::new("ensembl_id", ArrowType::Utf8, false),
        Field::new("gene_symbol", ArrowType::Utf8, false),
    ]));
    let token_id: ArrayRef = Arc::new(Int64Array::from(vec![3i64, 7, 9])); // → cols 0,1,2
    let ensembl: ArrayRef = Arc::new(StringArray::from(vec!["ENSG_A", "ENSG_B", "ENSG_C"]));
    let symbol: ArrayRef = Arc::new(StringArray::from(vec!["GENEA", "GENEB", "GENEC"]));

    let batch = RecordBatch::try_new(schema.clone(), vec![token_id, ensembl, symbol]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// expression_data fixture: 2 cells, each led by marker token 1 (value -2.0).
fn write_remap_matrix(path: &std::path::Path) {
    // (genes, expressions) per cell — token 1 is the marker to be dropped.
    let cells: Vec<(Vec<i64>, Vec<f32>)> = vec![
        (vec![1, 3, 9], vec![-2.0, 5.0, 2.0]),
        (vec![1, 7], vec![-2.0, 4.0]),
    ];
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
    let mut genes_b = ListBuilder::new(Int64Builder::new());
    let mut expr_b = ListBuilder::new(Float32Builder::new());
    for (g, e) in &cells {
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

    let batch = RecordBatch::try_new(schema.clone(), vec![genes, exprs]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn convert_with_gene_remap() {
    let dir = tempfile::tempdir().unwrap();
    let matrix = dir.path().join("expression_data.parquet");
    let genes = dir.path().join("gene_metadata.parquet");
    let h5ad = dir.path().join("out.h5ad");
    write_remap_matrix(&matrix);
    write_gene_dict(&genes);

    // NOTE: no --n-vars; it must be derived from the gene dictionary.
    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("convert")
        .arg(&matrix)
        .arg(&h5ad)
        .arg("--genes")
        .arg(&genes)
        .status()
        .expect("run scx convert --genes");
    assert!(status.success(), "scx convert --genes exited with {status}");

    let mut reader = H5AdReader::open(&h5ad, 5000).expect("open output h5ad");
    // n_vars derived from the dictionary (3 genes), n_obs from the matrix (2).
    assert_eq!(reader.shape(), (2, 3));

    // var axis populated from the dictionary's ensembl_id.
    let var = futures::executor::block_on(reader.var()).expect("var");
    assert_eq!(var.index, vec!["ENSG_A", "ENSG_B", "ENSG_C"]);
    // gene_symbol carried through as a var column.
    assert!(var.columns.iter().any(|c| c.name == "gene_symbol"
        && matches!(&c.data, ColumnData::String(v) if v == &vec!["GENEA","GENEB","GENEC"])));

    // The matrix is correctly remapped (token→column) and the marker is gone.
    let expected = [[5.0f64, 0.0, 2.0], [0.0, 4.0, 0.0]];
    let dense = futures::executor::block_on(async {
        let mut dense = vec![vec![0.0f64; 3]; 2];
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("chunk");
            let values = chunk.data.data.to_f64();
            for local in 0..chunk.nrows {
                let g = chunk.row_offset + local;
                let start = chunk.data.indptr[local] as usize;
                let end = chunk.data.indptr[local + 1] as usize;
                for k in start..end {
                    dense[g][chunk.data.indices[k] as usize] = values[k];
                }
            }
        }
        dense
    });
    for (r, want) in expected.iter().enumerate() {
        assert_eq!(dense[r], want.to_vec(), "remapped row {r} mismatch");
    }
}
