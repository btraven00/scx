#![cfg(feature = "bixverse")]

use std::io::Write;
use std::path::Path;

use bixverse_sc_data::data_io::{CellGeneSparseWriter, CsrCellChunk};
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::{executor::block_on, StreamExt};
use scx_core::bixverse::{BixverseLayer, BixverseReader};
use scx_core::detect::{detect, Format};
use scx_core::dtype::{DataType, TypedVec};
use scx_core::ir::ColumnData;
use scx_core::stream::DatasetReader;
use scx_core::OpenOptions;
use tempfile::TempDir;

const N_VARS: usize = 5;

/// One value above `u16::MAX`, so a `U32` raw chunk is exercised too.
const DENSE: [[u32; N_VARS]; 4] = [
    [1, 0, 3, 0, 70_000],
    [0, 2, 0, 0, 0],
    [4, 5, 6, 7, 8],
    [0, 0, 0, 9, 0],
];

fn write_store(path: &Path, v3: bool) {
    let n = DENSE.len();
    let mut w = if v3 {
        CellGeneSparseWriter::new_v3(path, true, n, N_VARS, 1e4)
    } else {
        CellGeneSparseWriter::new(path, true, n, N_VARS, 1e4)
    }
    .unwrap();
    for (i, row) in DENSE.iter().enumerate() {
        let (idx, data): (Vec<u32>, Vec<u32>) = row
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0)
            .map(|(j, &v)| (j as u32, v))
            .unzip();
        w.write_cell_chunk(CsrCellChunk::from_data(&data, &idx, i, 1e4, true))
            .unwrap();
    }
    w.finalise().unwrap();
}

fn expected_csr() -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let (mut indptr, mut indices, mut data) = (vec![0u64], Vec::new(), Vec::new());
    for row in &DENSE {
        for (j, &v) in row.iter().enumerate().filter(|(_, &v)| v > 0) {
            indices.push(j as u32);
            data.push(v as f32);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, data)
}

/// Stream X and stitch the chunks back into one CSR.
fn read_x(reader: &mut dyn DatasetReader) -> (Vec<u64>, Vec<u32>, Vec<f32>, usize) {
    let (mut indptr, mut indices, mut data) = (vec![0u64], Vec::new(), Vec::new());
    let mut n_chunks = 0;
    block_on(async {
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            assert_eq!(chunk.row_offset, indptr.len() - 1);
            let base = *indptr.last().unwrap();
            indptr.extend(chunk.data.indptr[1..].iter().map(|p| p + base));
            indices.extend(chunk.data.indices);
            match chunk.data.data {
                TypedVec::F32(v) => data.extend(v),
                other => panic!("expected F32, got {other:?}"),
            }
            n_chunks += 1;
        }
    });
    (indptr, indices, data, n_chunks)
}

fn write_gz(path: &Path, text: &str) {
    let mut enc = GzEncoder::new(std::fs::File::create(path).unwrap(), Compression::default());
    enc.write_all(text.as_bytes()).unwrap();
    enc.finish().unwrap();
}

#[test]
fn streams_v3_and_v4_stores_as_the_source_matrix() {
    let dir = TempDir::new().unwrap();
    for v3 in [true, false] {
        let bin = dir
            .path()
            .join(format!("cells_v{}.bin", if v3 { 3 } else { 4 }));
        write_store(&bin, v3);

        let mut reader = BixverseReader::open(&bin, 3, BixverseLayer::Raw).unwrap();
        assert_eq!(reader.shape(), (DENSE.len(), N_VARS));
        let (indptr, indices, data, n_chunks) = read_x(&mut reader);
        let (e_indptr, e_indices, e_data) = expected_csr();
        assert_eq!(n_chunks, 2, "4 cells at chunk size 3");
        assert_eq!(indptr, e_indptr, "v3: {v3}");
        assert_eq!(indices, e_indices, "v3: {v3}");
        assert_eq!(data, e_data, "v3: {v3}");
    }
}

#[test]
fn factory_opens_the_raw_and_the_norm_layer() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);
    let path = bin.to_str().unwrap();
    let (e_indptr, e_indices, e_data) = expected_csr();

    let mut raw = block_on(scx_core::open(path, &OpenOptions::new(10))).unwrap();
    assert_eq!(raw.dtype(), DataType::F32);
    let (indptr, indices, data, _) = read_x(raw.as_mut());
    assert_eq!(
        (indptr, indices, data),
        (e_indptr.clone(), e_indices.clone(), e_data)
    );

    let mut opts = OpenOptions::new(10);
    opts.layer = Some("norm".into());
    let mut norm = block_on(scx_core::open(path, &opts)).unwrap();
    let (indptr, indices, data, _) = read_x(norm.as_mut());
    assert_eq!((indptr, indices), (e_indptr, e_indices));
    // ln1p(x / library size * 1e4), stored as f16.
    let expected = DENSE.iter().flat_map(|row| {
        let sum: u32 = row.iter().sum();
        row.iter()
            .filter(|&&v| v > 0)
            .map(move |&v| (v as f32 / sum as f32 * 1e4).ln_1p())
    });
    for (got, want) in data.iter().zip(expected) {
        assert!((got - want).abs() < 1e-2, "{got} vs {want}");
    }
}

#[test]
fn detects_a_store_by_file_and_by_directory() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);
    assert_eq!(detect(&bin), Some(Format::Bixverse));
    assert_eq!(detect(dir.path()), Some(Format::Bixverse));

    let reader = BixverseReader::open(dir.path(), 10, BixverseLayer::Raw).unwrap();
    assert_eq!(reader.shape(), (DENSE.len(), N_VARS));
}

#[test]
fn missing_sidecar_gives_a_synthetic_index() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);

    let mut reader = BixverseReader::open(&bin, 10, BixverseLayer::Raw).unwrap();
    let obs = block_on(reader.obs()).unwrap();
    let var = block_on(reader.var()).unwrap();
    assert_eq!(obs.index, ["cell_0", "cell_1", "cell_2", "cell_3"]);
    assert_eq!(var.index.len(), N_VARS);
    assert!(obs.columns.is_empty() && var.columns.is_empty());
}

#[test]
fn tsv_gz_sidecar_is_typed() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);
    write_gz(
        &dir.path().join("obs.tsv.gz"),
        "cell_id\tn_umi\tscore\tkeep\tcluster\tnote\n\
         a\t1\t0.5\tTRUE\tx\tNA\n\
         b\t2\tNA\tFALSE\tx\tp\n\
         c\t3\t1.5\tTRUE\ty\tq\n\
         d\t4\t2.5\tTRUE\ty\tr\n",
    );

    let mut reader = BixverseReader::open(&bin, 10, BixverseLayer::Raw).unwrap();
    let obs = block_on(reader.obs()).unwrap();
    assert_eq!(obs.index, ["a", "b", "c", "d"]);

    let col = |name: &str| obs.columns.iter().find(|c| c.name == name).unwrap();
    assert!(matches!(&col("n_umi").data, ColumnData::Int(v) if v == &[1, 2, 3, 4]));
    assert!(matches!(&col("score").data, ColumnData::Float(_)));
    assert!(col("score").is_na(1) && !col("score").is_na(0));
    assert!(matches!(&col("keep").data, ColumnData::Bool(v) if v == &[true, false, true, true]));
    assert!(matches!(
        &col("cluster").data,
        ColumnData::Categorical { codes, levels, .. } if codes == &[0, 0, 1, 1] && levels == &["x", "y"]
    ));
    assert!(matches!(&col("note").data, ColumnData::String(_)));
    assert!(col("note").is_na(0));
}

#[test]
fn plain_tsv_sidecar_keeps_int64_and_rejects_ragged_rows() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);
    std::fs::write(
        dir.path().join("var.tsv"),
        "gene_id\tid\ng0\t3000000000\ng1\t1\ng2\t2\ng3\t3\ng4\t4\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("obs.tsv"), "cell_id\tx\na\t1\nb\n").unwrap();

    let mut reader = BixverseReader::open(&bin, 10, BixverseLayer::Raw).unwrap();
    let var = block_on(reader.var()).unwrap();
    assert!(matches!(&var.columns[0].data, ColumnData::Int64(v) if v[0] == 3_000_000_000));
    let err = block_on(reader.obs()).unwrap_err().to_string();
    assert!(err.contains("row 2 has 1 fields"), "{err}");
}

#[test]
fn sidecar_row_count_must_match_the_store() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);
    write_gz(&dir.path().join("obs.tsv.gz"), "cell_id\tx\na\t1\n");

    let mut reader = BixverseReader::open(&bin, 10, BixverseLayer::Raw).unwrap();
    assert!(block_on(reader.obs()).is_err());
}

#[test]
fn rejects_a_gene_based_store() {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_genes.bin");
    CellGeneSparseWriter::new(&bin, false, DENSE.len(), N_VARS, 1e4)
        .unwrap()
        .finalise()
        .unwrap();
    assert!(BixverseReader::open(&bin, 10, BixverseLayer::Raw).is_err());
    assert!(BixverseReader::open(dir.path().join("missing.bin"), 10, BixverseLayer::Raw).is_err());
}

#[cfg(feature = "bixverse-parquet")]
#[test]
fn parquet_sidecar_matches_the_tsv_sidecar() {
    use arrow::array::{
        ArrayRef, BooleanArray, DictionaryArray, Float64Array, Int32Array, Int64Array, StringArray,
        UInt64Array,
    };
    use arrow::datatypes::UInt32Type;
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    write_store(&bin, false);

    let cols: Vec<(&str, ArrayRef)> = vec![
        (
            "cell_id",
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
        ),
        ("n_umi", Arc::new(Int32Array::from(vec![1, 2, 3, 4]))),
        (
            "score",
            Arc::new(Float64Array::from(vec![
                Some(0.5),
                None,
                Some(1.5),
                Some(2.5),
            ])),
        ),
        (
            "cluster",
            Arc::new(DictionaryArray::<UInt32Type>::from_iter([
                "x", "x", "y", "y",
            ])),
        ),
        (
            "keep",
            Arc::new(BooleanArray::from(vec![true, false, true, true])),
        ),
        (
            "note",
            Arc::new(StringArray::from(vec![
                None,
                Some("p"),
                Some("q"),
                Some("r"),
            ])),
        ),
        (
            "big",
            Arc::new(Int64Array::from(vec![3_000_000_000, 1, 2, 3])),
        ),
        ("huge", Arc::new(UInt64Array::from(vec![u64::MAX, 1, 2, 3]))),
    ];
    let batch = RecordBatch::try_from_iter(cols).unwrap();
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(dir.path().join("obs.parquet")).unwrap(),
        batch.schema(),
        None,
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let mut reader = BixverseReader::open(&bin, 10, BixverseLayer::Raw).unwrap();
    let obs = block_on(reader.obs()).unwrap();
    assert_eq!(obs.index, ["a", "b", "c", "d"]);
    let col = |name: &str| obs.columns.iter().find(|c| c.name == name).unwrap();
    assert!(matches!(&col("n_umi").data, ColumnData::Int(v) if v == &[1, 2, 3, 4]));
    assert!(col("score").is_na(1) && !col("score").is_na(0));
    assert!(matches!(
        &col("cluster").data,
        ColumnData::Categorical { codes, levels, .. } if codes == &[0, 0, 1, 1] && levels == &["x", "y"]
    ));
    assert!(matches!(&col("keep").data, ColumnData::Bool(v) if v == &[true, false, true, true]));
    assert!(matches!(&col("note").data, ColumnData::String(_)));
    assert!(col("note").is_na(0));
    assert!(matches!(&col("big").data, ColumnData::Int64(v) if v[0] == 3_000_000_000));
    // Beyond i64: widened to float rather than read as missing.
    assert!(matches!(&col("huge").data, ColumnData::Float(v) if v[0] == u64::MAX as f64));
    assert!(!col("huge").is_na(0));
}
