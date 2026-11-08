//! The Zarr reader's contract is "whatever H5AdReader returns for the same
//! AnnData". The fixtures (scripts/prepare_zarr_fixtures.py) are one AnnData
//! written three ways by anndata itself: h5ad, Zarr v2 and Zarr v3. Every
//! slot is compared through its Debug form, which is exact and, unlike `==`,
//! treats NaN (nullable columns) consistently.

use std::path::{Path, PathBuf};

use futures::{stream, Stream, StreamExt};

use super::{ZarrAdReader, ZarrAdWriter};
use crate::detect::{detect, Format};
use crate::error::Result;
use crate::h5ad::H5AdReader;
use crate::ir::{ColumnData, MatrixChunk};
use crate::stream::DatasetReader;
use crate::stream::DatasetWriter;

const DIR: &str = "../../tests/fixtures/zarr";
/// Deliberately not a divisor of 60 cells, so row chunks straddle Zarr chunks.
const CHUNK: usize = 7;

fn fixture(name: &str) -> PathBuf {
    Path::new(DIR).join(name)
}

async fn chunks<S: Stream<Item = Result<MatrixChunk>>>(s: S) -> Vec<String> {
    s.map(|c| format!("{:?}", c.unwrap())).collect().await
}

fn sorted_debug<V: std::fmt::Debug>(m: &std::collections::HashMap<String, V>) -> Vec<String> {
    let mut v: Vec<String> = m.iter().map(|(k, x)| format!("{k}: {x:?}")).collect();
    v.sort();
    v
}

async fn assert_same_as_h5ad(zarr: &str, h5ad: &str) {
    assert_store_same_as_h5ad(&fixture(zarr), &fixture(h5ad), zarr, true).await;
}

/// Same values in every slot; with `same_layout`, also the same storage
/// (indptr, layer and obsp metas): a dense h5ad matrix comes back from the
/// writer as CSR, so a written store only matches it in values.
async fn assert_store_same_as_h5ad(
    zarr_path: &Path,
    h5ad_path: &Path,
    zarr: &str,
    same_layout: bool,
) {
    let mut z = ZarrAdReader::open(zarr_path, CHUNK).unwrap();
    let mut h = H5AdReader::open(h5ad_path, CHUNK).unwrap();

    assert_eq!(z.shape(), h.shape(), "{zarr}: shape");
    assert_eq!(z.dtype(), h.dtype(), "{zarr}: dtype");
    if same_layout {
        assert_eq!(z.x_indptr(), h.x_indptr(), "{zarr}: X indptr");
    }
    assert_eq!(
        chunks(z.x_stream()).await,
        chunks(h.x_stream()).await,
        "{zarr}: X"
    );

    assert_eq!(
        format!("{:?}", z.obs().await.unwrap()),
        format!("{:?}", h.obs().await.unwrap()),
        "{zarr}: obs"
    );
    assert_eq!(
        format!("{:?}", z.var().await.unwrap()),
        format!("{:?}", h.var().await.unwrap()),
        "{zarr}: var"
    );
    assert_eq!(
        sorted_debug(&z.obsm().await.unwrap().map),
        sorted_debug(&h.obsm().await.unwrap().map),
        "{zarr}: obsm"
    );
    assert_eq!(
        sorted_debug(&z.varm().await.unwrap().map),
        sorted_debug(&h.varm().await.unwrap().map),
        "{zarr}: varm"
    );
    let (zu, hu) = (z.uns().await.unwrap(), h.uns().await.unwrap());
    assert_eq!(zu.raw, hu.raw, "{zarr}: uns");
    assert_eq!(zu.arrays, hu.arrays, "{zarr}: uns arrays");

    let (zl, hl) = (
        z.layer_metas().await.unwrap(),
        h.layer_metas().await.unwrap(),
    );
    let names =
        |m: &[crate::ir::SparseMatrixMeta]| m.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&zl), names(&hl), "{zarr}: layers");
    if same_layout {
        assert_eq!(format!("{zl:?}"), format!("{hl:?}"), "{zarr}: layer metas");
    }
    for (zm, hm) in zl.iter().zip(&hl) {
        assert_eq!(
            chunks(z.layer_stream(zm, CHUNK)).await,
            chunks(h.layer_stream(hm, CHUNK)).await,
            "{zarr}: layers['{}']",
            zm.name
        );
    }
    let (zp, hp) = (z.obsp_metas().await.unwrap(), h.obsp_metas().await.unwrap());
    assert_eq!(names(&zp), names(&hp), "{zarr}: obsp");
    if same_layout {
        assert_eq!(format!("{zp:?}"), format!("{hp:?}"), "{zarr}: obsp metas");
    }
    for (zm, hm) in zp.iter().zip(&hp) {
        assert_eq!(
            chunks(z.obsp_stream(zm, CHUNK)).await,
            chunks(h.obsp_stream(hm, CHUNK)).await,
            "{zarr}: obsp['{}']",
            zm.name
        );
    }
}

#[tokio::test]
async fn zarr_v2_reads_exactly_what_h5ad_reads() {
    assert_same_as_h5ad("small_v2.zarr", "small.h5ad").await;
}

#[tokio::test]
async fn zarr_v3_sharded_reads_exactly_what_h5ad_reads() {
    assert_same_as_h5ad("small_v3.zarr", "small.h5ad").await;
}

#[test]
fn detected_as_anndata_zarr() {
    assert_eq!(detect(&fixture("small_v2.zarr")), Some(Format::ZarrAd));
    assert_eq!(detect(&fixture("small_v3.zarr")), Some(Format::ZarrAd));
}

#[tokio::test]
async fn factory_opens_zarr_stores() {
    for name in ["small_v2.zarr", "small_v3.zarr"] {
        let path = fixture(name);
        let r = crate::open(path.to_str().unwrap(), &crate::OpenOptions::new(CHUNK))
            .await
            .unwrap();
        assert_eq!(r.shape(), (60, 40), "{name}");
    }
}

#[tokio::test]
async fn explicit_layer_is_read_like_h5ad() {
    for name in ["small_v2.zarr", "small_v3.zarr"] {
        let mut z = ZarrAdReader::open_layer(fixture(name), CHUNK, Some("counts")).unwrap();
        let mut h = H5AdReader::open_layer(fixture("small.h5ad"), CHUNK, Some("counts")).unwrap();
        assert_eq!(z.x_source(), "layers/counts");
        assert_eq!(
            chunks(z.x_stream()).await,
            chunks(h.x_stream()).await,
            "{name}: counts as X"
        );
        assert!(ZarrAdReader::open_layer(fixture(name), CHUNK, Some("nope")).is_err());
    }
}

/// The layer serving as X is not listed again as a layer (as h5ad after #34):
/// otherwise `scx convert` of a store written with `adata.X = None` writes the
/// matrix twice.
#[tokio::test]
async fn layer_serving_as_x_is_not_also_a_layer() {
    for name in ["small_v2.zarr", "small_v3.zarr"] {
        let mut z = ZarrAdReader::open_layer(fixture(name), CHUNK, Some("counts")).unwrap();
        let names: Vec<String> = z
            .layer_metas()
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert!(!names.contains(&"counts".to_string()), "{name}: {names:?}");
    }
}

/// Unsorted column indices within a row (valid CSR; every row of the 10x
/// ladder files is like this) must come out sorted, as H5AdReader does since
/// the sort fix: otherwise picklerick builds an invalid dgCMatrix. The oracle
/// comparison covers values; this also checks the order directly.
#[tokio::test]
async fn unsorted_indices_are_sorted_like_h5ad() {
    assert_same_as_h5ad("small_unsorted_v3.zarr", "small_unsorted.h5ad").await;

    let mut z = ZarrAdReader::open(fixture("small_unsorted_v3.zarr"), CHUNK).unwrap();
    let chunks: Vec<MatrixChunk> = z.x_stream().map(|c| c.unwrap()).collect().await;
    for c in &chunks {
        for r in 0..c.nrows {
            let (a, b) = (c.data.indptr[r] as usize, c.data.indptr[r + 1] as usize);
            assert!(
                c.data.indices[a..b].windows(2).all(|p| p[0] < p[1]),
                "row {} not sorted",
                c.row_offset + r
            );
        }
    }
}

#[test]
fn store_errors_keep_their_source() {
    use std::error::Error;

    let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no chunk");
    let err = super::zerr("X/data", io);
    assert_eq!(err.to_string(), "zarr error: X/data: no chunk");
    let source = err.source().expect("source dropped");
    assert!(source.downcast_ref::<std::io::Error>().is_some());
}

/// Write everything the h5ad reader returns for `src` as a Zarr store, X as
/// `dtype`.
async fn write_from_h5ad(src: &Path, out: &Path, dtype: crate::dtype::DataType) {
    let mut h = H5AdReader::open(src, CHUNK).unwrap();
    let (n_obs, n_vars) = h.shape();
    let mut w = ZarrAdWriter::create(out, n_obs, n_vars, dtype).unwrap();
    w.write_obs(&h.obs().await.unwrap()).await.unwrap();
    w.write_var(&h.var().await.unwrap()).await.unwrap();
    w.write_obsm(&h.obsm().await.unwrap()).await.unwrap();
    w.write_varm(&h.varm().await.unwrap()).await.unwrap();
    w.write_uns(&h.uns().await.unwrap()).await.unwrap();
    for m in h.layer_metas().await.unwrap() {
        w.begin_sparse("layers", &m.name, &m).await.unwrap();
        let mut s = h.layer_stream(&m, CHUNK);
        while let Some(c) = s.next().await {
            w.write_sparse_chunk(&c.unwrap()).await.unwrap();
        }
        w.end_sparse().await.unwrap();
    }
    for m in h.obsp_metas().await.unwrap() {
        w.begin_sparse("obsp", &m.name, &m).await.unwrap();
        let mut s = h.obsp_stream(&m, CHUNK);
        while let Some(c) = s.next().await {
            w.write_sparse_chunk(&c.unwrap()).await.unwrap();
        }
        w.end_sparse().await.unwrap();
    }
    let mut s = h.x_stream();
    while let Some(c) = s.next().await {
        w.write_x_chunk(&c.unwrap()).await.unwrap();
    }
    w.finalize().await.unwrap();
}

/// Everything the h5ad reader returns, written as a Zarr v3 store, reads back
/// identically through the Zarr reader (the writer's contract is the reader's
/// in reverse) and as the h5ad it came from. `varied` covers every column
/// encoding and uns value type the writer has; its dense matrices come back
/// as CSR, so only values are compared.
#[tokio::test]
async fn writer_roundtrips_what_h5ad_reads() {
    let dir = tempfile::tempdir().unwrap();
    for (name, same_layout) in [("small", true), ("varied", false)] {
        let out = dir.path().join(format!("{name}.zarr"));
        let src = fixture(&format!("{name}.h5ad"));
        let dtype = H5AdReader::open(&src, CHUNK).unwrap().dtype();
        write_from_h5ad(&src, &out, dtype).await;
        assert_eq!(detect(&out), Some(Format::ZarrAd));
        assert_store_same_as_h5ad(&out, &src, name, same_layout).await;
    }
    // Typed uns arrays are written in their own type, in row chunks.
    use zarrs::{array::data_type as zdt, filesystem::FilesystemStore};
    let img = zarrs::array::Array::open(
        std::sync::Arc::new(FilesystemStore::new(dir.path().join("varied.zarr")).unwrap()),
        "/uns/spatial/lib/images/hires",
    )
    .unwrap();
    assert_eq!(*img.data_type(), zdt::uint8());
    assert_eq!(img.shape(), [160, 160, 3]);
}

/// X is stored in the dtype asked for, whatever the source's, with the same
/// values (all small integers here, exact in every dtype).
#[tokio::test]
async fn writer_stores_x_in_the_dtype_asked_for() {
    use crate::dtype::DataType;
    let dir = tempfile::tempdir().unwrap();
    let src = fixture("varied.h5ad");
    let values = |mut r: Box<dyn DatasetReader>| async move {
        let c: Vec<MatrixChunk> = r.x_stream().map(|c| c.unwrap()).collect().await;
        c.iter()
            .map(|c| {
                (
                    c.data.indptr.clone(),
                    c.data.indices.clone(),
                    c.data.data.to_f64(),
                )
            })
            .collect::<Vec<_>>()
    };
    let want = values(Box::new(H5AdReader::open(&src, CHUNK).unwrap())).await;
    for dtype in [DataType::F32, DataType::F64, DataType::I32, DataType::U32] {
        let out = dir.path().join(format!("{dtype}.zarr"));
        write_from_h5ad(&src, &out, dtype).await;
        let z = ZarrAdReader::open(&out, CHUNK).unwrap();
        assert_eq!(z.dtype(), dtype);
        assert_eq!(values(Box::new(z)).await, want, "{dtype}");
    }
}

/// Categoricals with more levels than int8 (int16) or int16 (int32) codes
/// hold, a float column with a missing value (stored as NaN, so it reads back
/// as NaN without a mask), nullable int64, and a column named `_index`, which
/// is reserved and dropped.
#[tokio::test]
async fn writer_roundtrips_wide_categoricals_and_masked_columns() {
    use crate::ir::{Column, ColumnData, ObsTable};
    let n = 3;
    let cat = |levels: usize| ColumnData::Categorical {
        codes: vec![0, (levels - 1) as u32, 1],
        levels: (0..levels).map(|i| format!("l{i}")).collect(),
        ordered: false,
    };
    let obs = ObsTable {
        index: (0..n).map(|i| format!("c{i}")).collect(),
        columns: vec![
            Column::new("wide16", cat(200)),
            Column::new("wide32", cat(40_000)),
            Column::with_missing(
                "f",
                ColumnData::Float(vec![1.0, 2.0, 3.0]),
                vec![false, true, false],
            ),
            Column::with_missing(
                "i64",
                ColumnData::Int64(vec![5, 0, 7]),
                vec![false, true, false],
            ),
            Column::new("_index", ColumnData::Int(vec![9, 9, 9])),
        ],
    };
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("cats.zarr");
    let mut w = ZarrAdWriter::create(&out, n, 0, crate::dtype::DataType::F32).unwrap();
    w.write_obs(&obs).await.unwrap();
    w.write_x_chunk(&MatrixChunk {
        row_offset: 0,
        nrows: n,
        data: crate::ir::SparseMatrixCSR {
            shape: (n, 0),
            indptr: vec![0; n + 1],
            indices: vec![],
            data: crate::dtype::TypedVec::F32(vec![]),
        },
    })
    .await
    .unwrap();
    w.finalize().await.unwrap();

    let got = ZarrAdReader::open(&out, CHUNK)
        .unwrap()
        .obs()
        .await
        .unwrap();
    let mut want = obs.clone();
    want.columns.pop(); // `_index`
    want.columns[2] = Column::new("f", ColumnData::Float(vec![1.0, f64::NAN, 3.0]));
    assert_eq!(format!("{:?}", got.columns), format!("{:?}", want.columns));
}

/// Calls out of order are errors, not a corrupt store.
#[tokio::test]
async fn writer_rejects_misuse() {
    use crate::dtype::DataType;
    use crate::ir::{DenseMatrix, Embeddings};
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    assert!(ZarrAdWriter::create_at(p("a.zarr"), 1, 1, DataType::F32, "layers/").is_err());
    assert!(ZarrAdWriter::create_at(p("b.zarr"), 1, 1, DataType::F32, "obsm/x").is_err());

    let mut w = ZarrAdWriter::create_at(p("c.zarr"), 1, 1, DataType::F32, "layers/counts").unwrap();
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: 1,
        data: crate::ir::SparseMatrixCSR {
            shape: (1, 1),
            indptr: vec![0, 0],
            indices: vec![],
            data: crate::dtype::TypedVec::F32(vec![]),
        },
    };
    assert!(
        w.write_sparse_chunk(&chunk).await.is_err(),
        "chunk before begin_sparse"
    );
    assert!(w.end_sparse().await.is_err(), "end before begin_sparse");
    let mut bad = Embeddings::default();
    bad.map.insert(
        "X_bad".into(),
        DenseMatrix {
            shape: (1, 2),
            data: vec![0.0],
        },
    );
    assert!(
        w.write_obsm(&bad).await.is_err(),
        "1 value for a 1 x 2 matrix"
    );
    w.write_x_chunk(&chunk).await.unwrap();
    w.finalize().await.unwrap();
    assert!(w.write_x_chunk(&chunk).await.is_err(), "X after finalize");
    assert!(w.finalize().await.is_err(), "finalize twice");
    let z = ZarrAdReader::open(p("c.zarr"), CHUNK).unwrap();
    assert_eq!(z.x_source(), "layers/counts");
}

/// Writing fewer rows than declared is an error, not a silently short store.
#[tokio::test]
async fn writer_rejects_missing_rows() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ZarrAdWriter::create(
        dir.path().join("short.zarr"),
        5,
        3,
        crate::dtype::DataType::F32,
    )
    .unwrap();
    assert!(w.finalize().await.is_err());
}

/// Rows as (index in the full matrix, column indices, values): chunk
/// boundaries differ between the lazy path (chunks start at each range) and
/// the trait default (chunks split where the source chunks did).
async fn rows<S: Stream<Item = Result<MatrixChunk>>>(s: S) -> Vec<(usize, Vec<u32>, Vec<f64>)> {
    s.map(|c| c.unwrap())
        .flat_map(|c| {
            stream::iter((0..c.nrows).map(move |r| {
                let (a, b) = (c.data.indptr[r] as usize, c.data.indptr[r + 1] as usize);
                (
                    c.row_offset + r,
                    c.data.indices[a..b].to_vec(),
                    c.data.data.slice(a..b).to_f64(),
                )
            }))
        })
        .collect()
        .await
}

/// Row ranges read only those rows, with their offsets in the full matrix,
/// and agree with the trait default (slice the full stream) on every reader.
#[tokio::test]
async fn x_stream_rows_matches_sliced_full_stream() {
    let ranges = vec![50..60, 3..9, 7..12, 20..20];
    let want = [3..12, 50..60]
        .iter()
        .flat_map(|r| r.clone())
        .collect::<Vec<_>>();
    let expected = {
        let mut h = H5AdReader::open(fixture("small.h5ad"), CHUNK).unwrap();
        let full: Vec<MatrixChunk> = h.x_stream().map(|c| c.unwrap()).collect().await;
        let norm = crate::stream::normalize_ranges(ranges.clone());
        let sliced = full
            .iter()
            .flat_map(|c| crate::stream::select_rows(c, &norm));
        rows(stream::iter(sliced.map(Ok))).await
    };
    assert_eq!(expected.iter().map(|r| r.0).collect::<Vec<_>>(), want);
    for name in ["small_v2.zarr", "small_v3.zarr"] {
        let mut z = ZarrAdReader::open(fixture(name), CHUNK).unwrap();
        assert_eq!(
            rows(z.x_stream_rows(ranges.clone())).await,
            expected,
            "{name}"
        );
    }
    let mut h = H5AdReader::open(fixture("small.h5ad"), CHUNK).unwrap();
    assert_eq!(
        rows(h.x_stream_rows(ranges.clone())).await,
        expected,
        "h5ad"
    );
    // Lazy chunks start at each range and are at most CHUNK rows.
    let offsets: Vec<usize> = h
        .x_stream_rows(ranges.clone())
        .map(|c| c.unwrap().row_offset)
        .collect()
        .await;
    assert_eq!(offsets, vec![3, 10, 50, 57]);

    let mut bad = h.x_stream_rows(vec![55..61]);
    assert!(bad.next().await.unwrap().is_err(), "range past n_obs");
}

/// A row-range stream reads only its rows' part of `X/indptr` (see the h5ad
/// reader's test of the same name). The indptr is stored in chunks of 2 and
/// its last chunk, holding only the final entry, is garbage: rows 0..3 (the
/// first two chunks) still stream, while the whole-matrix stream fails.
#[tokio::test]
async fn row_range_stream_reads_only_its_indptr() {
    use std::sync::Arc;
    use zarrs::array::{data_type as zdt, ArrayBuilder, FillValue};
    use zarrs::filesystem::FilesystemStore;
    use zarrs::group::GroupBuilder;

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("bad_tail.zarr");
    std::fs::create_dir(&root).unwrap();
    let store = Arc::new(FilesystemStore::new(&root).unwrap());
    let attrs = serde_json::json!({"encoding-type": "csr_matrix", "shape": [4, 3]});
    GroupBuilder::new()
        .attributes(attrs.as_object().unwrap().clone())
        .build(store.clone(), "/X")
        .unwrap()
        .store_metadata()
        .unwrap();
    let ip = ArrayBuilder::new(vec![5], vec![2], zdt::int64(), FillValue::from(0i64))
        .build(store.clone(), "/X/indptr")
        .unwrap();
    ip.store_metadata().unwrap();
    ip.store_array_subset(&vec![0..5u64], &[0i64, 1, 3, 3, 4][..])
        .unwrap();
    let idx = ArrayBuilder::new(vec![4], vec![4], zdt::int32(), FillValue::from(0i32))
        .build(store.clone(), "/X/indices")
        .unwrap();
    idx.store_metadata().unwrap();
    idx.store_array_subset(&vec![0..4u64], &[2i32, 0, 1, 0][..])
        .unwrap();
    let data = ArrayBuilder::new(vec![4], vec![4], zdt::float32(), FillValue::from(0f32))
        .build(store.clone(), "/X/data")
        .unwrap();
    data.store_metadata().unwrap();
    data.store_array_subset(&vec![0..4u64], &[1f32, 2.0, 3.0, 4.0][..])
        .unwrap();
    let last = root.join("X/indptr/c/2");
    assert!(last.exists(), "chunk layout changed");
    std::fs::write(&last, b"bad").unwrap();

    let mut r = ZarrAdReader::open(&root, 10).unwrap();
    let mut s = r.x_stream_rows(vec![1..2, 0..1, 2..3]);
    let c = s.next().await.unwrap().unwrap();
    assert!(s.next().await.is_none());
    drop(s);
    assert_eq!((c.row_offset, c.nrows), (0, 3));
    assert_eq!(c.data.indptr, vec![0, 1, 3, 3]);
    assert_eq!(c.data.indices, vec![2, 0, 1]);
    assert_eq!(c.data.data.to_f64(), vec![1.0, 2.0, 3.0]);

    assert!(r.x_stream().next().await.unwrap().is_err());
    assert!(
        r.x_indptr().is_empty(),
        "an unreadable indptr is not a row pointer"
    );
}

/// `varied` holds what `small` lacks: dense int X, a CSC layer and obsp, a
/// dense layer, int16..uint64 and float32 columns, float/int/bool and ordered
/// categories, int obsm and every uns value type. Same contract as `small`.
#[tokio::test]
async fn varied_encodings_read_exactly_what_h5ad_reads() {
    assert_same_as_h5ad("varied_v2.zarr", "varied.h5ad").await;
    assert_same_as_h5ad("varied_v3.zarr", "varied.h5ad").await;
    // The Visium-like image is carried typed, not as JSON (see h5_json).
    let mut z = ZarrAdReader::open(fixture("varied_v3.zarr"), CHUNK).unwrap();
    let uns = z.uns().await.unwrap();
    let img = &uns.arrays["spatial/lib/images/hires"];
    assert_eq!(img.shape, [160, 160, 3]);
    assert!(matches!(img.data, crate::ir::UnsData::U8(_)));
}

/// Without X the matrix comes from layers by h5ad's rules: the only layer, or
/// counts/X among several; several without either is an error naming them.
#[tokio::test]
async fn without_x_layers_resolve_like_h5ad() {
    assert_same_as_h5ad("nox_one_v3.zarr", "nox_one.h5ad").await;
    let z = ZarrAdReader::open(fixture("nox_one_v3.zarr"), CHUNK).unwrap();
    assert_eq!(z.x_source(), "layers/only");
    let e = ZarrAdReader::open(fixture("nox_many_v3.zarr"), CHUNK)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("multiple layers present (only, other)"), "{e}");
    assert!(H5AdReader::open(fixture("nox_many.h5ad"), CHUNK).is_err());
}

/// X stored as CSC streams by gene exactly as h5ad's does, and by cell it
/// refuses rather than being misread as CSR.
#[tokio::test]
async fn csc_x_streams_by_gene_like_h5ad() {
    let mut z = ZarrAdReader::open(fixture("csc_x_v3.zarr"), CHUNK).unwrap();
    let mut h = H5AdReader::open(fixture("csc_x.h5ad"), CHUNK).unwrap();
    assert_eq!(z.shape(), h.shape());
    assert_eq!(z.x_col_ptr(), h.x_col_ptr());
    assert!(!z.x_col_ptr().is_empty());
    let all = vec![0..z.shape().1];
    assert_eq!(
        chunks(z.x_stream_cols(all.clone())).await,
        chunks(h.x_stream_cols(all)).await
    );
    let some = vec![2..5, 9..11];
    assert_eq!(
        chunks(z.x_stream_cols(some.clone())).await,
        chunks(h.x_stream_cols(some)).await
    );
    // By cell it refuses, as h5ad does.
    let e = format!("{:?}", z.x_stream().next().await.unwrap().unwrap_err());
    assert!(e.contains("gene-major"), "{e}");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dst = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

/// Malformed stores fail with an error naming what is wrong; a dataframe
/// column that is listed but missing, or in an encoding the reader doesn't
/// know, is skipped rather than failing the whole frame.
#[tokio::test]
async fn malformed_stores_are_reported() {
    use serde_json::{json, Map, Value};
    let dir = tempfile::tempdir().unwrap();
    // small_v3.zarr with the attributes of `node` edited.
    let broken = |name: &str, node: &str, edit: &dyn Fn(&mut Map<String, Value>)| {
        let p = dir.path().join(name);
        copy_dir(&fixture("small_v3.zarr"), &p);
        let f = p.join(node).join("zarr.json");
        let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        edit(v["attributes"].as_object_mut().unwrap());
        std::fs::write(&f, v.to_string()).unwrap();
        p
    };
    let err = |p: PathBuf| ZarrAdReader::open(p, CHUNK).err().unwrap().to_string();

    let e = err(broken("root", "", &|a| {
        a.insert("encoding-type".into(), json!("mudata"));
    }));
    assert!(e.contains("not an AnnData store"), "{e}");
    let e = err(broken("shape", "X", &|a| {
        a.insert("shape".into(), json!([60]));
    }));
    assert!(e.contains("bad X shape"), "{e}");
    let e = err(broken("rows", "X", &|a| {
        a.insert("shape".into(), json!([61, 40]));
    }));
    assert!(e.contains("X/indptr length 61 != n_obs+1 62"), "{e}");

    let p = broken("cols", "obs", &|a| {
        let order = a["column-order"].as_array_mut().unwrap();
        order.push(json!("ghost"));
        order.push(json!("weird"));
    });
    let weird = p.join("obs/weird");
    std::fs::create_dir_all(&weird).unwrap();
    std::fs::write(
        weird.join("zarr.json"),
        json!({"zarr_format": 3, "node_type": "group", "attributes": {}}).to_string(),
    )
    .unwrap();
    let mut z = ZarrAdReader::open(&p, CHUNK).unwrap();
    let mut h = H5AdReader::open(fixture("small.h5ad"), CHUNK).unwrap();
    let names = |t: crate::ir::ObsTable| t.columns.into_iter().map(|c| c.name).collect::<Vec<_>>();
    assert_eq!(names(z.obs().await.unwrap()), names(h.obs().await.unwrap()));

    std::fs::remove_dir_all(p.join("obs")).unwrap();
    let e = z.obs().await.err().unwrap().to_string();
    assert!(e.contains("missing dataframe obs"), "{e}");
}

/// raw/X reads like X when it has X's cells and genes, sparse or dense;
/// with other genes, or none written, there is no raw.
#[tokio::test]
async fn raw_with_x_genes_reads_like_x() {
    let dir = tempfile::tempdir().unwrap();
    // `src` with `x` copied to raw/X and var to raw/var.
    let with_raw = |name: &str, src: &str, x: &str| {
        let p = dir.path().join(name);
        copy_dir(&fixture(src), &p);
        std::fs::write(
            p.join("raw/zarr.json"),
            r#"{"zarr_format":3,"node_type":"group","attributes":{"encoding-type":"raw","encoding-version":"0.1.0"}}"#,
        )
        .unwrap();
        copy_dir(&p.join(x), &p.join("raw/X"));
        copy_dir(&p.join("var"), &p.join("raw/var"));
        p
    };
    let raw_meta = |p: &Path| {
        let mut z = ZarrAdReader::open(p, CHUNK).unwrap();
        futures::executor::block_on(z.raw_meta())
    };

    assert!(raw_meta(&fixture("small_v3.zarr")).unwrap().is_none());
    let mut plain = ZarrAdReader::open(fixture("small_v3.zarr"), CHUNK).unwrap();
    assert!(plain.raw_var().await.unwrap().is_none());
    for (name, src) in [("sparse", "small_v3.zarr"), ("dense", "varied_v3.zarr")] {
        let mut z = ZarrAdReader::open(with_raw(name, src, "X"), CHUNK).unwrap();
        let meta = z.raw_meta().await.unwrap().expect(name);
        let raw_var = format!("{:?}", z.raw_var().await.unwrap().unwrap());
        assert_eq!(raw_var, format!("{:?}", z.var().await.unwrap()), "{name}");
        let raw = chunks(z.raw_stream(&meta, CHUNK)).await;
        assert_eq!(raw, chunks(z.x_stream()).await, "{name}");
    }

    let p = with_raw("other_genes", "small_v3.zarr", "X");
    std::fs::remove_dir_all(p.join("raw/var")).unwrap();
    assert!(raw_meta(&p).unwrap().is_none());

    let e = raw_meta(&with_raw("rank1", "small_v3.zarr", "obs/score"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("must be 2-D, has rank 1"), "{e}");
}

/// Regression: a uint8 or int8 column (CELLxGENE `var/Selected`) was read as
/// bool. Only a bool array is bool.
#[tokio::test]
async fn one_byte_int_columns_stay_ints() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("s.zarr");
    copy_dir(&fixture("small_v3.zarr"), &p);
    let n = ZarrAdReader::open(&p, CHUNK).unwrap().shape().0;
    for (name, dtype, byte) in [("u8col", "uint8", 1u8), ("i8col", "int8", 0xffu8)] {
        let a = p.join("obs").join(name);
        std::fs::create_dir_all(a.join("c")).unwrap();
        let meta = json!({
            "zarr_format": 3, "node_type": "array", "shape": [n], "data_type": dtype,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [n]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": [{"name": "bytes"}],
            "attributes": {"encoding-type": "array", "encoding-version": "0.2.0"}
        });
        std::fs::write(a.join("zarr.json"), meta.to_string()).unwrap();
        std::fs::write(a.join("c/0"), vec![byte; n]).unwrap();
    }
    let f = p.join("obs/zarr.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
    let order = v["attributes"]["column-order"].as_array_mut().unwrap();
    order.push(json!("u8col"));
    order.push(json!("i8col"));
    std::fs::write(&f, v.to_string()).unwrap();

    let obs = ZarrAdReader::open(&p, CHUNK).unwrap().obs().await.unwrap();
    let col = |name: &str| {
        obs.columns
            .iter()
            .find(|c| c.name == name)
            .unwrap()
            .data
            .clone()
    };
    assert!(matches!(col("u8col"), ColumnData::Int(v) if v == vec![1; n]));
    assert!(matches!(col("i8col"), ColumnData::Int(v) if v == vec![-1; n]));
}

/// Every typed uns dtype, 1-D to 3-D, at the top level and nested, survives
/// a write and read through h5ad, h5seurat and Zarr unchanged.
#[tokio::test]
async fn typed_uns_arrays_round_trip_in_every_format() {
    use crate::dtype::{DataType, TypedVec};
    use crate::h5ad::H5AdWriter;
    use crate::h5seurat::{H5SeuratReader, H5SeuratWriter};
    use crate::ir::{
        Embeddings, ObsTable, SparseMatrixCSR, UnsArray, UnsData, UnsTable, VarTable,
        UNS_TYPED_MIN as N,
    };

    let arr = |shape: Vec<usize>, data| UnsArray { shape, data };
    let r = || 0..N;
    let uns = UnsTable {
        raw: serde_json::json!({"a": {"k": 1}}),
        arrays: [
            (
                "a/b/i8",
                arr(vec![N], UnsData::I8(r().map(|i| i as i8).collect())),
            ),
            (
                "a/i16",
                arr(
                    vec![N / 4, 4],
                    UnsData::I16(r().map(|i| i as i16).collect()),
                ),
            ),
            (
                "i32",
                arr(
                    vec![N / 8, 4, 2],
                    UnsData::I32(r().map(|i| -(i as i32)).collect()),
                ),
            ),
            (
                "i64",
                arr(vec![N], UnsData::I64(r().map(|i| i as i64 - 5).collect())),
            ),
            (
                "u8",
                arr(vec![N / 2, 2], UnsData::U8(r().map(|i| i as u8).collect())),
            ),
            (
                "u16",
                arr(vec![N], UnsData::U16(r().map(|i| i as u16).collect())),
            ),
            (
                "u32",
                arr(vec![N], UnsData::U32(r().map(|i| i as u32 * 3).collect())),
            ),
            (
                "u64",
                arr(vec![N], UnsData::U64(r().map(|i| i as u64 * 7).collect())),
            ),
            (
                "f32",
                arr(
                    vec![N / 2, 2],
                    UnsData::F32(r().map(|i| i as f32 / 2.0).collect()),
                ),
            ),
            (
                "f64",
                arr(vec![N], UnsData::F64(r().map(|i| i as f64 / 3.0).collect())),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect(),
    };

    async fn write(mut w: Box<dyn DatasetWriter>, uns: &UnsTable) {
        let names = |p: &str| (0..2).map(|i| format!("{p}{i}")).collect();
        w.write_obs(&ObsTable {
            index: names("c"),
            columns: vec![],
        })
        .await
        .unwrap();
        w.write_var(&VarTable {
            index: names("g"),
            columns: vec![],
        })
        .await
        .unwrap();
        w.write_obsm(&Embeddings::default()).await.unwrap();
        w.write_uns(uns).await.unwrap();
        let x = SparseMatrixCSR {
            shape: (2, 2),
            indptr: vec![0, 0, 0],
            indices: vec![],
            data: TypedVec::F32(vec![]),
        };
        w.write_x_chunk(&MatrixChunk {
            row_offset: 0,
            nrows: 2,
            data: x,
        })
        .await
        .unwrap();
        w.finalize().await.unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let (h5ad, seurat, zarr) = (
        dir.path().join("u.h5ad"),
        dir.path().join("u.h5seurat"),
        dir.path().join("u.zarr"),
    );
    let f32 = DataType::F32;
    let h5ad_w = H5AdWriter::create_compressed(&h5ad, 2, 2, f32, Some(4)).unwrap();
    write(Box::new(h5ad_w), &uns).await;
    let seurat_w = H5SeuratWriter::create(&seurat, 2, 2, f32, None, None, None, false).unwrap();
    write(Box::new(seurat_w), &uns).await;
    write(
        Box::new(ZarrAdWriter::create(&zarr, 2, 2, f32).unwrap()),
        &uns,
    )
    .await;

    let readers: Vec<(&str, Box<dyn DatasetReader>)> = vec![
        ("h5ad", Box::new(H5AdReader::open(&h5ad, CHUNK).unwrap())),
        (
            "h5seurat",
            Box::new(H5SeuratReader::open(&seurat, CHUNK, None, None).unwrap()),
        ),
        ("zarr", Box::new(ZarrAdReader::open(&zarr, CHUNK).unwrap())),
    ];
    for (name, mut rd) in readers {
        let got = rd.uns().await.unwrap();
        assert_eq!(got.arrays, uns.arrays, "{name}");
        assert_eq!(got.raw["a"]["k"], 1, "{name}");
        assert_eq!(got.keys(), uns.keys(), "{name}");
    }
}

/// X streamed in chunks that straddle zarr chunk boundaries and fill more
/// than one buffered store (8 zarr chunks) reads back exactly.
#[tokio::test]
async fn writer_x_across_buffered_chunk_stores() {
    use crate::dtype::{DataType, TypedVec};
    use crate::ir::{ObsTable, SparseMatrixCSR};
    let (n_vars, row_nnz) = (6_000_000usize, [5_000_001usize, 4_000_003, 7]);
    let row = |r: usize| -> (Vec<u32>, Vec<i32>) {
        let idx = (0..row_nnz[r]).map(|k| k as u32).collect();
        let val = (0..row_nnz[r]).map(|k| (k * 3 + r) as i32).collect();
        (idx, val)
    };
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("big.zarr");
    let mut w = ZarrAdWriter::create(&out, 3, n_vars, DataType::I32).unwrap();
    w.write_obs(&ObsTable {
        index: vec!["a".into(), "b".into(), "c".into()],
        columns: vec![],
    })
    .await
    .unwrap();
    for (r, &nnz) in row_nnz.iter().enumerate() {
        let (indices, data) = row(r);
        w.write_x_chunk(&MatrixChunk {
            row_offset: r,
            nrows: 1,
            data: SparseMatrixCSR {
                shape: (1, n_vars),
                indptr: vec![0, nnz as u64],
                indices,
                data: TypedVec::I32(data),
            },
        })
        .await
        .unwrap();
    }
    w.finalize().await.unwrap();

    let mut rd = ZarrAdReader::open(&out, 1).unwrap();
    let mut s = rd.x_stream();
    let mut r = 0;
    while let Some(c) = s.next().await {
        let c = c.unwrap();
        let (idx, val) = row(r);
        assert_eq!(c.data.indices, idx, "row {r} indices");
        assert_eq!(c.data.data, TypedVec::I32(val), "row {r} data");
        r += 1;
    }
    assert_eq!(r, 3);
}
