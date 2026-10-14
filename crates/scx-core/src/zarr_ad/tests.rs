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
use crate::ir::MatrixChunk;
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
    assert_store_same_as_h5ad(&fixture(zarr), &fixture(h5ad), zarr).await;
}

async fn assert_store_same_as_h5ad(zarr_path: &Path, h5ad_path: &Path, zarr: &str) {
    let mut z = ZarrAdReader::open(zarr_path, CHUNK).unwrap();
    let mut h = H5AdReader::open(h5ad_path, CHUNK).unwrap();

    assert_eq!(z.shape(), h.shape(), "{zarr}: shape");
    assert_eq!(z.dtype(), h.dtype(), "{zarr}: dtype");
    assert_eq!(z.x_indptr(), h.x_indptr(), "{zarr}: X indptr");
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
    assert_eq!(
        z.uns().await.unwrap().raw,
        h.uns().await.unwrap().raw,
        "{zarr}: uns"
    );

    let (zl, hl) = (
        z.layer_metas().await.unwrap(),
        h.layer_metas().await.unwrap(),
    );
    assert_eq!(format!("{zl:?}"), format!("{hl:?}"), "{zarr}: layer metas");
    for (zm, hm) in zl.iter().zip(&hl) {
        assert_eq!(
            chunks(z.layer_stream(zm, CHUNK)).await,
            chunks(h.layer_stream(hm, CHUNK)).await,
            "{zarr}: layers['{}']",
            zm.name
        );
    }
    let (zp, hp) = (z.obsp_metas().await.unwrap(), h.obsp_metas().await.unwrap());
    assert_eq!(format!("{zp:?}"), format!("{hp:?}"), "{zarr}: obsp metas");
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

/// Everything the h5ad reader returns, written as a Zarr v3 store, reads back
/// identically through the Zarr reader (the writer's contract is the reader's
/// in reverse) and as the h5ad it came from.
#[tokio::test]
async fn writer_roundtrips_what_h5ad_reads() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("written.zarr");
    let src = fixture("small.h5ad");
    {
        let mut h = H5AdReader::open(&src, CHUNK).unwrap();
        let (n_obs, n_vars) = h.shape();
        let mut w = ZarrAdWriter::create(&out, n_obs, n_vars, h.dtype()).unwrap();
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
    assert_eq!(detect(&out), Some(Format::ZarrAd));
    assert_store_same_as_h5ad(&out, &src, "written.zarr").await;
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
