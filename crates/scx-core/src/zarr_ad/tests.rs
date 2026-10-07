//! The Zarr reader's contract is "whatever H5AdReader returns for the same
//! AnnData". The fixtures (scripts/prepare_zarr_fixtures.py) are one AnnData
//! written three ways by anndata itself: h5ad, Zarr v2 and Zarr v3. Every
//! slot is compared through its Debug form, which is exact and, unlike `==`,
//! treats NaN (nullable columns) consistently.

use std::path::{Path, PathBuf};

use futures::{Stream, StreamExt};

use super::ZarrAdReader;
use crate::detect::{detect, Format};
use crate::error::Result;
use crate::h5ad::H5AdReader;
use crate::ir::MatrixChunk;
use crate::stream::DatasetReader;

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
    let mut z = ZarrAdReader::open(fixture(zarr), CHUNK).unwrap();
    let mut h = H5AdReader::open(fixture(h5ad), CHUNK).unwrap();

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
