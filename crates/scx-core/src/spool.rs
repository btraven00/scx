//! Downloaded (or piped-in) copies of a source, hashed as they are written.
//!
//! Formats that need random access (HDF5, `.rds`'s second pass) can't read a
//! network stream or a pipe, so the bytes go to a temporary file first. The
//! hash ([`HashAlgo`]) is computed on the way through, for provenance, with no
//! second read.

use std::io::{BufWriter, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::provenance::{HashAlgo, Hasher};
use async_trait::async_trait;

use crate::dtype::DataType;
use crate::error::Result;
use crate::ir::{Embeddings, ObsTable, SparseMatrixMeta, UnsTable, VarTable, Varm};
use crate::stream::{ChunkStream, DatasetReader};

/// A temporary copy of a source, deleted when dropped.
pub struct Spool {
    path: PathBuf,
    /// `"<algo>:<hex>"` of the bytes, hashed as they were written.
    hash: String,
}

impl Spool {
    /// Where the copy is on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `"<algo>:<hex>"` of the copied bytes.
    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// Copy everything `src` yields to a temporary file named after `name`
    /// (its extension is kept, for format detection).
    pub fn from_reader(name: &str, mut src: impl Read, algo: HashAlgo) -> Result<Spool> {
        let mut w = SpoolWriter::create(name, algo)?;
        std::io::copy(&mut src, &mut w)?;
        w.finish()
    }
}

impl Drop for Spool {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Writes a [`Spool`], hashing as it goes. Dropped unfinished (an error
/// midway), the partial file is deleted.
struct SpoolWriter {
    spool: Spool,
    file: BufWriter<std::fs::File>,
    hasher: Hasher,
}

impl SpoolWriter {
    fn create(name: &str, algo: HashAlgo) -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("scx-{}-{nanos}-{name}", std::process::id()));
        let file = BufWriter::new(std::fs::File::create(&path)?);
        Ok(Self {
            spool: Spool {
                path,
                hash: String::new(),
            },
            file,
            hasher: Hasher::new(algo),
        })
    }

    fn finish(mut self) -> Result<Spool> {
        self.flush()?;
        self.spool.hash = self.hasher.finish();
        Ok(self.spool)
    }
}

impl Write for SpoolWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// Download `url` (any object-store location) to a temporary file, deleted
/// when the returned [`Spool`] is dropped.
#[cfg(feature = "net")]
pub async fn spool(url: &str, algo: HashAlgo) -> Result<Spool> {
    use futures::StreamExt;
    use object_store::ObjectStoreExt;

    let (store, path) = crate::net::resolve_store(url)?;
    let name = path.filename().unwrap_or("download").to_string();
    let mut w = SpoolWriter::create(&name, algo)?;
    let net = |e: object_store::Error| crate::parquet::net_err(format!("download '{url}': {e}"));
    let mut body = store.get(&path).await.map_err(net)?.into_stream();
    while let Some(bytes) = body.next().await {
        w.write_all(&bytes.map_err(net)?)?;
    }
    w.finish()
}

/// A reader over a [`Spool`], which it keeps (and so the file) until dropped.
/// Every method forwards to `inner`; `source_hash` is the spool's.
pub struct Spooled {
    // Field order matters: `inner` (and its open handles) drops before the
    // spool deletes the file.
    inner: Box<dyn DatasetReader + Send>,
    spool: Spool,
}

impl Spooled {
    pub fn new(inner: Box<dyn DatasetReader + Send>, spool: Spool) -> Self {
        Self { inner, spool }
    }
}

#[async_trait]
impl DatasetReader for Spooled {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn dtype(&self) -> DataType {
        self.inner.dtype()
    }
    fn source_hash(&self) -> Option<String> {
        Some(self.spool.hash().to_string())
    }
    async fn obs(&mut self) -> Result<ObsTable> {
        self.inner.obs().await
    }
    async fn var(&mut self) -> Result<VarTable> {
        self.inner.var().await
    }
    async fn obsm(&mut self) -> Result<Embeddings> {
        self.inner.obsm().await
    }
    async fn uns(&mut self) -> Result<UnsTable> {
        self.inner.uns().await
    }
    async fn varm(&mut self) -> Result<Varm> {
        self.inner.varm().await
    }
    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        self.inner.layer_metas().await
    }
    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        self.inner.obsp_metas().await
    }
    async fn raw_meta(&mut self) -> Result<Option<SparseMatrixMeta>> {
        self.inner.raw_meta().await
    }
    async fn raw_var(&mut self) -> Result<Option<VarTable>> {
        self.inner.raw_var().await
    }
    fn raw_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk: usize) -> ChunkStream<'a> {
        self.inner.raw_stream(meta, chunk)
    }
    fn layer_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk: usize) -> ChunkStream<'a> {
        self.inner.layer_stream(meta, chunk)
    }
    fn obsp_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk: usize) -> ChunkStream<'a> {
        self.inner.obsp_stream(meta, chunk)
    }
    fn x_indptr(&self) -> &[u64] {
        self.inner.x_indptr()
    }
    fn x_col_ptr(&self) -> &[u64] {
        self.inner.x_col_ptr()
    }
    fn x_stream(&mut self) -> ChunkStream<'_> {
        self.inner.x_stream()
    }
    fn x_stream_cols(&mut self, ranges: Vec<Range<usize>>) -> ChunkStream<'_> {
        self.inner.x_stream_cols(ranges)
    }
    fn reads_row_ranges(&self) -> bool {
        self.inner.reads_row_ranges()
    }
    fn x_stream_rows(&mut self, ranges: Vec<Range<usize>>) -> ChunkStream<'_> {
        self.inner.x_stream_rows(ranges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spool_copies_hashes_and_cleans_up() {
        let s = Spool::from_reader("x.h5ad", &b"abc"[..], HashAlgo::Sha256).unwrap();
        assert_eq!(std::fs::read(s.path()).unwrap(), b"abc");
        assert!(s.path().to_string_lossy().ends_with("x.h5ad"));
        // sha256("abc")
        assert_eq!(
            s.hash(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let p = s.path().to_path_buf();
        drop(s);
        assert!(!p.exists());
    }

    /// Every `Spooled` method gives what the reader underneath gives.
    #[tokio::test]
    async fn spooled_forwards_to_the_inner_reader() {
        use futures::StreamExt;
        let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny/tiny_raw.h5ad");
        let opts = crate::OpenOptions::default();
        let spool = Spool::from_reader(
            "x.h5ad",
            std::fs::File::open(&src).unwrap(),
            HashAlgo::default(),
        )
        .unwrap();
        let path = spool.path().to_str().unwrap().to_string();
        let mut s = Spooled::new(crate::open(&path, &opts).await.unwrap(), spool);
        let mut d = crate::open(src.to_str().unwrap(), &opts).await.unwrap();
        let dbg = |x: &dyn std::fmt::Debug| format!("{x:?}");
        async fn all(st: ChunkStream<'_>) -> String {
            let v: Vec<_> = st.map(|c| c.unwrap()).collect().await;
            format!("{v:?}")
        }

        assert_eq!(s.shape(), d.shape());
        assert_eq!(s.dtype(), d.dtype());
        assert!(s.source_hash().unwrap().starts_with("blake3:"));
        assert_eq!(dbg(&s.obs().await.unwrap()), dbg(&d.obs().await.unwrap()));
        assert_eq!(dbg(&s.var().await.unwrap()), dbg(&d.var().await.unwrap()));
        assert_eq!(dbg(&s.obsm().await.unwrap()), dbg(&d.obsm().await.unwrap()));
        assert_eq!(dbg(&s.uns().await.unwrap()), dbg(&d.uns().await.unwrap()));
        assert_eq!(dbg(&s.varm().await.unwrap()), dbg(&d.varm().await.unwrap()));
        let raw = s.raw_meta().await.unwrap().expect("tiny_raw has raw");
        assert_eq!(
            dbg(&s.raw_var().await.unwrap()),
            dbg(&d.raw_var().await.unwrap())
        );
        assert_eq!(
            all(s.raw_stream(&raw, 3)).await,
            all(d.raw_stream(&raw, 3)).await
        );
        for m in s.layer_metas().await.unwrap() {
            assert_eq!(
                all(s.layer_stream(&m, 3)).await,
                all(d.layer_stream(&m, 3)).await
            );
        }
        for m in s.obsp_metas().await.unwrap() {
            assert_eq!(
                all(s.obsp_stream(&m, 3)).await,
                all(d.obsp_stream(&m, 3)).await
            );
        }
        assert_eq!(s.x_indptr(), d.x_indptr());
        assert_eq!(s.x_col_ptr(), d.x_col_ptr());
        assert_eq!(s.reads_row_ranges(), d.reads_row_ranges());
        assert_eq!(all(s.x_stream()).await, all(d.x_stream()).await);
        assert_eq!(
            all(s.x_stream_rows(vec![0..2])).await,
            all(d.x_stream_rows(vec![0..2])).await
        );
        // Both fail or both succeed (h5ad X is cell-major).
        let first = |mut st: ChunkStream<'_>| {
            futures::executor::block_on(st.next())
                .map(|r| r.map(|c| format!("{c:?}")).map_err(|e| e.to_string()))
        };
        let (a, b) = (
            first(s.x_stream_cols(vec![0..2])),
            first(d.x_stream_cols(vec![0..2])),
        );
        assert_eq!(a, b);
    }
}
