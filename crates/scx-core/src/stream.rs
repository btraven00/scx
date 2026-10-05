use std::ops::Range;
use std::pin::Pin;

use async_trait::async_trait;
use futures::{stream, Stream};

use crate::dtype::DataType;
use crate::error::{Result, ScxError};
use crate::ir::{Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable, VarTable, Varm};

/// A boxed stream of row chunks, as every reader returns.
pub type ChunkStream<'a> = Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>>;

/// Rows `0..n_rows` as consecutive chunks of up to `chunk_size` rows, each
/// produced by `read` when the stream is polled. Every reader's X, layer and
/// obsp stream is this loop around a function that reads one row range.
pub(crate) fn row_chunks<'a, F>(n_rows: usize, chunk_size: usize, mut read: F) -> ChunkStream<'a>
where
    F: FnMut(Range<usize>) -> Result<MatrixChunk> + Send + 'a,
{
    let step = chunk_size.max(1);
    Box::pin(stream::iter(
        (0..n_rows)
            .step_by(step)
            .map(move |start| read(start..(start + step).min(n_rows))),
    ))
}

fn no_such_matrix<'a>(what: &str) -> ChunkStream<'a> {
    let msg = format!("this format has no {what} matrices");
    Box::pin(stream::once(
        async move { Err(ScxError::InvalidFormat(msg)) },
    ))
}

/// Reads a single-cell dataset as a stream of matrix chunks plus metadata.
#[async_trait]
pub trait DatasetReader: Send {
    /// Dataset shape: (n_obs, n_vars).
    fn shape(&self) -> (usize, usize);

    /// Data type of the count matrix.
    fn dtype(&self) -> DataType;

    /// Read cell metadata.
    async fn obs(&mut self) -> Result<ObsTable>;

    /// Read feature metadata.
    async fn var(&mut self) -> Result<VarTable>;

    // The metadata methods below default to "none" for formats that have no
    // such slot (MatrixMarket, 10x, BPCells, Parquet).

    /// Read embedding matrices.
    async fn obsm(&mut self) -> Result<Embeddings> {
        Ok(Embeddings::default())
    }

    /// Read unstructured metadata.
    async fn uns(&mut self) -> Result<UnsTable> {
        Ok(UnsTable::default())
    }

    /// Read variable embedding matrices (e.g., gene loadings).
    async fn varm(&mut self) -> Result<Varm> {
        Ok(Varm::default())
    }

    /// Return metadata (name, shape, indptr) for each additional count-matrix layer.
    /// The indptr is loaded eagerly (cheap: n_obs+1 entries); data/indices are streamed
    /// via `layer_stream`.
    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    /// Return metadata for each pairwise observation matrix (neighbor graphs, etc.).
    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    /// Stream row-chunks for the named layer.  `meta` must come from `layer_metas()`.
    ///
    /// The default is for formats without any: it yields an error, since
    /// reaching it means a reader listed a matrix it can't stream.
    fn layer_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> ChunkStream<'a> {
        no_such_matrix("layer")
    }

    /// Stream row-chunks for the named obsp matrix.  `meta` must come from `obsp_metas()`.
    ///
    /// The default is for formats without any: it yields an error, since
    /// reaching it means a reader listed a matrix it can't stream.
    fn obsp_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> ChunkStream<'a> {
        no_such_matrix("obsp")
    }

    /// CSR row-pointer array for the main X matrix (n_obs+1 entries).
    /// Returns an empty slice when not available (dense X, BPCells, etc.).
    /// Free to call — indptr is loaded at open time, not streamed.
    fn x_indptr(&self) -> &[u64] {
        &[]
    }

    /// Stream the count matrix as row-chunks.
    fn x_stream(&mut self) -> ChunkStream<'_>;
}

/// Writes a single-cell dataset from a stream of matrix chunks plus metadata.
#[async_trait]
pub trait DatasetWriter: Send {
    /// Write cell metadata.
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()>;

    /// Write feature metadata.
    async fn write_var(&mut self, var: &VarTable) -> Result<()>;

    /// Write embedding matrices.
    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()>;

    /// Write unstructured metadata.
    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()>;

    /// Write variable embedding matrices (e.g., gene loadings).
    async fn write_varm(&mut self, varm: &Varm) -> Result<()>;

    /// Begin writing a named sparse matrix (layer or obsp).
    /// `group_prefix` is e.g. "layers" or "obsp".
    /// Must be followed by one or more `write_sparse_chunk` calls and then `end_sparse`.
    async fn begin_sparse(
        &mut self,
        group_prefix: &str,
        name: &str,
        meta: &SparseMatrixMeta,
    ) -> Result<()>;

    /// Append a row-chunk to the currently open sparse matrix.
    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;

    /// Finalize the currently open sparse matrix (write indptr, shape, etc.).
    async fn end_sparse(&mut self) -> Result<()>;

    /// Write a chunk of the count matrix. Chunks must arrive in row order.
    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;

    /// Finalize the output (flush, write footers, etc.).
    async fn finalize(&mut self) -> Result<()>;
}
