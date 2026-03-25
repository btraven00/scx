use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::dtype::DataType;
use crate::error::Result;
use crate::ir::{Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable, VarTable, Varm};

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

    /// Read embedding matrices.
    async fn obsm(&mut self) -> Result<Embeddings>;

    /// Read unstructured metadata.
    async fn uns(&mut self) -> Result<UnsTable>;

    /// Read variable embedding matrices (e.g., gene loadings).
    async fn varm(&mut self) -> Result<Varm>;

    /// Return metadata (name, shape, indptr) for each additional count-matrix layer.
    /// The indptr is loaded eagerly (cheap: n_obs+1 entries); data/indices are streamed
    /// via `layer_stream`.
    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>>;

    /// Return metadata for each pairwise observation matrix (neighbor graphs, etc.).
    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>>;

    /// Stream row-chunks for the named layer.  `meta` must come from `layer_metas()`.
    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>>;

    /// Stream row-chunks for the named obsp matrix.  `meta` must come from `obsp_metas()`.
    fn obsp_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>>;

    /// Stream the count matrix as row-chunks.
    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>>;
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
    async fn begin_sparse(&mut self, group_prefix: &str, name: &str, meta: &SparseMatrixMeta) -> Result<()>;

    /// Append a row-chunk to the currently open sparse matrix.
    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;

    /// Finalize the currently open sparse matrix (write indptr, shape, etc.).
    async fn end_sparse(&mut self) -> Result<()>;

    /// Write a chunk of the count matrix. Chunks must arrive in row order.
    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;

    /// Finalize the output (flush, write footers, etc.).
    async fn finalize(&mut self) -> Result<()>;
}
