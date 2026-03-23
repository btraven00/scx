use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::dtype::DataType;
use crate::error::Result;
use crate::ir::{Embeddings, MatrixChunk, ObsTable, UnsTable, VarTable};

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

    /// Write a chunk of the count matrix. Chunks must arrive in row order.
    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;

    /// Finalize the output (flush, write footers, etc.).
    async fn finalize(&mut self) -> Result<()>;
}
