use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::dtype::DataType;
use crate::error::Result;
use crate::ir::{Embeddings, Layers, MatrixChunk, Obsp, ObsTable, UnsTable, VarTable, Varm, Varp};

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

    /// Read additional count matrices (layers).
    async fn layers(&mut self) -> Result<Layers>;

    /// Read pairwise observation matrices (e.g., neighbor graphs).
    async fn obsp(&mut self) -> Result<Obsp>;

    /// Read pairwise variable matrices.
    async fn varp(&mut self) -> Result<Varp>;

    /// Read variable embedding matrices (e.g., gene loadings).
    async fn varm(&mut self) -> Result<Varm>;

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

    /// Write additional count matrices (layers).
    async fn write_layers(&mut self, layers: &Layers) -> Result<()>;

    /// Write pairwise observation matrices (e.g., neighbor graphs).
    async fn write_obsp(&mut self, obsp: &Obsp) -> Result<()>;

    /// Write pairwise variable matrices.
    async fn write_varp(&mut self, varp: &Varp) -> Result<()>;

    /// Write variable embedding matrices (e.g., gene loadings).
    async fn write_varm(&mut self, varm: &Varm) -> Result<()>;

    /// Write a chunk of the count matrix. Chunks must arrive in row order.
    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()>;

    /// Finalize the output (flush, write footers, etc.).
    async fn finalize(&mut self) -> Result<()>;
}
