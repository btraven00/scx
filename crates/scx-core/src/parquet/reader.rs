//! Streaming Parquet reader over an [`object_store`] handle.

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, ListArray,
    StringArray,
};
use arrow::datatypes::DataType as ArrowType;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::{pin_mut, Stream, StreamExt};
use object_store::path::Path as StorePath;
use object_store::ObjectStore;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta,
    UnsTable, VarTable, Varm,
};
use crate::stream::DatasetReader;

/// Column whose values become the obs index, when present (Tahoe-100M barcode).
const INDEX_COLUMN: &str = "BARCODE_SUB_LIB_ID";

/// Streams a Parquet dataset (one cell per row, sparse list columns) off any
/// `object_store` backend — in-memory, local filesystem, or cloud (S3/GCS/Azure).
pub struct ParquetReader {
    store: Arc<dyn ObjectStore>,
    path: StorePath,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
}

impl ParquetReader {
    /// Open a Parquet object for streaming.
    ///
    /// Reads only the file footer (row count) up front — the row groups stream
    /// lazily via [`DatasetReader::x_stream`]. `n_vars` is supplied by the
    /// caller because the expression file does not carry the full gene axis
    /// (Tahoe sources it from the `gene_metadata` subset); `chunk_size` is the
    /// Parquet batch size, i.e. rows per emitted [`MatrixChunk`].
    pub async fn open(
        store: Arc<dyn ObjectStore>,
        path: impl Into<StorePath>,
        n_vars: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        let path = path.into();
        let object_reader = ParquetObjectReader::new(store.clone(), path.clone());
        let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
            .await
            .map_err(net_err)?;
        let n_obs = builder.metadata().file_metadata().num_rows().max(0) as usize;
        Ok(Self {
            store,
            path,
            n_obs,
            n_vars,
            chunk_size,
        })
    }

    /// Full-pass scan of the scalar (non-list) columns into an [`ObsTable`].
    async fn read_obs(&self) -> Result<ObsTable> {
        let object_reader = ParquetObjectReader::new(self.store.clone(), self.path.clone());
        let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
            .await
            .map_err(net_err)?;
        let schema = builder.schema().clone();

        // Pick out scalar columns we know how to materialise; list columns
        // (genes/expressions) carry X and are streamed separately, not here.
        let mut accs: Vec<ColumnAcc> = schema
            .fields()
            .iter()
            .filter_map(|f| ColumnAcc::for_field(f.name(), f.data_type()))
            .collect();

        let stream = builder
            .with_batch_size(self.chunk_size)
            .build()
            .map_err(net_err)?;
        pin_mut!(stream);
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(net_err)?;
            for acc in &mut accs {
                acc.append(&batch)?;
            }
        }

        // A barcode column, if present, becomes the index rather than a column.
        let index_pos = accs.iter().position(|a| a.name == INDEX_COLUMN);
        let index = match index_pos {
            Some(pos) => match &accs.remove(pos).data {
                ColumnData::String(v) => v.clone(),
                other => string_index_from(other.len()),
            },
            None => string_index_from(self.n_obs),
        };

        Ok(ObsTable {
            index,
            columns: accs
                .into_iter()
                .map(|a| Column {
                    name: a.name,
                    data: a.data,
                })
                .collect(),
        })
    }
}

#[async_trait]
impl DatasetReader for ParquetReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        // `expressions` is float32 in the Tahoe schema.
        DataType::F32
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        self.read_obs().await
    }

    async fn var(&mut self) -> Result<VarTable> {
        Ok(VarTable::default())
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        Ok(Embeddings::default())
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        Ok(UnsTable::default())
    }

    async fn varm(&mut self) -> Result<Varm> {
        Ok(Varm::default())
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    fn layer_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        Box::pin(futures::stream::empty())
    }

    fn obsp_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        Box::pin(futures::stream::empty())
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let n_vars = self.n_vars;
        // Clone the cheap handles so the stream owns them — the async footer
        // read is deferred into the stream body (x_stream is sync-returns-stream).
        let store = self.store.clone();
        let path = self.path.clone();
        let chunk_size = self.chunk_size;
        Box::pin(async_stream::try_stream! {
            let object_reader = ParquetObjectReader::new(store, path);
            let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
                .await
                .map_err(net_err)?;
            let stream = builder.with_batch_size(chunk_size).build().map_err(net_err)?;
            pin_mut!(stream);
            let mut row_offset = 0usize;
            while let Some(batch) = stream.next().await {
                let batch = batch.map_err(net_err)?;
                let chunk = batch_to_chunk(&batch, n_vars, row_offset)?;
                row_offset += chunk.nrows;
                yield chunk;
            }
        })
    }
}

/// Convert a network-stack error into [`ScxError::Net`] (keeps the variant free
/// of the `net`-only crate types).
fn net_err<E: std::fmt::Display>(e: E) -> ScxError {
    ScxError::Net(e.to_string())
}

fn string_index_from(n: usize) -> Vec<String> {
    (0..n).map(|i| i.to_string()).collect()
}

/// Turn one Parquet `RecordBatch` (parallel `genes`/`expressions` lists) into a
/// CSR row-chunk. Each batch row is one cell, i.e. one CSR row.
fn batch_to_chunk(batch: &RecordBatch, n_vars: usize, row_offset: usize) -> Result<MatrixChunk> {
    let genes = list_column(batch, "genes")?;
    let exprs = list_column(batch, "expressions")?;

    let nrows = batch.num_rows();
    let mut indptr: Vec<u64> = Vec::with_capacity(nrows + 1);
    indptr.push(0);
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();

    for r in 0..nrows {
        if genes.is_null(r) {
            indptr.push(indices.len() as u64);
            continue;
        }
        let row_genes = genes.value(r);
        let row_genes = row_genes
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| net_err("`genes` list values are not Int64"))?;
        let row_exprs = exprs.value(r);
        let row_exprs = row_exprs
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| net_err("`expressions` list values are not Float32"))?;

        if row_genes.len() != row_exprs.len() {
            return Err(net_err(format!(
                "row {}: genes/expressions length mismatch ({} vs {})",
                row_offset + r,
                row_genes.len(),
                row_exprs.len()
            )));
        }

        for k in 0..row_genes.len() {
            let col = row_genes.value(k);
            if col < 0 || col as usize >= n_vars {
                return Err(net_err(format!(
                    "row {}: gene index {} out of range for n_vars={}",
                    row_offset + r,
                    col,
                    n_vars
                )));
            }
            indices.push(col as u32);
            data.push(row_exprs.value(k));
        }
        indptr.push(indices.len() as u64);
    }

    Ok(MatrixChunk {
        row_offset,
        nrows,
        data: SparseMatrixCSR {
            shape: (nrows, n_vars),
            indptr,
            indices,
            data: TypedVec::F32(data),
        },
    })
}

fn list_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ScxError::MissingField(name.to_string()))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| net_err(format!("`{name}` column is not a List")))
}

/// Accumulates one scalar obs column across record batches.
struct ColumnAcc {
    name: String,
    data: ColumnData,
}

impl ColumnAcc {
    /// Returns an accumulator for scalar columns we can materialise; `None` for
    /// list columns (X) and unsupported arrow types (skipped from obs).
    fn for_field(name: &str, dtype: &ArrowType) -> Option<Self> {
        let data = match dtype {
            ArrowType::Utf8 | ArrowType::LargeUtf8 => ColumnData::String(Vec::new()),
            ArrowType::Int64 | ArrowType::Int32 => ColumnData::Int(Vec::new()),
            ArrowType::Float32 | ArrowType::Float64 => ColumnData::Float(Vec::new()),
            ArrowType::Boolean => ColumnData::Bool(Vec::new()),
            _ => return None,
        };
        Some(Self {
            name: name.to_string(),
            data,
        })
    }

    fn append(&mut self, batch: &RecordBatch) -> Result<()> {
        let Some(col) = batch.column_by_name(&self.name) else {
            return Ok(());
        };
        let any = col.as_any();
        match &mut self.data {
            ColumnData::String(out) => {
                let arr = any
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| net_err(format!("`{}` is not Utf8", self.name)))?;
                for i in 0..arr.len() {
                    out.push(if arr.is_null(i) {
                        String::new()
                    } else {
                        arr.value(i).to_string()
                    });
                }
            }
            ColumnData::Int(out) => {
                if let Some(arr) = any.downcast_ref::<Int64Array>() {
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) { 0 } else { arr.value(i) as i32 });
                    }
                } else if let Some(arr) = any.downcast_ref::<Int32Array>() {
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) { 0 } else { arr.value(i) });
                    }
                } else {
                    return Err(net_err(format!("`{}` is not an integer column", self.name)));
                }
            }
            ColumnData::Float(out) => {
                if let Some(arr) = any.downcast_ref::<Float64Array>() {
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) { 0.0 } else { arr.value(i) });
                    }
                } else if let Some(arr) = any.downcast_ref::<Float32Array>() {
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            0.0
                        } else {
                            arr.value(i) as f64
                        });
                    }
                } else {
                    return Err(net_err(format!("`{}` is not a float column", self.name)));
                }
            }
            ColumnData::Bool(out) => {
                let arr = any
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| net_err(format!("`{}` is not Boolean", self.name)))?;
                for i in 0..arr.len() {
                    out.push(!arr.is_null(i) && arr.value(i));
                }
            }
            ColumnData::Categorical { .. } => {}
        }
        Ok(())
    }
}
