//! Streaming Parquet reader over an [`object_store`] handle.
//!
//! The matrix encoding (per-cell lists, dense, …) is sniffed at [`open`](ParquetReader::open)
//! into a [`ParquetLayout`] and dispatched per batch — see [`super::layout`].

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow::datatypes::DataType as ArrowType;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::{pin_mut, Stream, StreamExt};
use object_store::path::Path as StorePath;
use object_store::ObjectStore;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;

use super::layout::ParquetLayout;
use super::{net_err, GeneDict};
use crate::dtype::DataType;
use crate::error::Result;
use crate::ir::{
    Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable, VarTable,
    Varm,
};
use crate::stream::DatasetReader;

/// Column whose values become the obs index, when present (Tahoe-100M barcode).
const INDEX_COLUMN: &str = "BARCODE_SUB_LIB_ID";

/// Streams a Parquet dataset off any `object_store` backend — in-memory, local
/// filesystem, or cloud (S3/GCS/HTTP). The on-disk matrix encoding is detected
/// at open time; see [`ParquetLayout`].
pub struct ParquetReader {
    store: Arc<dyn ObjectStore>,
    path: StorePath,
    layout: ParquetLayout,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    /// `token_id → column` map when reading via a gene dictionary; `None` means
    /// the `genes` integers are used as direct column indices.
    token_map: Option<Arc<[i32]>>,
    /// Gene axis from the dictionary; empty when no dictionary was supplied.
    var: VarTable,
}

impl ParquetReader {
    /// Open a Parquet object for streaming.
    ///
    /// Reads only the footer (schema + row count) up front; row groups stream
    /// lazily via [`DatasetReader::x_stream`]. The layout is sniffed from the
    /// schema.
    ///
    /// `gene_dict` (a loaded `gene_metadata`) applies to the per-cell-list
    /// layout: it supplies the token→column map and the `var` axis, and
    /// **derives `n_vars`** (so `n_vars` may be `None`). Without a dictionary,
    /// `n_vars` is **required** for per-cell lists (the file has no gene axis)
    /// and **derived** for dense; a value contradicting a dense layout's column
    /// count is an error. `chunk_size` is the Parquet batch size — rows per
    /// emitted [`MatrixChunk`].
    pub async fn open(
        store: Arc<dyn ObjectStore>,
        path: impl Into<StorePath>,
        n_vars: Option<usize>,
        gene_dict: Option<GeneDict>,
        chunk_size: usize,
    ) -> Result<Self> {
        let path = path.into();
        let object_reader = ParquetObjectReader::new(store.clone(), path.clone());
        let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
            .await
            .map_err(net_err)?;

        let layout = ParquetLayout::sniff(builder.schema().as_ref())?;
        let is_per_cell = matches!(layout, ParquetLayout::PerCellLists { .. });

        // A gene dictionary only applies to the per-cell-list layout.
        let gene_dict = match gene_dict {
            Some(d) if is_per_cell => Some(d),
            Some(_) => {
                tracing::warn!(
                    "--genes ignored: the gene dictionary applies only to the per-cell-list \
                     layout, but this file is not in that layout"
                );
                None
            }
            None => None,
        };

        let (n_vars, token_map, var) = match &gene_dict {
            // Dictionary present: it owns the gene axis.
            Some(dict) => (dict.n_vars(), Some(dict.token_to_col.clone()), dict.var.clone()),
            None => {
                let n_vars = match layout.intrinsic_n_vars() {
                    Some(derived) => {
                        if let Some(req) = n_vars {
                            if req != derived {
                                return Err(net_err(format!(
                                    "n_vars={req} was provided but the dense layout has {derived} gene columns"
                                )));
                            }
                        }
                        derived
                    }
                    None => n_vars.ok_or_else(|| {
                        net_err(
                            "this Parquet layout does not carry the gene axis; provide n_vars \
                             (--n-vars) or a gene dictionary (--genes)",
                        )
                    })?,
                };
                (n_vars, None, VarTable::default())
            }
        };

        // For the supported one-cell-per-row layouts, footer rows == n_obs.
        let n_obs = builder.metadata().file_metadata().num_rows().max(0) as usize;

        Ok(Self {
            store,
            path,
            layout,
            n_obs,
            n_vars,
            chunk_size,
            token_map,
            var,
        })
    }

    /// Full-pass scan of the non-matrix scalar columns into an [`ObsTable`].
    async fn read_obs(&self) -> Result<ObsTable> {
        let object_reader = ParquetObjectReader::new(self.store.clone(), self.path.clone());
        let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
            .await
            .map_err(net_err)?;
        let schema = builder.schema().clone();

        // Materialise scalar columns that aren't matrix data (the layout knows
        // which columns carry X — list columns, or dense gene columns).
        let mut accs: Vec<ColumnAcc> = schema
            .fields()
            .iter()
            .filter(|f| !self.layout.is_matrix_column(f.name()))
            .filter_map(|f| ColumnAcc::for_field(f.name(), f.data_type()))
            .collect();

        // No obs columns: skip the scan entirely (a sequential index is enough).
        if accs.is_empty() {
            return Ok(ObsTable {
                index: string_index_from(self.n_obs),
                columns: Vec::new(),
            });
        }

        // Project only the obs columns so this pass doesn't also download the
        // (large) matrix columns — over the network that would be a second full
        // fetch on top of x_stream's.
        let roots: Vec<usize> = accs
            .iter()
            .filter_map(|a| schema.index_of(a.name.as_str()).ok())
            .collect();
        let mask = ProjectionMask::roots(builder.parquet_schema(), roots);

        let stream = builder
            .with_batch_size(self.chunk_size)
            .with_projection(mask)
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
        // Both supported layouts emit f32 (dense f64 columns are cast).
        DataType::F32
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        self.read_obs().await
    }

    async fn var(&mut self) -> Result<VarTable> {
        // Populated from the gene dictionary when one was supplied; else empty.
        Ok(self.var.clone())
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
        // Clone the cheap handles + layout so the stream owns them — the async
        // footer read is deferred into the stream body (x_stream is sync-returns-stream).
        let store = self.store.clone();
        let path = self.path.clone();
        let chunk_size = self.chunk_size;
        let layout = self.layout.clone();
        let token_map = self.token_map.clone();
        Box::pin(async_stream::try_stream! {
            let object_reader = ParquetObjectReader::new(store, path);
            let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
                .await
                .map_err(net_err)?;
            // Project only the matrix columns so the X read skips the obs columns.
            let roots: Vec<usize> = layout
                .matrix_column_names()
                .iter()
                .filter_map(|n| builder.schema().index_of(n).ok())
                .collect();
            let mask = ProjectionMask::roots(builder.parquet_schema(), roots);
            let stream = builder
                .with_batch_size(chunk_size)
                .with_projection(mask)
                .build()
                .map_err(net_err)?;
            pin_mut!(stream);
            let mut row_offset = 0usize;
            while let Some(batch) = stream.next().await {
                let batch = batch.map_err(net_err)?;
                let chunk = layout.batch_to_chunk(&batch, n_vars, row_offset, token_map.as_deref())?;
                row_offset += chunk.nrows;
                yield chunk;
            }
        })
    }
}

fn string_index_from(n: usize) -> Vec<String> {
    (0..n).map(|i| i.to_string()).collect()
}

/// Accumulates one scalar obs column across record batches.
struct ColumnAcc {
    name: String,
    data: ColumnData,
}

impl ColumnAcc {
    /// Returns an accumulator for scalar columns we can materialise; `None` for
    /// list columns and unsupported arrow types (skipped from obs).
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
