//! Matrix-encoding layouts for single-cell Parquet.
//!
//! A Parquet file can encode the count matrix several ways. Rather than hardcode
//! one, the reader **sniffs** the Arrow schema at open time into a
//! [`ParquetLayout`] and dispatches each `RecordBatch` through the matching
//! converter. The transport and streaming loop are identical across layouts;
//! only the per-batch unpacking differs.
//!
//! Supported today:
//!
//! - [`ParquetLayout::PerCellLists`] — one cell per row, nonzeros in two
//!   parallel list columns (`genes: List<Int64>`, `expressions: List<Float32>`).
//!   The Tahoe-100M `expression_data` layout. Gene axis (`n_vars`) is external.
//! - [`ParquetLayout::Dense`] — one cell per row, one float column per gene.
//!   The scanpy `adata.to_df().to_parquet()` layout. `n_vars` is the float-column
//!   count (free); non-float columns become obs.
//!
//! Not yet: long-format COO triples (`cell_id, gene_id, value`). COO needs the
//! rows grouped by cell (a cross-batch buffer) and can't read `n_obs` from the
//! footer the way the per-cell layouts can — it's a separate increment. The
//! sniffer reports a clear error rather than silently misreading such a file.

use arrow::array::{Array, Float32Array, Float64Array, Int64Array, ListArray};
use arrow::datatypes::{DataType as ArrowType, Schema};
use arrow::record_batch::RecordBatch;

use super::net_err;
use crate::dtype::TypedVec;
use crate::error::{Result, ScxError};
use crate::ir::{MatrixChunk, SparseMatrixCSR};

/// Default column names for the per-cell-lists layout (Tahoe-100M).
const GENES_COL: &str = "genes";
const EXPRS_COL: &str = "expressions";

/// How the count matrix is laid out across a Parquet file's columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParquetLayout {
    /// One cell per row; nonzeros in parallel `genes`/`expressions` list columns.
    PerCellLists { genes: String, exprs: String },
    /// One cell per row; one dense float column per gene (in this order).
    Dense { gene_cols: Vec<String> },
}

impl ParquetLayout {
    /// Detect the layout from the file's Arrow schema.
    ///
    /// Per-cell-lists is checked first (most specific: two named list columns);
    /// otherwise any float columns are taken as a dense gene matrix. A file with
    /// neither shape — or a long-format/COO file we don't handle yet — errors.
    pub(crate) fn sniff(schema: &Schema) -> Result<Self> {
        let is_list = |name: &str| {
            schema
                .column_with_name(name)
                .map(|(_, f)| matches!(f.data_type(), ArrowType::List(_) | ArrowType::LargeList(_)))
                .unwrap_or(false)
        };
        if is_list(GENES_COL) && is_list(EXPRS_COL) {
            return Ok(Self::PerCellLists {
                genes: GENES_COL.to_string(),
                exprs: EXPRS_COL.to_string(),
            });
        }

        let gene_cols: Vec<String> = schema
            .fields()
            .iter()
            .filter(|f| matches!(f.data_type(), ArrowType::Float32 | ArrowType::Float64))
            .map(|f| f.name().to_string())
            .collect();
        if !gene_cols.is_empty() {
            return Ok(Self::Dense { gene_cols });
        }

        Err(net_err(
            "could not determine Parquet matrix encoding: expected `genes`+`expressions` \
             list columns (per-cell) or float columns (dense). Long-format COO \
             (cell/gene/value triples) is not supported yet.",
        ))
    }

    /// The gene axis if the layout determines it (Dense), else `None` — the
    /// per-cell layout does not carry the full gene set, so `n_vars` must be
    /// supplied by the caller.
    pub(crate) fn intrinsic_n_vars(&self) -> Option<usize> {
        match self {
            Self::PerCellLists { .. } => None,
            Self::Dense { gene_cols } => Some(gene_cols.len()),
        }
    }

    /// True if `name` is a column carrying matrix data (so it must be excluded
    /// from obs).
    pub(crate) fn is_matrix_column(&self, name: &str) -> bool {
        match self {
            Self::PerCellLists { genes, exprs } => name == genes || name == exprs,
            Self::Dense { gene_cols } => gene_cols.iter().any(|c| c == name),
        }
    }

    /// Convert one `RecordBatch` (one cell per row, for both supported layouts)
    /// into a CSR row-chunk starting at `row_offset`.
    pub(crate) fn batch_to_chunk(
        &self,
        batch: &RecordBatch,
        n_vars: usize,
        row_offset: usize,
    ) -> Result<MatrixChunk> {
        match self {
            Self::PerCellLists { genes, exprs } => {
                per_cell_lists_chunk(batch, genes, exprs, n_vars, row_offset)
            }
            Self::Dense { gene_cols } => dense_chunk(batch, gene_cols, row_offset),
        }
    }
}

/// Per-cell parallel lists → CSR. Each batch row is a cell; its `genes` list
/// holds the nonzero column indices and `expressions` the aligned values.
fn per_cell_lists_chunk(
    batch: &RecordBatch,
    genes_col: &str,
    exprs_col: &str,
    n_vars: usize,
    row_offset: usize,
) -> Result<MatrixChunk> {
    let genes = list_column(batch, genes_col)?;
    let exprs = list_column(batch, exprs_col)?;

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
            .ok_or_else(|| net_err(format!("`{genes_col}` list values are not Int64")))?;
        let row_exprs = exprs.value(r);
        let row_exprs = row_exprs
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| net_err(format!("`{exprs_col}` list values are not Float32")))?;

        if row_genes.len() != row_exprs.len() {
            return Err(net_err(format!(
                "row {}: {genes_col}/{exprs_col} length mismatch ({} vs {})",
                row_offset + r,
                row_genes.len(),
                row_exprs.len()
            )));
        }

        for k in 0..row_genes.len() {
            let col = row_genes.value(k);
            if col < 0 || col as usize >= n_vars {
                return Err(net_err(format!(
                    "row {}: gene index {col} out of range for n_vars={n_vars}",
                    row_offset + r,
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

/// Dense float columns → CSR. Each batch row is a cell; the gene columns are
/// read in order, and nonzero entries become CSR cells. Values are emitted as
/// f32 (f64 columns are cast — full f64 preservation is a follow-up).
fn dense_chunk(batch: &RecordBatch, gene_cols: &[String], row_offset: usize) -> Result<MatrixChunk> {
    let n_vars = gene_cols.len();
    // Downcast each gene column once per batch (not per row).
    let cols: Vec<DenseCol> = gene_cols
        .iter()
        .map(|name| DenseCol::new(batch, name))
        .collect::<Result<_>>()?;

    let nrows = batch.num_rows();
    let mut indptr: Vec<u64> = Vec::with_capacity(nrows + 1);
    indptr.push(0);
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();

    for r in 0..nrows {
        for (j, col) in cols.iter().enumerate() {
            let v = col.value(r);
            if v != 0.0 {
                indices.push(j as u32);
                data.push(v);
            }
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

/// A downcast handle to one dense gene column (f32 or f64), read as f32.
enum DenseCol<'a> {
    F32(&'a Float32Array),
    F64(&'a Float64Array),
}

impl<'a> DenseCol<'a> {
    fn new(batch: &'a RecordBatch, name: &str) -> Result<Self> {
        let col = batch
            .column_by_name(name)
            .ok_or_else(|| ScxError::MissingField(name.to_string()))?;
        if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
            Ok(DenseCol::F32(a))
        } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
            Ok(DenseCol::F64(a))
        } else {
            Err(net_err(format!("`{name}` is not a float column")))
        }
    }

    #[inline]
    fn value(&self, r: usize) -> f32 {
        match self {
            DenseCol::F32(a) => {
                if a.is_null(r) {
                    0.0
                } else {
                    a.value(r)
                }
            }
            DenseCol::F64(a) => {
                if a.is_null(r) {
                    0.0
                } else {
                    a.value(r) as f32
                }
            }
        }
    }
}

fn list_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ScxError::MissingField(name.to_string()))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| net_err(format!("`{name}` column is not a List")))
}
