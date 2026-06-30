//! Gene dictionary for the per-cell-list layout (Tahoe-100M `gene_metadata`).
//!
//! Tahoe stores the count matrix (`expression_data`) and the gene axis
//! (`gene_metadata`) in **separate** Parquet files. The `genes` column of the
//! matrix holds *token IDs* — an arbitrary integer vocabulary — not column
//! indices. This loads the dictionary and builds the two things the reader
//! needs to turn tokens into a real matrix:
//!
//! - `token_to_col`: a dense `token_id → column` lookup (sentinel `-1` for
//!   reserved/absent tokens, which is how the leading marker token is dropped),
//! - `var`: the gene axis (`index = ensembl_id`, plus a `gene_symbol` column).
//!
//! See `scratch/gene-metadata-mapping.md` for the full problem write-up.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use futures::{pin_mut, StreamExt};
use object_store::path::Path as StorePath;
use object_store::ObjectStore;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};

use super::net_err;
use crate::error::{Result, ScxError};
use crate::ir::{Column, ColumnData, VarTable};

const TOKEN_COL: &str = "token_id";
const ENSEMBL_COL: &str = "ensembl_id";
const SYMBOL_COL: &str = "gene_symbol";

/// Guard against a pathological (non-per-gene/hashed) vocabulary blowing up the
/// dense lookup. Any real gene tokenizer is in the tens of thousands; well under
/// this. Above it we'd want a HashMap instead (not implemented in this spike).
const MAX_DENSE_TOKEN: i64 = 50_000_000;

/// The gene axis + token→column lookup, loaded from a `gene_metadata` Parquet.
pub struct GeneDict {
    /// `token_to_col[token_id] = column`, or `-1` if the token isn't a gene.
    pub(crate) token_to_col: Arc<[i32]>,
    /// Gene axis: `index = ensembl_id`, with a `gene_symbol` column.
    pub(crate) var: VarTable,
}

impl GeneDict {
    /// Number of genes (the derived `n_vars`).
    pub fn n_vars(&self) -> usize {
        self.var.index.len()
    }

    /// Load and index a `gene_metadata` Parquet object (columns `token_id`,
    /// `ensembl_id`, `gene_symbol`). The file is small (~62.7k rows for Tahoe),
    /// so it is read fully into memory.
    pub async fn load(
        store: Arc<dyn ObjectStore>,
        path: impl Into<StorePath>,
        chunk_size: usize,
    ) -> Result<Self> {
        let path = path.into();
        let object_reader = ParquetObjectReader::new(store, path);
        let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
            .await
            .map_err(net_err)?;
        let stream = builder
            .with_batch_size(chunk_size)
            .build()
            .map_err(net_err)?;
        pin_mut!(stream);

        let mut token_ids: Vec<i64> = Vec::new();
        let mut ensembl: Vec<String> = Vec::new();
        let mut symbol: Vec<String> = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(net_err)?;
            append_i64(&batch, TOKEN_COL, &mut token_ids)?;
            append_utf8(&batch, ENSEMBL_COL, &mut ensembl)?;
            append_utf8(&batch, SYMBOL_COL, &mut symbol)?;
        }

        if token_ids.is_empty() {
            return Err(net_err("gene_metadata is empty (no token_id rows)"));
        }
        let max_token = token_ids.iter().copied().max().unwrap_or(-1);
        if max_token < 0 {
            return Err(net_err("gene_metadata token_id values are all negative"));
        }
        if max_token > MAX_DENSE_TOKEN {
            return Err(net_err(format!(
                "gene_metadata max token_id {max_token} exceeds the dense-lookup limit \
                 ({MAX_DENSE_TOKEN}); a sparse/hashed vocabulary is not supported yet"
            )));
        }

        // token_id → its row index (= matrix column). Later duplicate tokens win;
        // for a well-formed dictionary token_ids are unique.
        let mut token_to_col = vec![-1i32; max_token as usize + 1];
        for (row, &tok) in token_ids.iter().enumerate() {
            if tok >= 0 {
                token_to_col[tok as usize] = row as i32;
            }
        }

        let var = VarTable {
            index: ensembl,
            columns: vec![Column {
                name: SYMBOL_COL.to_string(),
                data: ColumnData::String(symbol),
            }],
        };

        Ok(Self {
            token_to_col: token_to_col.into(),
            var,
        })
    }
}

fn append_i64(batch: &arrow::record_batch::RecordBatch, name: &str, out: &mut Vec<i64>) -> Result<()> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| ScxError::MissingField(name.to_string()))?;
    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        out.extend((0..a.len()).map(|i| if a.is_null(i) { -1 } else { a.value(i) }));
        Ok(())
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        out.extend((0..a.len()).map(|i| if a.is_null(i) { -1 } else { a.value(i) as i64 }));
        Ok(())
    } else {
        Err(net_err(format!("`{name}` is not an integer column")))
    }
}

fn append_utf8(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    out: &mut Vec<String>,
) -> Result<()> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| ScxError::MissingField(name.to_string()))?;
    let a = col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| net_err(format!("`{name}` is not Utf8")))?;
    out.extend((0..a.len()).map(|i| {
        if a.is_null(i) {
            String::new()
        } else {
            a.value(i).to_string()
        }
    }));
    Ok(())
}
