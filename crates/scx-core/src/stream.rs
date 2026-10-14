use std::ops::Range;
use std::pin::Pin;

use async_trait::async_trait;
use futures::{stream, Stream, StreamExt};

use crate::dtype::DataType;
use crate::error::{Result, ScxError};
use crate::ir::{Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable, VarTable, Varm};

/// A boxed stream of row chunks, as every reader returns.
pub type ChunkStream<'a> = Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>>;

/// Rows `0..n_rows` as consecutive chunks of up to `chunk_size` rows, each
/// produced by `read` when the stream is polled. Every reader's X, layer and
/// obsp stream is this loop around a function that reads one row range.
pub(crate) fn row_chunks<'a, F>(n_rows: usize, chunk_size: usize, read: F) -> ChunkStream<'a>
where
    F: FnMut(Range<usize>) -> Result<MatrixChunk> + Send + 'a,
{
    row_range_chunks(n_rows, vec![0..n_rows], chunk_size, read)
}

/// The rows in `ranges` (normalized first), each range as consecutive chunks
/// of up to `chunk_size` rows produced by `read`; chunks never span ranges.
/// A range past `n_rows` makes the stream yield one error.
pub(crate) fn row_range_chunks<'a, F>(
    n_rows: usize,
    ranges: Vec<Range<usize>>,
    chunk_size: usize,
    read: F,
) -> ChunkStream<'a>
where
    F: FnMut(Range<usize>) -> Result<MatrixChunk> + Send + 'a,
{
    let ranges = normalize_ranges(ranges);
    if let Some(r) = ranges.iter().find(|r| r.end > n_rows) {
        let msg = format!(
            "row range {}..{} is beyond the {n_rows} rows",
            r.start, r.end
        );
        return Box::pin(stream::once(
            async move { Err(ScxError::InvalidFormat(msg)) },
        ));
    }
    let step = chunk_size.max(1);
    Box::pin(
        stream::iter(ranges.into_iter().flat_map(move |r| {
            (r.start..r.end)
                .step_by(step)
                .map(move |start| start..(start + step).min(r.end))
        }))
        .map(read),
    )
}

/// `ranges` sorted, with empty ones dropped and overlapping or touching ones
/// merged: what every row-range reader expects.
pub fn normalize_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.retain(|r| r.start < r.end);
    ranges.sort_by_key(|r| r.start);
    let mut out: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

/// The rows of `chunk` that fall in `ranges` (normalized), as one chunk per
/// intersecting range, each with its `row_offset` in the full matrix.
pub(crate) fn select_rows(chunk: &MatrixChunk, ranges: &[Range<usize>]) -> Vec<MatrixChunk> {
    let (lo, hi) = (chunk.row_offset, chunk.row_offset + chunk.nrows);
    ranges
        .iter()
        .filter(|r| r.start < hi && r.end > lo)
        .map(|r| {
            let local = r.start.max(lo) - lo..r.end.min(hi) - lo;
            let csr = &chunk.data;
            let nnz = csr.indptr[local.start] as usize..csr.indptr[local.end] as usize;
            let mut sub = crate::sparse::csr_chunk(
                &csr.indptr,
                local.clone(),
                csr.shape.1,
                csr.indices[nnz.clone()].to_vec(),
                csr.data.slice(nnz),
            );
            sub.row_offset = lo + local.start;
            sub
        })
        .collect()
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

    /// Stream only the rows in `ranges` (any order; overlaps are merged), as
    /// chunks of up to the reader's chunk size that never span two ranges.
    /// Each chunk's `row_offset` is its first row's index in the full matrix.
    ///
    /// Formats with a row pointer (h5ad, Zarr) read just those rows, so a
    /// filter on `obs` costs what its rows cost (see
    /// [`ObsTable::rows_where`](crate::ir::ObsTable::rows_where)). The
    /// default reads everything and keeps the requested rows.
    fn x_stream_rows(&mut self, ranges: Vec<Range<usize>>) -> ChunkStream<'_> {
        let ranges = normalize_ranges(ranges);
        let n_rows = self.shape().0;
        if let Some(r) = ranges.iter().find(|r| r.end > n_rows) {
            let msg = format!(
                "row range {}..{} is beyond the {n_rows} rows",
                r.start, r.end
            );
            return Box::pin(stream::once(
                async move { Err(ScxError::InvalidFormat(msg)) },
            ));
        }
        let last = ranges.last().map_or(0, |r| r.end);
        Box::pin(
            self.x_stream()
                // Nothing after the last requested row is needed.
                .take_while(move |c| {
                    futures::future::ready(c.as_ref().map_or(true, |c| c.row_offset < last))
                })
                .flat_map(move |c| {
                    stream::iter(match c {
                        Ok(c) => select_rows(&c, &ranges).into_iter().map(Ok).collect(),
                        Err(e) => vec![Err(e)],
                    })
                }),
        )
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::TypedVec;
    use crate::ir::{mask_to_ranges, Column, ColumnData, SparseMatrixCSR};

    #[test]
    fn ranges_are_sorted_merged_and_nonempty() {
        assert_eq!(
            normalize_ranges(vec![9..12, 3..5, 5..7, 20..20, 1..2, 10..11]),
            vec![1..2, 3..7, 9..12]
        );
        assert!(normalize_ranges(vec![]).is_empty());
    }

    #[test]
    fn mask_runs_become_ranges() {
        assert_eq!(
            mask_to_ranges(&[true, true, false, true, false, false, true]),
            vec![0..2, 3..4, 6..7]
        );
        assert!(mask_to_ranges(&[false, false]).is_empty());
        assert_eq!(mask_to_ranges(&[]), Vec::<Range<usize>>::new());
    }

    #[test]
    fn rows_where_compares_as_text_and_skips_na() {
        let obs = ObsTable {
            index: (0..4).map(|i| format!("c{i}")).collect(),
            columns: vec![
                Column::new(
                    "line",
                    ColumnData::Categorical {
                        codes: vec![0, 1, 0, 1],
                        levels: vec!["A549".into(), "MCF7".into()],
                    },
                ),
                Column::with_missing(
                    "day",
                    ColumnData::Int(vec![1, 3, 3, 0]),
                    [false, false, false, true],
                ),
                Column::new("ok", ColumnData::Bool(vec![true, false, true, true])),
            ],
        };
        assert_eq!(
            obs.rows_where("line", &["A549"]).unwrap(),
            [true, false, true, false]
        );
        assert_eq!(
            obs.rows_where("day", &["3", "0"]).unwrap(),
            [false, true, true, false]
        );
        assert_eq!(
            obs.rows_where("ok", &["True"]).unwrap(),
            [true, false, true, true]
        );
        assert!(obs.rows_where("nope", &["x"]).is_err());
        let mask: Vec<bool> = obs
            .rows_where("line", &["A549"])
            .unwrap()
            .iter()
            .zip(obs.rows_where("ok", &["True"]).unwrap())
            .map(|(a, b)| *a && b)
            .collect();
        assert_eq!(mask_to_ranges(&mask), vec![0..1, 2..3]);
    }

    /// A 4-row chunk at offset 10: rows [10,11,12,13] with 1,0,2,1 nnz.
    fn chunk() -> MatrixChunk {
        MatrixChunk {
            row_offset: 10,
            nrows: 4,
            data: SparseMatrixCSR {
                shape: (4, 3),
                indptr: vec![0, 1, 1, 3, 4],
                indices: vec![2, 0, 1, 1],
                data: TypedVec::F32(vec![1.0, 2.0, 3.0, 4.0]),
            },
        }
    }

    #[test]
    fn select_rows_keeps_offsets_and_values() {
        let got = select_rows(&chunk(), &[0..11, 12..13, 13..20, 30..40]);
        let summary: Vec<(usize, usize, Vec<u64>, Vec<f64>)> = got
            .iter()
            .map(|c| {
                (
                    c.row_offset,
                    c.nrows,
                    c.data.indptr.clone(),
                    c.data.data.to_f64(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                (10, 1, vec![0, 1], vec![1.0]),
                (12, 1, vec![0, 2], vec![2.0, 3.0]),
                (13, 1, vec![0, 1], vec![4.0]),
            ]
        );
        assert!(select_rows(&chunk(), &[0..10, 14..20]).is_empty());
    }
}
