//! A dataset restricted to the cells whose `obs` matches a filter, as
//! `scx convert --where` writes it.
//!
//! [`SubsetReader`] wraps any reader and is itself a [`DatasetReader`], so a
//! conversion loop writes the subset without knowing it is one. Every
//! cell-axis slot is subset: X, layers and raw.X by row, obs and obsm by
//! row, and obsp (cell × cell) by row and column, renumbered. var, varm and
//! uns pass through unchanged.

use std::ops::Range;

use async_trait::async_trait;
use futures::{future::ready, stream, StreamExt};

use crate::concat::{gather, remap_chunk, VarMap};
use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    mask_to_ranges, Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
    SparseMatrixMeta, UnsTable, VarTable, Varm,
};
use crate::stream::{error_stream, require_cell_major, ChunkStream, DatasetReader};

/// `(column, value)` pairs as filters: grouped by column in first-seen
/// order, so one column's values OR together and the columns AND. How
/// `scx convert --where col=value` (repeated) and R's flattened
/// `where = list(...)` arrive.
pub fn group_filters(
    pairs: impl IntoIterator<Item = (String, String)>,
) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for (col, value) in pairs {
        match out.iter_mut().find(|(c, _)| *c == col) {
            Some((_, values)) if values.contains(&value) => {}
            Some((_, values)) => values.push(value),
            None => out.push((col, vec![value])),
        }
    }
    out
}

/// The cells of `inner` whose obs matches every `(column, values)` filter,
/// matched as [`ObsTable::rows_matching`] does (as text, NA never matches).
///
/// X is read through [`DatasetReader::x_stream_rows`], so formats that read
/// row ranges (h5ad, Zarr, dgCMatrix h5seurat, 10x HDF5) read only the kept
/// rows; others pass over every row and drop the rest. Layers, raw.X and
/// obsp stream whole and drop rows as they pass. Chunks hold at most
/// `chunk_size` rows.
///
/// [`x_indptr`](DatasetReader::x_indptr) is empty, and the metas from
/// [`obsp_metas`](DatasetReader::obsp_metas) carry an empty `indptr`: the
/// column restriction makes their row lengths known only once streamed.
/// Writers build their own `indptr` from the chunks.
pub struct SubsetReader {
    inner: Box<dyn DatasetReader>,
    /// Each source row's row in the subset (`None`: dropped). Also the
    /// column map for obsp.
    rows: VarMap,
    ranges: Vec<Range<usize>>,
    n_obs: usize,
    chunk_size: usize,
    obs: ObsTable,
    /// The source's metas, filled as the subset's are asked for.
    layers: Vec<SparseMatrixMeta>,
    obsp: Vec<SparseMatrixMeta>,
    raw: Option<SparseMatrixMeta>,
}

impl SubsetReader {
    /// Reads `inner`'s obs to find the matching cells. Fails on a filter
    /// column obs doesn't have, and on gene-major (CSC) X.
    pub async fn new(
        mut inner: Box<dyn DatasetReader>,
        filters: &[(String, Vec<String>)],
        chunk_size: usize,
    ) -> Result<Self> {
        require_cell_major(&*inner)?;
        let src_n = inner.shape().0;
        let obs = inner.obs().await?;
        let mask = obs.rows_matching(src_n, filters)?;
        let mut next = 0u32;
        let to_out: Vec<Option<u32>> = mask
            .iter()
            .map(|&keep| {
                keep.then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let kept: Vec<usize> = (0..src_n).filter(|&i| mask[i]).collect();
        let obs = ObsTable {
            index: if obs.index.len() == src_n {
                kept.iter().map(|&i| obs.index[i].clone()).collect()
            } else {
                obs.index
            },
            columns: obs.columns.iter().map(|c| take_column(c, &kept)).collect(),
        };
        Ok(Self {
            inner,
            rows: VarMap {
                identity: kept.len() == src_n,
                // Kept rows stay in order, so obsp column ids stay sorted.
                needs_sort: false,
                to_out,
            },
            ranges: mask_to_ranges(&mask),
            n_obs: kept.len(),
            chunk_size: chunk_size.max(1),
            obs,
            layers: Vec::new(),
            obsp: Vec::new(),
            raw: None,
        })
    }

    /// `meta` with the subset's row count and, from the source's `indptr`
    /// (when it has one), the subset's.
    fn sub_meta(&self, meta: &SparseMatrixMeta) -> SparseMatrixMeta {
        let ip = &meta.indptr;
        let indptr = if ip.len() == self.rows.to_out.len() + 1 {
            let mut out = vec![0u64];
            for (i, r) in self.rows.to_out.iter().enumerate() {
                if r.is_some() {
                    out.push(out.last().unwrap() + ip[i + 1] - ip[i]);
                }
            }
            out
        } else {
            Vec::new()
        };
        SparseMatrixMeta {
            name: meta.name.clone(),
            shape: (self.n_obs, meta.shape.1),
            indptr,
        }
    }
}

/// The kept rows of `chunk`, renumbered, or `None` when it keeps none.
fn take_rows(chunk: MatrixChunk, rows: &VarMap) -> Option<MatrixChunk> {
    let to_out = &rows.to_out[chunk.row_offset..chunk.row_offset + chunk.nrows];
    let keep: Vec<usize> = (0..chunk.nrows).filter(|&r| to_out[r].is_some()).collect();
    let row_offset = to_out[*keep.first()?]? as usize;
    if keep.len() == chunk.nrows {
        return Some(MatrixChunk {
            row_offset,
            ..chunk
        });
    }
    let csr = &chunk.data;
    let mut indptr = Vec::with_capacity(keep.len() + 1);
    indptr.push(0u64);
    let mut pos: Vec<usize> = Vec::new();
    for &r in &keep {
        pos.extend(csr.indptr[r] as usize..csr.indptr[r + 1] as usize);
        indptr.push(pos.len() as u64);
    }
    Some(MatrixChunk {
        row_offset,
        nrows: keep.len(),
        data: SparseMatrixCSR {
            shape: (keep.len(), csr.shape.1),
            indptr,
            indices: pos.iter().map(|&p| csr.indices[p]).collect(),
            data: gather(&csr.data, &pos),
        },
    })
}

/// `s` with the kept rows of each chunk, renumbered; `cols` also restricts
/// and renumbers columns (obsp).
fn keep_rows<'a>(s: ChunkStream<'a>, rows: &'a VarMap, cols: Option<usize>) -> ChunkStream<'a> {
    Box::pin(s.filter_map(move |c| {
        ready(match c {
            Ok(c) => take_rows(c, rows).map(|c| {
                Ok(match cols {
                    Some(n) => remap_chunk(c, rows, n),
                    None => c,
                })
            }),
            Err(e) => Some(Err(e)),
        })
    }))
}

/// Consecutive chunks of `s` joined into chunks of up to `rows` rows, so a
/// filter that keeps scattered rows doesn't write one tiny chunk per run.
fn rebatch(s: ChunkStream<'_>, rows: usize) -> ChunkStream<'_> {
    Box::pin(stream::unfold(
        (s, None::<MatrixChunk>),
        move |(mut s, pending)| async move {
            let mut parts: Vec<MatrixChunk> = pending.into_iter().collect();
            let mut n: usize = parts.iter().map(|c| c.nrows).sum();
            let mut next = None;
            while let Some(c) = s.next().await {
                match c {
                    Err(e) => return Some((Err(e), (s, None))),
                    Ok(c) if n > 0 && n + c.nrows > rows => {
                        next = Some(c);
                        break;
                    }
                    Ok(c) => {
                        n += c.nrows;
                        parts.push(c);
                    }
                }
            }
            if parts.is_empty() {
                return None;
            }
            Some((Ok(join_rows(parts)), (s, next)))
        },
    ))
}

/// Row-consecutive chunks as one.
fn join_rows(mut parts: Vec<MatrixChunk>) -> MatrixChunk {
    if parts.len() == 1 {
        return parts.pop().unwrap();
    }
    let nrows: usize = parts.iter().map(|c| c.nrows).sum();
    let mut indptr = Vec::with_capacity(nrows + 1);
    indptr.push(0u64);
    let mut indices = Vec::new();
    for c in &parts {
        let base = *indptr.last().unwrap();
        indptr.extend(c.data.indptr[1..].iter().map(|&p| base + p));
        indices.extend_from_slice(&c.data.indices);
    }
    macro_rules! cat {
        ($variant:path) => {
            $variant(
                parts
                    .iter()
                    .flat_map(|c| match &c.data.data {
                        $variant(v) => v.iter().copied(),
                        _ => unreachable!("chunks of one stream share a dtype"),
                    })
                    .collect(),
            )
        };
    }
    let data = match parts[0].data.data.dtype() {
        DataType::F32 => cat!(TypedVec::F32),
        DataType::F64 => cat!(TypedVec::F64),
        DataType::I32 => cat!(TypedVec::I32),
        DataType::U32 => cat!(TypedVec::U32),
    };
    MatrixChunk {
        row_offset: parts[0].row_offset,
        nrows,
        data: SparseMatrixCSR {
            shape: (nrows, parts[0].data.shape.1),
            indptr,
            indices,
            data,
        },
    }
}

/// Rows `idx` of `c`. Categories are kept whole, as anndata subsetting does.
fn take_column(c: &Column, idx: &[usize]) -> Column {
    fn t<T: Clone>(v: &[T], idx: &[usize]) -> Vec<T> {
        idx.iter().map(|&i| v[i].clone()).collect()
    }
    let data = match &c.data {
        ColumnData::Int(v) => ColumnData::Int(t(v, idx)),
        ColumnData::Int64(v) => ColumnData::Int64(t(v, idx)),
        ColumnData::Float(v) => ColumnData::Float(t(v, idx)),
        ColumnData::String(v) => ColumnData::String(t(v, idx)),
        ColumnData::Bool(v) => ColumnData::Bool(t(v, idx)),
        ColumnData::Categorical {
            codes,
            levels,
            ordered,
        } => ColumnData::Categorical {
            codes: t(codes, idx),
            levels: levels.clone(),
            ordered: *ordered,
        },
    };
    Column {
        name: c.name.clone(),
        data,
        mask: c.mask.as_ref().map(|m| t(m, idx)),
    }
}

#[async_trait]
impl DatasetReader for SubsetReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.inner.shape().1)
    }

    fn dtype(&self) -> DataType {
        self.inner.dtype()
    }

    fn source_hash(&self) -> Option<String> {
        self.inner.source_hash()
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        Ok(self.obs.clone())
    }

    async fn var(&mut self) -> Result<VarTable> {
        self.inner.var().await
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        let mut obsm = self.inner.obsm().await?;
        let src_n = self.rows.to_out.len();
        for m in obsm.map.values_mut() {
            let k = m.shape.1;
            if m.shape.0 != src_n {
                continue;
            }
            m.data = (0..src_n)
                .filter(|&i| self.rows.to_out[i].is_some())
                .flat_map(|i| m.data[i * k..(i + 1) * k].iter().copied())
                .collect();
            m.shape.0 = self.n_obs;
        }
        Ok(obsm)
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        self.inner.uns().await
    }

    async fn varm(&mut self) -> Result<Varm> {
        self.inner.varm().await
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        self.layers = self.inner.layer_metas().await?;
        Ok(self.layers.iter().map(|m| self.sub_meta(m)).collect())
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        self.obsp = self.inner.obsp_metas().await?;
        Ok(self
            .obsp
            .iter()
            .map(|m| SparseMatrixMeta {
                name: m.name.clone(),
                shape: (self.n_obs, self.n_obs),
                indptr: Vec::new(),
            })
            .collect())
    }

    async fn raw_meta(&mut self) -> Result<Option<SparseMatrixMeta>> {
        self.raw = self.inner.raw_meta().await?;
        Ok(self.raw.as_ref().map(|m| self.sub_meta(m)))
    }

    fn raw_stream<'a>(&'a self, _meta: &'a SparseMatrixMeta, chunk_size: usize) -> ChunkStream<'a> {
        match &self.raw {
            Some(src) => keep_rows(self.inner.raw_stream(src, chunk_size), &self.rows, None),
            None => error_stream("raw_stream called before raw_meta".into()),
        }
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> ChunkStream<'a> {
        match self.layers.iter().find(|m| m.name == meta.name) {
            Some(src) => keep_rows(self.inner.layer_stream(src, chunk_size), &self.rows, None),
            None => error_stream(format!("no layer '{}' in the source", meta.name)),
        }
    }

    fn obsp_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk_size: usize) -> ChunkStream<'a> {
        match self.obsp.iter().find(|m| m.name == meta.name) {
            Some(src) => keep_rows(
                self.inner.obsp_stream(src, chunk_size),
                &self.rows,
                Some(self.n_obs),
            ),
            None => error_stream(format!("no obsp '{}' in the source", meta.name)),
        }
    }

    fn x_stream(&mut self) -> ChunkStream<'_> {
        let to_out = &self.rows.to_out;
        let s = self.inner.x_stream_rows(self.ranges.clone()).map(move |c| {
            let c = c?;
            match to_out.get(c.row_offset).copied().flatten() {
                Some(row) => Ok(MatrixChunk {
                    row_offset: row as usize,
                    ..c
                }),
                None => Err(ScxError::InvalidFormat(format!(
                    "x_stream_rows returned row {}, outside the requested ranges",
                    c.row_offset
                ))),
            }
        });
        rebatch(Box::pin(s), self.chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny(name: &str) -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    fn w(filters: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> {
        filters
            .iter()
            .map(|(c, v)| (c.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    async fn open(name: &str, chunk: usize) -> Box<dyn DatasetReader> {
        crate::open(&tiny(name), &crate::OpenOptions::new(chunk))
            .await
            .unwrap()
    }

    /// Rows of a stream as dense (row, values), checking row order.
    async fn dense(s: ChunkStream<'_>, n_cols: usize, max_rows: usize) -> Vec<Vec<f64>> {
        let chunks: Vec<MatrixChunk> = s.map(|c| c.unwrap()).collect().await;
        let mut out = Vec::new();
        for c in chunks {
            assert!(c.nrows <= max_rows, "chunk of {} rows", c.nrows);
            assert_eq!(c.row_offset, out.len(), "chunks arrive in row order");
            let v = c.data.data.to_f64();
            for r in 0..c.nrows {
                let mut row = vec![0.0; n_cols];
                for k in c.data.indptr[r] as usize..c.data.indptr[r + 1] as usize {
                    row[c.data.indices[k] as usize] = v[k];
                }
                out.push(row);
            }
        }
        out
    }

    fn pick<T: Clone>(rows: &[T], idx: &[usize]) -> Vec<T> {
        idx.iter().map(|&i| rows[i].clone()).collect()
    }

    /// Every cell-axis slot of the subset equals the same rows of the full
    /// dataset; obsp is restricted to the kept cells on both axes.
    #[tokio::test]
    async fn subset_equals_the_full_dataset_subset() {
        // cell_type: B T NaN B NK T. "B or T" keeps 0,1,3,5.
        let keep = [0, 1, 3, 5];
        for chunk in [1, 2, 10] {
            let mut full = open("tiny.h5ad", chunk).await;
            let mut sub = SubsetReader::new(
                open("tiny.h5ad", chunk).await,
                &w(&[("cell_type", &["B", "T"])]),
                chunk,
            )
            .await
            .unwrap();
            assert_eq!(sub.shape(), (4, 5));

            let (fo, so) = (full.obs().await.unwrap(), sub.obs().await.unwrap());
            assert_eq!(so.index, pick(&fo.index, &keep));
            for (f, s) in fo.columns.iter().zip(&so.columns) {
                for (j, &i) in keep.iter().enumerate() {
                    assert_eq!(f.is_na(i), s.is_na(j), "{} NA", f.name);
                    if !f.is_na(i) {
                        assert_eq!(f.data.value_str(i), s.data.value_str(j), "{}", f.name);
                    }
                }
            }

            let x_full = dense(full.x_stream(), 5, chunk).await;
            assert_eq!(dense(sub.x_stream(), 5, chunk).await, pick(&x_full, &keep));

            let fl = full.layer_metas().await.unwrap();
            let sl = sub.layer_metas().await.unwrap();
            assert_eq!(sl[0].shape, (4, 5));
            let lf = dense(full.layer_stream(&fl[0], chunk), 5, chunk).await;
            let ls = dense(sub.layer_stream(&sl[0], chunk), 5, chunk).await;
            assert_eq!(ls, pick(&lf, &keep));
            let nnz: Vec<u64> = keep
                .iter()
                .map(|&i| fl[0].indptr[i + 1] - fl[0].indptr[i])
                .collect();
            let mut ip = vec![0u64];
            nnz.iter().for_each(|n| ip.push(ip.last().unwrap() + n));
            assert_eq!(sl[0].indptr, ip, "layer indptr subset");

            let fp = full.obsp_metas().await.unwrap();
            let sp = sub.obsp_metas().await.unwrap();
            assert_eq!(sp[0].shape, (4, 4));
            let pf = dense(full.obsp_stream(&fp[0], chunk), 6, chunk).await;
            let ps = dense(sub.obsp_stream(&sp[0], chunk), 4, chunk).await;
            let want: Vec<Vec<f64>> = keep.iter().map(|&i| pick(&pf[i], &keep)).collect();
            assert_eq!(ps, want);
            // (0,3) -> (0,2) and (3,0) -> (2,0); (2,5)/(5,2) drop with cell 2.
            assert_eq!(ps[0][2], 1.0);
            assert_eq!(ps[2][0], 0.5);

            let fm = full.obsm().await.unwrap();
            let sm = sub.obsm().await.unwrap();
            let (f, s) = (&fm.map["X_pca"], &sm.map["X_pca"]);
            let k = f.shape.1;
            assert_eq!(s.shape, (4, k));
            let want: Vec<f64> = keep
                .iter()
                .flat_map(|&i| f.data[i * k..(i + 1) * k].to_vec())
                .collect();
            assert_eq!(s.data, want);

            assert_eq!(
                sub.var().await.unwrap().index,
                full.var().await.unwrap().index
            );
        }
    }

    #[tokio::test]
    async fn raw_rows_are_subset() {
        let mut full = open("tiny_raw.h5ad", 2).await;
        // tiny_raw has no obs columns; filter on another file's obs is not
        // possible, so keep rows by building the subset over tiny.h5ad's
        // cell_type mask applied to tiny_raw: same 6 cells.
        let fr = full.raw_meta().await.unwrap().unwrap();
        let rf = dense(full.raw_stream(&fr, 2), 5, 2).await;
        let mut sub = SubsetReader::new(open("tiny_raw.h5ad", 2).await, &[], 2)
            .await
            .unwrap();
        // Drop rows 1 and 4 by hand to exercise the raw path.
        sub.rows.to_out = vec![Some(0), None, Some(1), Some(2), None, Some(3)];
        sub.n_obs = 4;
        let sr = sub.raw_meta().await.unwrap().unwrap();
        assert_eq!(sr.shape, (4, 5));
        assert_eq!(
            dense(sub.raw_stream(&sr, 2), 5, 2).await,
            pick(&rf, &[0, 2, 3, 5])
        );
    }

    async fn kept(filters: &[(&str, &[&str])]) -> Result<Vec<String>> {
        let mut r = SubsetReader::new(open("tiny.h5ad", 2).await, &w(filters), 2).await?;
        Ok(r.obs().await?.index)
    }

    #[tokio::test]
    async fn filters_match_as_text_and_and_together() {
        let c = |v: &[usize]| v.iter().map(|i| format!("cell{i}")).collect::<Vec<_>>();
        // Several values OR; categorical NaN never matches.
        assert_eq!(
            kept(&[("cell_type", &["B", "NK"])]).await.unwrap(),
            c(&[0, 3, 4])
        );
        // Columns AND: cell_type B and donor d1.
        assert_eq!(
            kept(&[("cell_type", &["B"]), ("donor", &["d1"])])
                .await
                .unwrap(),
            c(&[3])
        );
        // Booleans as True/False; nullable bool NA never matches.
        assert_eq!(kept(&[("is_doublet", &["True"])]).await.unwrap(), c(&[2]));
        assert_eq!(kept(&[("flag", &["False"])]).await.unwrap(), c(&[2, 5]));
        // Nullable int compared as text.
        assert_eq!(kept(&[("batch", &["2"])]).await.unwrap(), c(&[2, 3]));
        // No filter keeps everything.
        assert_eq!(kept(&[]).await.unwrap(), c(&[0, 1, 2, 3, 4, 5]));
        // No match: zero cells, and X streams nothing.
        let mut none = SubsetReader::new(
            open("tiny.h5ad", 2).await,
            &w(&[("cell_type", &["nope"])]),
            2,
        )
        .await
        .unwrap();
        assert_eq!(none.shape(), (0, 5));
        assert!(dense(none.x_stream(), 5, 2).await.is_empty());
        // A column obs doesn't have is an error naming the columns it has.
        let err = kept(&[("celltype", &["B"])]).await.unwrap_err().to_string();
        assert!(
            err.contains("'celltype'") && err.contains("cell_type"),
            "{err}"
        );
    }

    #[test]
    fn rebatch_joins_runs_up_to_the_chunk_size() {
        let one = |row: usize, v: f32| MatrixChunk {
            row_offset: row,
            nrows: 1,
            data: SparseMatrixCSR {
                shape: (1, 2),
                indptr: vec![0, 1],
                indices: vec![1],
                data: TypedVec::F32(vec![v]),
            },
        };
        let s: ChunkStream<'_> = Box::pin(stream::iter((0..5).map(|i| Ok(one(i, i as f32)))));
        let got: Vec<MatrixChunk> =
            futures::executor::block_on(rebatch(s, 2).map(|c| c.unwrap()).collect());
        let shape: Vec<(usize, usize)> = got.iter().map(|c| (c.row_offset, c.nrows)).collect();
        assert_eq!(shape, [(0, 2), (2, 2), (4, 1)]);
        assert_eq!(got[1].data.indptr, [0, 1, 2]);
        assert_eq!(got[1].data.data.to_f64(), [2.0, 3.0]);
    }
}
