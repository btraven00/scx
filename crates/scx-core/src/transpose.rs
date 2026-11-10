//! Out-of-core transpose of X to gene-major (CSC) storage.
//!
//! Most files store cells as rows. Gene-wise consumers (one GLM per gene)
//! want each gene's cells contiguous, and transposing in memory costs about
//! twice the matrix. [`write_x_csc`] does it in passes over X instead, with
//! the memory it holds bounded by a budget.

use std::ops::Range;

use futures::StreamExt;

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::h5ad::writer::cast;
use crate::ir::SparseMatrixCSR;
use crate::stream::{DatasetReader, DatasetWriter};

/// The default memory budget for transposing X, as `scx convert
/// --transpose-mem`'s default (4 GB); `scx diff` uses it as is.
pub const DEFAULT_TRANSPOSE_MEM: usize = 4_000_000_000;

/// Copy gene-major (CSC) X to `writer` by column, as stored: one by-gene
/// pass, holding one reader chunk of genes at a time. Returns the number of
/// chunks.
pub async fn copy_x_csc(
    reader: &mut dyn DatasetReader,
    writer: &mut dyn DatasetWriter,
) -> Result<usize> {
    let (n_obs, n_vars) = reader.shape();
    let dtype = reader.dtype();
    let mut n = 0;
    let mut stream = reader.x_stream_cols(vec![0..n_vars]);
    while let Some(chunk) = stream.next().await {
        writer.write_x_columns(&chunk?.data).await?;
        n += 1;
    }
    if n == 0 {
        // No genes: still mark X as CSC.
        let empty = SparseMatrixCSR {
            shape: (0, n_obs),
            indptr: vec![0],
            indices: Vec::new(),
            data: TypedVec::empty(dtype),
        };
        writer.write_x_columns(&empty).await?;
    }
    Ok(n)
}

/// Bytes [`x_cell_major`] holds for the reader's gene-major X: its nonzeros
/// (cell id + value) twice, as read and as transposed. 0 for cell-major X.
pub fn cell_major_bytes(reader: &dyn DatasetReader) -> u64 {
    let nnz = reader.x_col_ptr().last().copied().unwrap_or(0);
    2 * nnz * (4 + reader.dtype().size() as u64)
}

/// Gene-major (CSC) X read whole and transposed to cell-major CSR, for
/// writers that take X by cell. Holds [`cell_major_bytes`]; callers check it
/// against a budget first.
pub async fn x_cell_major<R: DatasetReader + ?Sized>(reader: &mut R) -> Result<SparseMatrixCSR> {
    let (n_obs, n_vars) = reader.shape();
    let col_ptr = reader.x_col_ptr().to_vec();
    if col_ptr.is_empty() {
        return Err(ScxError::InvalidFormat("X is not gene-major".into()));
    }
    let nnz = *col_ptr.last().unwrap() as usize;
    let mut cells = Vec::with_capacity(nnz);
    let mut data = TypedVec::empty(reader.dtype());
    let mut stream = reader.x_stream_cols(vec![0..n_vars]);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        cells.extend_from_slice(&chunk.data.indices);
        data.append(&chunk.data.data);
    }
    drop(stream);
    if let Some(&c) = cells.iter().find(|&&c| c as usize >= n_obs) {
        return Err(ScxError::InvalidFormat(format!(
            "X: cell index {c} out of range for {n_obs} cells"
        )));
    }
    if cells.len() != nnz {
        return Err(ScxError::InvalidFormat(format!(
            "X: read {} nonzeros by gene, column pointer says {nnz}",
            cells.len()
        )));
    }
    Ok(crate::sparse::csc_to_csr(
        (n_obs, n_vars),
        &col_ptr,
        &cells,
        &data,
    ))
}

/// Gene-major X as row chunks of `chunk` cells, holding about `mem_bytes`.
/// When [`cell_major_bytes`] fits, X is read once, through [`x_cell_major`].
/// Otherwise it is read once to count each cell's nonzeros and then once per
/// block of consecutive cells whose nonzeros fit: [`write_x_csc`]'s passes
/// with the axes swapped. One reader chunk is held on top of the budget.
pub fn x_cell_major_stream<'r>(
    reader: &'r mut (dyn DatasetReader + Send),
    mem_bytes: usize,
    chunk: usize,
) -> crate::stream::ChunkStream<'r> {
    use futures::stream;
    let ((n_obs, n_vars), dtype, chunk) = (reader.shape(), reader.dtype(), chunk.max(1));
    let by_rows = move |start: usize, m: SparseMatrixCSR| {
        stream::iter((0..m.shape.0).step_by(chunk).map(move |s| {
            let mut c = crate::sparse::csr_rows(&m, s..(s + chunk).min(m.shape.0));
            c.row_offset += start;
            Ok(c)
        }))
    };
    if cell_major_bytes(reader) <= mem_bytes as u64 {
        return Box::pin(
            stream::once(x_cell_major(reader)).flat_map(move |res| match res {
                Ok(m) => by_rows(0, m).left_stream(),
                Err(e) => stream::iter([Err(e)]).right_stream(),
            }),
        );
    }
    // (reader, cell pointer, the blocks of cells still to read)
    type Plan = (Vec<u64>, std::vec::IntoIter<Range<usize>>);
    let next_block = move |(reader, plan): (&'r mut (dyn DatasetReader + Send), Option<Plan>)| async move {
        let (row_ptr, mut blocks) = match plan {
            Some(p) => p,
            None => match count_ptr(reader.x_stream_cols(vec![0..n_vars]), n_obs).await {
                Ok(ptr) => {
                    let max_nnz = (mem_bytes / (4 + dtype.size())).max(1) as u64;
                    let blocks = column_blocks(&ptr, max_nnz).into_iter();
                    (ptr, blocks)
                }
                Err(e) => return Some((Err(e), None)),
            },
        };
        let cells = blocks.next()?;
        let block = read_columns(
            reader.x_stream_cols(vec![0..n_vars]),
            &row_ptr,
            cells.clone(),
            n_vars,
            dtype,
        )
        .await;
        let state = block.is_ok().then_some((reader, Some((row_ptr, blocks))));
        Some((block.map(|m| (cells.start, m)), state))
    };
    Box::pin(
        stream::unfold(Some((reader, None)), move |st| async move {
            next_block(st?).await
        })
        .flat_map(move |res| match res {
            Ok((start, m)) => by_rows(start, m).left_stream(),
            Err(e) => stream::iter([Err(e)]).right_stream(),
        }),
    )
}

/// Write the reader's X to `writer` by column (CSC), holding about
/// `mem_bytes` of transposed nonzeros at a time. Returns the number of
/// column blocks.
///
/// X is streamed `1 + blocks` times: once to count each column's nonzeros,
/// then once per block of consecutive columns whose nonzeros fit the budget.
/// Each pass keeps only its block's columns and drops each value straight
/// into place, so cells come out ascending within each column. A column with
/// more nonzeros than the budget gets a block of its own. One reader chunk is
/// held on top of the budget. When X fits the budget, that's one block and
/// two reads.
pub async fn write_x_csc(
    reader: &mut dyn DatasetReader,
    writer: &mut dyn DatasetWriter,
    mem_bytes: usize,
) -> Result<usize> {
    let (n_obs, n_vars) = reader.shape();
    let dtype = reader.dtype();

    let col_ptr = count_ptr(reader.x_stream(), n_vars).await?;

    // Per nonzero: the u32 cell id and the value, plus the i32 copy of the
    // ids the h5ad writer makes.
    let max_nnz = (mem_bytes / (4 + dtype.size() + 4)).max(1) as u64;
    let blocks = column_blocks(&col_ptr, max_nnz);
    for cols in &blocks {
        let block = read_columns(reader.x_stream(), &col_ptr, cols.clone(), n_obs, dtype).await?;
        writer.write_x_columns(&block).await?;
    }
    if blocks.is_empty() {
        // No columns: still mark X as CSC.
        let empty = SparseMatrixCSR {
            shape: (0, n_obs),
            indptr: vec![0],
            indices: Vec::new(),
            data: TypedVec::empty(dtype),
        };
        writer.write_x_columns(&empty).await?;
    }
    Ok(blocks.len())
}

/// Consecutive column ranges, each holding at most `max_nnz` nonzeros
/// unless a single column alone exceeds it.
fn column_blocks(col_ptr: &[u64], max_nnz: u64) -> Vec<Range<usize>> {
    let n = col_ptr.len() - 1;
    let mut blocks = Vec::new();
    let mut start = 0;
    for end in start + 1..=n {
        if end - 1 > start && col_ptr[end] - col_ptr[start] > max_nnz {
            blocks.push(start..end - 1);
            start = end - 1;
        }
    }
    if start < n {
        blocks.push(start..n);
    }
    blocks
}

/// The pointer array of `stream`'s minor axis: entry `i + 1` counts the
/// nonzeros with index `i`, cumulated. One pass over the stream.
async fn count_ptr(mut stream: crate::stream::ChunkStream<'_>, n: usize) -> Result<Vec<u64>> {
    let mut ptr = vec![0u64; n + 1];
    while let Some(chunk) = stream.next().await {
        for &i in &chunk?.data.indices {
            let slot = ptr.get_mut(i as usize + 1).ok_or_else(|| {
                ScxError::InvalidFormat(format!("X: index {i} out of range for {n}"))
            })?;
            *slot += 1;
        }
    }
    for j in 0..n {
        ptr[j + 1] += ptr[j];
    }
    Ok(ptr)
}

/// Minor-axis entries `cols` of `stream` (a pass over X by cell or by gene)
/// as the CSR of their transpose: `cols` × the stream's rows (`n_obs` of
/// them), from one pass. By cell that is a block of genes of the CSC; by gene,
/// a block of cells of the CSR.
async fn read_columns(
    mut stream: crate::stream::ChunkStream<'_>,
    col_ptr: &[u64],
    cols: Range<usize>,
    n_obs: usize,
    dtype: DataType,
) -> Result<SparseMatrixCSR> {
    let base = col_ptr[cols.start];
    let indptr: Vec<u64> = col_ptr[cols.start..=cols.end]
        .iter()
        .map(|&p| p - base)
        .collect();
    let nnz = *indptr.last().unwrap() as usize;
    // Each column's next free slot.
    let mut next: Vec<u64> = indptr[..cols.len()].to_vec();
    let mut indices = vec![0u32; nnz];
    let mut data = match dtype {
        DataType::F32 => TypedVec::F32(vec![0.0; nnz]),
        DataType::F64 => TypedVec::F64(vec![0.0; nnz]),
        DataType::I32 => TypedVec::I32(vec![0; nnz]),
        DataType::U32 => TypedVec::U32(vec![0; nnz]),
    };

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let csr = &chunk.data;
        // A reader's chunks needn't have its declared dtype (BPCells h5seurat
        // declares f64 and streams u32): bring them to the block's.
        let converted;
        let values = if csr.data.dtype() == dtype {
            &csr.data
        } else {
            converted = match dtype {
                DataType::F32 => TypedVec::F32(cast(&csr.data, |x| x as f32)),
                DataType::F64 => TypedVec::F64(cast(&csr.data, |x| x)),
                DataType::I32 => TypedVec::I32(cast(&csr.data, |x| x as i32)),
                DataType::U32 => TypedVec::U32(cast(&csr.data, |x| x as u32)),
            };
            &converted
        };
        let mut place = |dst: &mut dyn FnMut(usize, usize)| {
            for r in 0..chunk.nrows {
                for k in csr.indptr[r] as usize..csr.indptr[r + 1] as usize {
                    let c = csr.indices[k] as usize;
                    if cols.contains(&c) {
                        let slot = &mut next[c - cols.start];
                        indices[*slot as usize] = (chunk.row_offset + r) as u32;
                        dst(k, *slot as usize);
                        *slot += 1;
                    }
                }
            }
        };
        match (&mut data, values) {
            (TypedVec::F32(d), TypedVec::F32(s)) => place(&mut |k, i| d[i] = s[k]),
            (TypedVec::F64(d), TypedVec::F64(s)) => place(&mut |k, i| d[i] = s[k]),
            (TypedVec::I32(d), TypedVec::I32(s)) => place(&mut |k, i| d[i] = s[k]),
            (TypedVec::U32(d), TypedVec::U32(s)) => place(&mut |k, i| d[i] = s[k]),
            _ => unreachable!("values were converted to the block's dtype"),
        }
    }

    // Every pass must see the nonzeros the counting pass saw.
    if next.iter().zip(&indptr[1..]).any(|(n, end)| n != end) {
        return Err(ScxError::InvalidFormat(format!(
            "X changed between passes while transposing columns {}..{}",
            cols.start, cols.end
        )));
    }
    Ok(SparseMatrixCSR {
        shape: (cols.len(), n_obs),
        indptr,
        indices,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::write::{write_h5ad_csr, H5AdOptions};
    use crate::h5_str::read_str_attr;
    use crate::h5ad::{H5AdReader, H5AdWriter};
    use crate::ir::{MatrixChunk, ObsTable, VarTable};
    use sprs::TriMatI;

    #[test]
    fn blocks_respect_budget_and_keep_oversized_columns_whole() {
        // Columns with 2, 5, 1, 1, 0 nonzeros.
        let col_ptr = [0, 2, 7, 8, 9, 9];
        assert_eq!(column_blocks(&col_ptr, 3), vec![0..1, 1..2, 2..5]);
        assert_eq!(column_blocks(&col_ptr, 100), vec![0..5]);
        assert!(column_blocks(&[0], 3).is_empty());
    }

    /// A tiny budget forces several passes; the file must hold exactly the
    /// CSC of the input, as anndata reads `csc_matrix`.
    #[tokio::test]
    async fn multi_pass_transpose_writes_the_csc_of_x() {
        let (n_obs, n_vars) = (7, 5);
        let mut tri = TriMatI::<f32, u32>::new((n_obs, n_vars));
        for (i, j, v) in [
            (0, 1, 1.),
            (0, 4, 2.),
            (2, 1, 3.),
            (3, 0, 4.),
            (5, 1, 5.),
            (6, 3, 6.),
        ] {
            tri.add_triplet(i, j, v);
        }
        let x = tri.to_csr::<u32>();
        let tmp = tempfile::tempdir().unwrap();
        let (src, dst) = (tmp.path().join("in.h5ad"), tmp.path().join("out.h5ad"));
        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("c{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("g{i}")).collect(),
            columns: vec![],
        };
        write_h5ad_csr(&src, x.view(), obs, var, &H5AdOptions::default()).unwrap();

        let mut reader = H5AdReader::open(&src, 2).unwrap();
        let mut writer = H5AdWriter::create(&dst, n_obs, n_vars, DataType::F32).unwrap();
        // 12 bytes per f32 nonzero: blocks of at most 2.
        let blocks = write_x_csc(&mut reader, &mut writer, 24).await.unwrap();
        writer.finalize().await.unwrap();
        assert!(
            blocks > 1,
            "budget should force several passes, got {blocks}"
        );

        let csc = x.to_csc();
        let f = hdf5::File::open(&dst).unwrap();
        let g = f.group("X").unwrap();
        assert_eq!(read_str_attr(&g, "encoding-type").unwrap(), "csc_matrix");
        let shape = g.attr("shape").unwrap().read_raw::<u64>().unwrap();
        assert_eq!(shape, vec![n_obs as u64, n_vars as u64]);
        let indptr = f.dataset("X/indptr").unwrap().read_raw::<u64>().unwrap();
        let indices = f.dataset("X/indices").unwrap().read_raw::<u32>().unwrap();
        let data = f.dataset("X/data").unwrap().read_raw::<f32>().unwrap();
        let want_ptr: Vec<u64> = csc
            .indptr()
            .raw_storage()
            .iter()
            .map(|&p| p as u64)
            .collect();
        assert_eq!(indptr, want_ptr);
        assert_eq!(indices, csc.indices());
        assert_eq!(data, csc.data());

        // Read back by gene: genes 1..4 in blocks of 2 are those genes' slices
        // of the CSC, with `row_offset` the first gene.
        let mut back = H5AdReader::open(&dst, 2).unwrap();
        let chunks: Vec<_> = back
            .x_stream_cols(vec![1..4])
            .map(|c| c.unwrap())
            .collect()
            .await;
        assert_eq!(
            chunks
                .iter()
                .map(|c| (c.row_offset, c.nrows))
                .collect::<Vec<_>>(),
            [(1, 2), (3, 1)]
        );
        for c in &chunks {
            for g in 0..c.nrows {
                let col = csc.outer_view(c.row_offset + g).unwrap();
                let nz = c.data.indptr[g] as usize..c.data.indptr[g + 1] as usize;
                assert_eq!(c.data.shape.1, n_obs);
                assert_eq!(&c.data.indices[nz.clone()], col.indices());
                let TypedVec::F32(v) = &c.data.data else {
                    panic!("f32")
                };
                assert_eq!(&v[nz], col.data());
            }
        }
    }

    /// Gene-major X back by cell: in one read when it fits the budget, else
    /// in blocks of cells (one read each). Both give X's rows, in order, in
    /// chunks of `chunk` cells with their offsets in the full matrix.
    #[tokio::test]
    async fn cell_major_stream_reads_blocks_of_cells_within_budget() {
        let (n_obs, n_vars) = (7, 5);
        let mut tri = TriMatI::<f32, u32>::new((n_obs, n_vars));
        for (i, j, v) in [
            (0, 1, 1.),
            (0, 4, 2.),
            (2, 1, 3.),
            (3, 0, 4.),
            (5, 1, 5.),
            (6, 3, 6.),
            (6, 4, 7.),
        ] {
            tri.add_triplet(i, j, v);
        }
        let x = tri.to_csr::<u32>();
        let tmp = tempfile::tempdir().unwrap();
        let (src, dst) = (tmp.path().join("in.h5ad"), tmp.path().join("csc.h5ad"));
        let frame = |n: usize, p: &str| (0..n).map(|i| format!("{p}{i}")).collect::<Vec<_>>();
        let obs = ObsTable {
            index: frame(n_obs, "c"),
            columns: vec![],
        };
        let var = VarTable {
            index: frame(n_vars, "g"),
            columns: vec![],
        };
        write_h5ad_csr(&src, x.view(), obs, var, &H5AdOptions::default()).unwrap();
        let mut reader = H5AdReader::open(&src, 2).unwrap();
        let mut writer = H5AdWriter::create(&dst, n_obs, n_vars, DataType::F32).unwrap();
        write_x_csc(&mut reader, &mut writer, 1 << 20)
            .await
            .unwrap();
        writer.finalize().await.unwrap();

        // 8 bytes per f32 nonzero: 16 bytes is blocks of at most 2 nonzeros.
        for budget in [16, 1 << 20] {
            let mut csc = H5AdReader::open(&dst, 2).unwrap();
            assert!(!csc.x_col_ptr().is_empty());
            let chunks: Vec<_> = x_cell_major_stream(&mut csc, budget, 2)
                .map(|c| c.unwrap())
                .collect()
                .await;
            let mut row = 0;
            for c in &chunks {
                assert_eq!(c.row_offset, row, "budget {budget}");
                assert!(c.nrows <= 2);
                for r in 0..c.nrows {
                    let want = x.outer_view(row + r).unwrap();
                    let nz = c.data.indptr[r] as usize..c.data.indptr[r + 1] as usize;
                    assert_eq!(
                        &c.data.indices[nz.clone()],
                        want.indices(),
                        "budget {budget}"
                    );
                    let TypedVec::F32(v) = &c.data.data else {
                        panic!("f32")
                    };
                    assert_eq!(&v[nz], want.data(), "budget {budget}");
                }
                row += c.nrows;
            }
            assert_eq!(row, n_obs, "budget {budget}");
        }
    }

    /// A reader whose X streams `passes[i]` on its i-th pass (the last pass
    /// repeats), declaring `dtype`, which its chunks needn't share.
    struct Stub {
        shape: (usize, usize),
        dtype: DataType,
        passes: Vec<Vec<MatrixChunk>>,
        pass: usize,
    }

    #[async_trait::async_trait]
    impl DatasetReader for Stub {
        fn shape(&self) -> (usize, usize) {
            self.shape
        }
        fn dtype(&self) -> DataType {
            self.dtype
        }
        async fn obs(&mut self) -> Result<ObsTable> {
            Ok(ObsTable::default())
        }
        async fn var(&mut self) -> Result<VarTable> {
            Ok(VarTable::default())
        }
        fn x_stream(&mut self) -> crate::stream::ChunkStream<'_> {
            let chunks = self.passes[self.pass.min(self.passes.len() - 1)].clone();
            self.pass += 1;
            futures::stream::iter(chunks.into_iter().map(Ok)).boxed()
        }
    }

    /// One chunk of `n_vars` columns holding `rows`, as (column, value) pairs.
    fn f64_chunk(n_vars: usize, rows: &[&[(u32, f64)]]) -> MatrixChunk {
        let mut indptr = vec![0u64];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for row in rows {
            for &(c, v) in *row {
                indices.push(c);
                data.push(v);
            }
            indptr.push(indices.len() as u64);
        }
        MatrixChunk {
            row_offset: 0,
            nrows: rows.len(),
            data: SparseMatrixCSR {
                shape: (rows.len(), n_vars),
                indptr,
                indices,
                data: TypedVec::F64(data),
            },
        }
    }

    /// Chunks whose dtype differs from the reader's declared one (as BPCells
    /// h5seurat streams u32 under an f64 dtype) are cast to the declared
    /// dtype, for each output dtype.
    #[tokio::test]
    async fn chunks_are_cast_to_the_declared_dtype() {
        let x = f64_chunk(3, &[&[(0, 1.0), (2, 2.0)], &[(1, 3.0)]]);
        for dtype in [DataType::F32, DataType::I32, DataType::U32] {
            let mut reader = Stub {
                shape: (2, 3),
                dtype,
                passes: vec![vec![x.clone()]],
                pass: 0,
            };
            let dst = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
            let mut writer = H5AdWriter::create(dst.path(), 2, 3, dtype).unwrap();
            write_x_csc(&mut reader, &mut writer, 1 << 20)
                .await
                .unwrap();
            writer.finalize().await.unwrap();

            let f = hdf5::File::open(dst.path()).unwrap();
            let data = f.dataset("X/data").unwrap();
            assert_eq!(data.dtype().unwrap().size(), 4, "{dtype}");
            // By column: gene 0 (cell 0), gene 1 (cell 1), gene 2 (cell 0).
            assert_eq!(data.read_raw::<f64>().unwrap(), [1.0, 3.0, 2.0], "{dtype}");
            let indices = f.dataset("X/indices").unwrap().read_raw::<u32>().unwrap();
            assert_eq!(indices, [0, 1, 0], "{dtype}");
        }
    }

    /// Every pass reads X again; if it changed since the counting pass the
    /// columns would be misplaced, so the transpose stops with an error.
    #[tokio::test]
    async fn x_changing_between_passes_is_an_error() {
        let mut reader = Stub {
            shape: (1, 2),
            dtype: DataType::F64,
            passes: vec![
                vec![f64_chunk(2, &[&[(0, 1.0), (1, 2.0)]])],
                vec![f64_chunk(2, &[&[(0, 1.0)]])],
            ],
            pass: 0,
        };
        let dst = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
        let mut writer = H5AdWriter::create(dst.path(), 1, 2, DataType::F64).unwrap();
        let err = write_x_csc(&mut reader, &mut writer, 1 << 20)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("changed between passes"), "{err}");
    }

    /// X with no genes has no column blocks; it is still written as an empty
    /// gene-major matrix.
    #[tokio::test]
    async fn x_without_genes_is_still_marked_csc() {
        let mut reader = Stub {
            shape: (2, 0),
            dtype: DataType::F32,
            passes: vec![vec![]],
            pass: 0,
        };
        let dst = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
        let mut writer = H5AdWriter::create(dst.path(), 2, 0, DataType::F32).unwrap();
        assert_eq!(
            write_x_csc(&mut reader, &mut writer, 1 << 20)
                .await
                .unwrap(),
            0
        );
        writer.finalize().await.unwrap();

        let f = hdf5::File::open(dst.path()).unwrap();
        let g = f.group("X").unwrap();
        assert_eq!(read_str_attr(&g, "encoding-type").unwrap(), "csc_matrix");
        assert_eq!(g.attr("shape").unwrap().read_raw::<u64>().unwrap(), [2, 0]);
        assert_eq!(
            f.dataset("X/indptr").unwrap().read_raw::<u64>().unwrap(),
            [0]
        );
    }

    /// A gene-major X whose indptr doesn't have n_vars + 1 entries is refused
    /// on open, naming the dataset.
    #[tokio::test]
    async fn malformed_gene_major_indptr_is_refused() {
        let mut reader = Stub {
            shape: (2, 3),
            dtype: DataType::F32,
            passes: vec![vec![f64_chunk(3, &[&[(0, 1.0)], &[(2, 2.0)]])]],
            pass: 0,
        };
        let dst = tempfile::NamedTempFile::with_suffix(".h5ad").unwrap();
        let mut writer = H5AdWriter::create(dst.path(), 2, 3, DataType::F32).unwrap();
        write_x_csc(&mut reader, &mut writer, 1 << 20)
            .await
            .unwrap();
        writer.finalize().await.unwrap();
        drop(writer);
        {
            let f = hdf5::File::open_rw(dst.path()).unwrap();
            f.unlink("X/indptr").unwrap();
            f.new_dataset_builder()
                .with_data(&[0u64, 1, 2])
                .create("X/indptr")
                .unwrap();
        }
        let err = H5AdReader::open(dst.path(), 2).err().expect("opened");
        assert!(err.to_string().contains("X/indptr has 3 entries"), "{err}");
    }
}
