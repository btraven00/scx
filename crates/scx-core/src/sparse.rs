use std::ops::Range;

use crate::dtype::{DataType, TypedVec};
use crate::ir::{MatrixChunk, SparseMatrixCSR};

/// Rows `rows` of a stored CSR matrix (CSC on disk for Seurat and 10x: the
/// same arrays, cells as the compressed axis) as a chunk. `indptr` is the
/// matrix's full pointer array; `indices` and `data` hold exactly the
/// nonzeros of `rows`. Column indices come out sorted within each row,
/// whatever the file had.
pub(crate) fn csr_chunk(
    indptr: &[u64],
    rows: Range<usize>,
    n_cols: usize,
    indices: Vec<u32>,
    data: TypedVec,
) -> MatrixChunk {
    let base = indptr[rows.start];
    let mut csr = SparseMatrixCSR {
        shape: (rows.len(), n_cols),
        indptr: indptr[rows.start..=rows.end]
            .iter()
            .map(|&p| p - base)
            .collect(),
        indices,
        data,
    };
    sort_csr_indices(&mut csr);
    MatrixChunk {
        row_offset: rows.start,
        nrows: rows.len(),
        data: csr,
    }
}

/// Row-major dense values (`n_cols` per row) as CSR, dropping exact zeros.
pub(crate) fn dense_to_csr(values: &[f64], n_cols: usize, dtype: DataType) -> SparseMatrixCSR {
    let nrows = values.len().checked_div(n_cols).unwrap_or(0);
    let mut indptr = Vec::with_capacity(nrows + 1);
    let (mut indices, mut kept) = (Vec::new(), Vec::new());
    indptr.push(0u64);
    for row in values.chunks(n_cols.max(1)) {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                indices.push(j as u32);
                kept.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    let data = match dtype {
        DataType::F32 => TypedVec::F32(kept.iter().map(|&x| x as f32).collect()),
        DataType::F64 => TypedVec::F64(kept),
        DataType::I32 => TypedVec::I32(kept.iter().map(|&x| x as i32).collect()),
        DataType::U32 => TypedVec::U32(kept.iter().map(|&x| x as u32).collect()),
    };
    SparseMatrixCSR {
        shape: (nrows, n_cols),
        indptr,
        indices,
        data,
    }
}

/// Sort each row's column indices ascending, carrying the values along.
///
/// H5AD allows unsorted indices within a row (scipy's `has_sorted_indices` is
/// False after e.g. concatenation or slicing), but dgCMatrix, BPCells and any
/// CSR -> CSC consumer require them sorted: picklerick handed such rows straight
/// to `new("dgCMatrix")` and R rejected them ("'i' slot is not increasing
/// within columns"). Rows already in order are left untouched, so sorted input
/// pays one linear scan. Duplicate indices are kept, adjacent; summing them is
/// a separate canonicalisation this does not do.
pub fn sort_csr_indices(csr: &mut SparseMatrixCSR) {
    fn sort_rows<T: Copy>(indptr: &[u64], indices: &mut [u32], data: &mut [T]) {
        let mut row: Vec<(u32, T)> = Vec::new();
        for w in indptr.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            if indices[a..b].windows(2).all(|p| p[0] < p[1]) {
                continue;
            }
            row.clear();
            row.extend(
                indices[a..b]
                    .iter()
                    .copied()
                    .zip(data[a..b].iter().copied()),
            );
            row.sort_by_key(|&(i, _)| i);
            for (k, &(i, v)) in row.iter().enumerate() {
                indices[a + k] = i;
                data[a + k] = v;
            }
        }
    }
    let SparseMatrixCSR {
        indptr,
        indices,
        data,
        ..
    } = csr;
    match data {
        TypedVec::F32(v) => sort_rows(indptr, indices, v),
        TypedVec::F64(v) => sort_rows(indptr, indices, v),
        TypedVec::I32(v) => sort_rows(indptr, indices, v),
        TypedVec::U32(v) => sort_rows(indptr, indices, v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_chunk_rebases_indptr_and_sorts_rows() {
        // full indptr of a 3-row matrix; take rows 1..3 (nnz 2..5)
        let indptr = [0u64, 2, 4, 5];
        let chunk = csr_chunk(
            &indptr,
            1..3,
            4,
            vec![3, 0, 2],
            TypedVec::I32(vec![30, 10, 20]),
        );
        assert_eq!((chunk.row_offset, chunk.nrows), (1, 2));
        assert_eq!(chunk.data.shape, (2, 4));
        assert_eq!(chunk.data.indptr, vec![0, 2, 3]);
        assert_eq!(chunk.data.indices, vec![0, 3, 2]);
        assert_eq!(chunk.data.data.to_f64(), vec![10.0, 30.0, 20.0]);
    }

    #[test]
    fn dense_to_csr_drops_zeros_and_keeps_dtype() {
        let csr = dense_to_csr(
            &[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 7.0],
            3,
            DataType::U32,
        );
        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.indptr, vec![0, 1, 1, 3]);
        assert_eq!(csr.indices, vec![1, 0, 2]);
        assert!(matches!(csr.data, TypedVec::U32(ref v) if v == &[2, 5, 7]));
    }

    #[test]
    fn test_sort_csr_indices_sorts_rows_and_carries_values() {
        // Row 0 unsorted, row 1 empty, row 2 already sorted, row 3 reversed.
        let mut csr = SparseMatrixCSR {
            shape: (4, 5),
            indptr: vec![0, 3, 3, 5, 8],
            indices: vec![4, 0, 2, 1, 3, 4, 2, 0],
            data: TypedVec::F32(vec![40.0, 0.0, 20.0, 11.0, 13.0, 34.0, 32.0, 30.0]),
        };
        sort_csr_indices(&mut csr);
        assert_eq!(csr.indptr, vec![0, 3, 3, 5, 8]);
        assert_eq!(csr.indices, vec![0, 2, 4, 1, 3, 0, 2, 4]);
        match csr.data {
            TypedVec::F32(v) => assert_eq!(v, vec![0.0, 20.0, 40.0, 11.0, 13.0, 30.0, 32.0, 34.0]),
            _ => panic!("dtype changed"),
        }
    }

    #[test]
    fn test_sort_csr_indices_sorted_input_unchanged() {
        let orig = SparseMatrixCSR {
            shape: (2, 3),
            indptr: vec![0, 2, 3],
            indices: vec![0, 2, 1],
            data: TypedVec::I32(vec![1, 2, 3]),
        };
        let mut csr = orig.clone();
        sort_csr_indices(&mut csr);
        assert_eq!(csr.indices, orig.indices);
        assert_eq!(csr.data.to_f64(), orig.data.to_f64());
    }
}
