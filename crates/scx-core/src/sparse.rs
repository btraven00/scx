use crate::dtype::TypedVec;
use crate::ir::{SparseMatrixCSC, SparseMatrixCSR};

/// Convert a full CSC matrix to CSR.
pub fn csc_to_csr(csc: &SparseMatrixCSC) -> SparseMatrixCSR {
    let (nrows, ncols) = csc.shape;
    let nnz = csc.indices.len();
    let data_f64 = csc.data.to_f64();

    // Count entries per row
    let mut row_counts = vec![0u64; nrows];
    for &row_idx in &csc.indices {
        row_counts[row_idx as usize] += 1;
    }

    // Build CSR indptr
    let mut indptr = vec![0u64; nrows + 1];
    for i in 0..nrows {
        indptr[i + 1] = indptr[i] + row_counts[i];
    }

    // Fill CSR data
    let mut csr_indices = vec![0u32; nnz];
    let mut csr_data = vec![0f64; nnz];
    let mut cursor = indptr.clone();

    for col in 0..ncols {
        let col_start = csc.indptr[col] as usize;
        let col_end = csc.indptr[col + 1] as usize;
        for (&row_idx, &val) in csc.indices[col_start..col_end]
            .iter()
            .zip(data_f64[col_start..col_end].iter())
        {
            let row = row_idx as usize;
            let dest = cursor[row] as usize;
            csr_indices[dest] = col as u32;
            csr_data[dest] = val;
            cursor[row] += 1;
        }
    }

    // Preserve original dtype
    let typed_data = match &csc.data {
        TypedVec::F32(_) => TypedVec::F32(csr_data.iter().map(|&x| x as f32).collect()),
        TypedVec::F64(_) => TypedVec::F64(csr_data),
        TypedVec::I32(_) => TypedVec::I32(csr_data.iter().map(|&x| x as i32).collect()),
        TypedVec::U32(_) => TypedVec::U32(csr_data.iter().map(|&x| x as u32).collect()),
    };

    SparseMatrixCSR {
        shape: (nrows, ncols),
        indptr,
        indices: csr_indices,
        data: typed_data,
    }
}

/// Extract a row-slice [row_start..row_end) from a CSR matrix.
pub fn csr_slice_rows(csr: &SparseMatrixCSR, row_start: usize, row_end: usize) -> SparseMatrixCSR {
    let nrows = row_end - row_start;
    let nnz_start = csr.indptr[row_start] as usize;
    let nnz_end = csr.indptr[row_end] as usize;

    let indptr: Vec<u64> = csr.indptr[row_start..=row_end]
        .iter()
        .map(|&p| p - csr.indptr[row_start])
        .collect();

    let indices = csr.indices[nnz_start..nnz_end].to_vec();

    let data = match &csr.data {
        TypedVec::F32(v) => TypedVec::F32(v[nnz_start..nnz_end].to_vec()),
        TypedVec::F64(v) => TypedVec::F64(v[nnz_start..nnz_end].to_vec()),
        TypedVec::I32(v) => TypedVec::I32(v[nnz_start..nnz_end].to_vec()),
        TypedVec::U32(v) => TypedVec::U32(v[nnz_start..nnz_end].to_vec()),
    };

    SparseMatrixCSR {
        shape: (nrows, csr.shape.1),
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
    fn test_csc_to_csr_identity() {
        // 3x3 identity matrix in CSC
        let csc = SparseMatrixCSC {
            shape: (3, 3),
            indptr: vec![0, 1, 2, 3],
            indices: vec![0, 1, 2],
            data: TypedVec::F32(vec![1.0, 1.0, 1.0]),
        };

        let csr = csc_to_csr(&csc);
        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.indptr, vec![0, 1, 2, 3]);
        assert_eq!(csr.indices, vec![0, 1, 2]);
        match &csr.data {
            TypedVec::F32(v) => assert_eq!(v, &vec![1.0, 1.0, 1.0]),
            _ => panic!("expected F32"),
        }
    }

    #[test]
    fn test_csc_to_csr_rectangular() {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]] in CSC
        let csc = SparseMatrixCSC {
            shape: (2, 3),
            indptr: vec![0, 1, 2, 3],
            indices: vec![0, 1, 0],
            data: TypedVec::F32(vec![1.0, 3.0, 2.0]),
        };

        let csr = csc_to_csr(&csc);
        assert_eq!(csr.shape, (2, 3));
        assert_eq!(csr.indptr, vec![0, 2, 3]);
        assert_eq!(csr.indices, vec![0, 2, 1]); // row 0: cols 0,2; row 1: col 1
        match &csr.data {
            TypedVec::F32(v) => assert_eq!(v, &vec![1.0, 2.0, 3.0]),
            _ => panic!("expected F32"),
        }
    }

    #[test]
    fn test_csc_to_csr_preserves_dtype() {
        // [[1,0,2],[0,3,0]] in CSC — same layout, varying value dtype.
        for data in [
            TypedVec::F64(vec![1.0, 3.0, 2.0]),
            TypedVec::I32(vec![1, 3, 2]),
            TypedVec::U32(vec![1, 3, 2]),
        ] {
            let want = data.dtype();
            let csc = SparseMatrixCSC {
                shape: (2, 3),
                indptr: vec![0, 1, 2, 3],
                indices: vec![0, 1, 0],
                data,
            };
            let csr = csc_to_csr(&csc);
            assert_eq!(csr.data.dtype(), want, "dtype preserved through CSC->CSR");
            assert_eq!(csr.indices, vec![0, 2, 1]);
        }
    }

    #[test]
    fn test_csr_slice_rows_preserves_dtype() {
        for data in [
            TypedVec::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            TypedVec::I32(vec![1, 2, 3, 4, 5]),
            TypedVec::U32(vec![1, 2, 3, 4, 5]),
        ] {
            let want = data.dtype();
            let csr = SparseMatrixCSR {
                shape: (3, 3),
                indptr: vec![0, 2, 3, 5],
                indices: vec![0, 2, 1, 0, 2],
                data,
            };
            let slice = csr_slice_rows(&csr, 1, 3);
            assert_eq!(
                slice.data.dtype(),
                want,
                "dtype preserved through row slice"
            );
            assert_eq!(slice.indices, vec![1, 0, 2]);
        }
    }

    #[test]
    fn test_csr_slice_rows() {
        // 3x3 matrix: [[1,0,2],[0,3,0],[4,0,5]] in CSR
        let csr = SparseMatrixCSR {
            shape: (3, 3),
            indptr: vec![0, 2, 3, 5],
            indices: vec![0, 2, 1, 0, 2],
            data: TypedVec::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        };

        let slice = csr_slice_rows(&csr, 1, 3);
        assert_eq!(slice.shape, (2, 3));
        assert_eq!(slice.indptr, vec![0, 1, 3]);
        assert_eq!(slice.indices, vec![1, 0, 2]);
        match &slice.data {
            TypedVec::F32(v) => assert_eq!(v, &vec![3.0, 4.0, 5.0]),
            _ => panic!("expected F32"),
        }
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
