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
        for idx in col_start..col_end {
            let row = csc.indices[idx] as usize;
            let dest = cursor[row] as usize;
            csr_indices[dest] = col as u32;
            csr_data[dest] = data_f64[idx];
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
}
