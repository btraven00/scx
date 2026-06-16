use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F64,
    I32,
    U32,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::F32 => write!(f, "float32"),
            DataType::F64 => write!(f, "float64"),
            DataType::I32 => write!(f, "int32"),
            DataType::U32 => write!(f, "uint32"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypedVec {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    U32(Vec<u32>),
}

impl TypedVec {
    pub fn dtype(&self) -> DataType {
        match self {
            TypedVec::F32(_) => DataType::F32,
            TypedVec::F64(_) => DataType::F64,
            TypedVec::I32(_) => DataType::I32,
            TypedVec::U32(_) => DataType::U32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TypedVec::F32(v) => v.len(),
            TypedVec::F64(v) => v.len(),
            TypedVec::I32(v) => v.len(),
            TypedVec::U32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to f64 values (lossy for large integers).
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            TypedVec::F32(v) => v.iter().map(|&x| x as f64).collect(),
            TypedVec::F64(v) => v.clone(),
            TypedVec::I32(v) => v.iter().map(|&x| x as f64).collect(),
            TypedVec::U32(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    /// Parallel version of `to_f64` — use when len() is large (>100k elements).
    /// Within each rayon thread LLVM auto-vectorises the cast loop to AVX2/SSE4.
    pub fn to_f64_par(&self) -> Vec<f64> {
        use rayon::prelude::*;
        match self {
            TypedVec::F32(v) => v.par_iter().map(|&x| x as f64).collect(),
            TypedVec::F64(v) => v.clone(),
            TypedVec::I32(v) => v.par_iter().map(|&x| x as f64).collect(),
            TypedVec::U32(v) => v.par_iter().map(|&x| x as f64).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_display_all_variants() {
        assert_eq!(DataType::F32.to_string(), "float32");
        assert_eq!(DataType::F64.to_string(), "float64");
        assert_eq!(DataType::I32.to_string(), "int32");
        assert_eq!(DataType::U32.to_string(), "uint32");
    }

    #[test]
    fn typedvec_dtype_len_is_empty() {
        assert_eq!(TypedVec::F32(vec![1.0, 2.0]).dtype(), DataType::F32);
        assert_eq!(TypedVec::F64(vec![1.0]).dtype(), DataType::F64);
        assert_eq!(TypedVec::I32(vec![1, 2, 3]).dtype(), DataType::I32);
        assert_eq!(TypedVec::U32(vec![]).dtype(), DataType::U32);

        assert_eq!(TypedVec::F32(vec![1.0, 2.0]).len(), 2);
        assert_eq!(TypedVec::F64(vec![1.0]).len(), 1);
        assert_eq!(TypedVec::I32(vec![1, 2, 3]).len(), 3);

        assert!(!TypedVec::I32(vec![1]).is_empty());
        assert!(TypedVec::U32(vec![]).is_empty());
    }

    #[test]
    fn typedvec_to_f64_serial_and_parallel_agree() {
        let expected = vec![1.0_f64, 2.0, 3.0];
        for v in [
            TypedVec::F32(vec![1.0, 2.0, 3.0]),
            TypedVec::F64(vec![1.0, 2.0, 3.0]),
            TypedVec::I32(vec![1, 2, 3]),
            TypedVec::U32(vec![1, 2, 3]),
        ] {
            assert_eq!(v.to_f64(), expected);
            assert_eq!(v.to_f64_par(), expected);
        }
    }
}
