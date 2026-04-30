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

    /// Cast to the requested `DataType`, reusing the allocation when the type matches.
    pub fn cast(self, dtype: DataType) -> Self {
        if self.dtype() == dtype {
            return self;
        }
        match dtype {
            DataType::F32 => TypedVec::F32(match &self {
                TypedVec::F32(v) => v.clone(),
                TypedVec::F64(v) => v.iter().map(|&x| x as f32).collect(),
                TypedVec::I32(v) => v.iter().map(|&x| x as f32).collect(),
                TypedVec::U32(v) => v.iter().map(|&x| x as f32).collect(),
            }),
            DataType::F64 => TypedVec::F64(match &self {
                TypedVec::F32(v) => v.iter().map(|&x| x as f64).collect(),
                TypedVec::F64(v) => v.clone(),
                TypedVec::I32(v) => v.iter().map(|&x| x as f64).collect(),
                TypedVec::U32(v) => v.iter().map(|&x| x as f64).collect(),
            }),
            DataType::I32 => TypedVec::I32(match &self {
                TypedVec::F32(v) => v.iter().map(|&x| x as i32).collect(),
                TypedVec::F64(v) => v.iter().map(|&x| x as i32).collect(),
                TypedVec::I32(v) => v.clone(),
                TypedVec::U32(v) => v.iter().map(|&x| x as i32).collect(),
            }),
            DataType::U32 => TypedVec::U32(match &self {
                TypedVec::F32(v) => v.iter().map(|&x| x as u32).collect(),
                TypedVec::F64(v) => v.iter().map(|&x| x as u32).collect(),
                TypedVec::I32(v) => v.iter().map(|&x| x as u32).collect(),
                TypedVec::U32(v) => v.clone(),
            }),
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
