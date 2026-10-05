use std::fmt;
use std::str::FromStr;

use crate::error::ScxError;

/// Value type of a matrix. Integers are counts; floats are anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F64,
    I32,
    U32,
}

impl DataType {
    /// Short name, as taken by `--dtype` and written to npy `meta.json`.
    pub fn code(self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            DataType::I32 => "i32",
            DataType::U32 => "u32",
        }
    }

    /// Bytes per value.
    pub fn size(self) -> usize {
        match self {
            DataType::F64 => 8,
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
        }
    }
}

/// Parses the short name (`f32`) or the numpy name (`float32`).
impl FromStr for DataType {
    type Err = ScxError;

    fn from_str(s: &str) -> Result<Self, ScxError> {
        [DataType::F32, DataType::F64, DataType::I32, DataType::U32]
            .into_iter()
            .find(|d| s == d.code() || s == d.to_string())
            .ok_or_else(|| {
                ScxError::InvalidFormat(format!("unknown dtype '{s}': use f32, f64, i32 or u32"))
            })
    }
}

/// The numpy name (`float32`), as used for array dtypes in Python.
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

/// A vector of matrix values, of one `DataType`.
#[derive(Debug, Clone)]
pub enum TypedVec {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    U32(Vec<u32>),
}

impl TypedVec {
    /// An empty vector of `dtype`; an empty chunk keeps the matrix's dtype.
    pub fn empty(dtype: DataType) -> Self {
        match dtype {
            DataType::F32 => TypedVec::F32(Vec::new()),
            DataType::F64 => TypedVec::F64(Vec::new()),
            DataType::I32 => TypedVec::I32(Vec::new()),
            DataType::U32 => TypedVec::U32(Vec::new()),
        }
    }

    /// Values from their little-endian bytes (raw HDF5 chunks, NPY bodies).
    pub fn from_le_bytes(dtype: DataType, bytes: &[u8]) -> Result<Self, ScxError> {
        if !bytes.len().is_multiple_of(dtype.size()) {
            return Err(ScxError::InvalidFormat(format!(
                "{} bytes is not a whole number of {dtype} values",
                bytes.len()
            )));
        }
        // pod_collect_to_vec copies, so `bytes` needn't be aligned.
        Ok(match dtype {
            DataType::F32 => TypedVec::F32(bytemuck::pod_collect_to_vec(bytes)),
            DataType::F64 => TypedVec::F64(bytemuck::pod_collect_to_vec(bytes)),
            DataType::I32 => TypedVec::I32(bytemuck::pod_collect_to_vec(bytes)),
            DataType::U32 => TypedVec::U32(bytemuck::pod_collect_to_vec(bytes)),
        })
    }

    /// The values as little-endian bytes, without copying.
    pub fn as_le_bytes(&self) -> &[u8] {
        match self {
            TypedVec::F32(v) => bytemuck::cast_slice(v),
            TypedVec::F64(v) => bytemuck::cast_slice(v),
            TypedVec::I32(v) => bytemuck::cast_slice(v),
            TypedVec::U32(v) => bytemuck::cast_slice(v),
        }
    }

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

    /// `to_f64` on all cores; worth it from roughly 100k values.
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
    fn datatype_parses_short_and_numpy_names() {
        for d in [DataType::F32, DataType::F64, DataType::I32, DataType::U32] {
            assert_eq!(d.code().parse::<DataType>().unwrap(), d);
            assert_eq!(d.to_string().parse::<DataType>().unwrap(), d);
        }
        assert!("f16".parse::<DataType>().is_err());
    }

    #[test]
    fn typedvec_le_bytes_roundtrip() {
        for v in [
            TypedVec::F32(vec![1.5, -2.0]),
            TypedVec::F64(vec![1e300, -0.5]),
            TypedVec::I32(vec![-7, 1 << 30]),
            TypedVec::U32(vec![0, u32::MAX]),
        ] {
            let bytes = v.as_le_bytes().to_vec();
            assert_eq!(bytes.len(), v.len() * v.dtype().size());
            let back = TypedVec::from_le_bytes(v.dtype(), &bytes).unwrap();
            assert_eq!(back.as_le_bytes(), &bytes[..]);
        }
        // unaligned input is fine; a partial value is not
        let b = [0u8, 1, 0, 0, 0];
        assert_eq!(
            TypedVec::from_le_bytes(DataType::U32, &b[1..])
                .unwrap()
                .to_f64(),
            [1.0]
        );
        assert!(TypedVec::from_le_bytes(DataType::U32, &b).is_err());
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
