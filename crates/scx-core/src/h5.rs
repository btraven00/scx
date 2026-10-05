//! HDF5 building blocks shared by the format readers: typed reads of index
//! and value datasets that accept every integer width and signedness the
//! formats use.

use hdf5::types::{FloatSize, TypeDescriptor};
use hdf5::File;
use ndarray::s;

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::MatrixChunk,
};

// ---------------------------------------------------------------------------
// Typed reads shared by the HDF5 readers (h5ad, H5Seurat, 10x)
// ---------------------------------------------------------------------------

/// Read an index dataset (indptr, shape, ...) as u64, whatever its integer
/// width or signedness. Float is accepted too: rhdf5 writes R doubles.
pub(crate) fn read_u64(ds: &hdf5::Dataset) -> Result<Vec<u64>> {
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Unsigned(_) => Ok(ds.read_raw::<u64>()?),
        TypeDescriptor::Integer(_) => ds
            .read_raw::<i64>()?
            .into_iter()
            .map(|x| {
                u64::try_from(x).map_err(|_| {
                    ScxError::InvalidFormat(format!("negative index {x} in {}", ds.name()))
                })
            })
            .collect(),
        TypeDescriptor::Float(_) => Ok(ds
            .read_raw::<f64>()?
            .into_iter()
            .map(|x| x as u64)
            .collect()),
        other => Err(ScxError::InvalidFormat(format!(
            "{}: expected integers, found {other:?}",
            ds.name()
        ))),
    }
}

/// Read `range` of an integer index dataset (indices) as u32.
pub(crate) fn read_u32_range(
    ds: &hdf5::Dataset,
    range: std::ops::Range<usize>,
) -> Result<Vec<u32>> {
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {
            Ok(ds.read_slice_1d::<u32, _>(s![range])?.to_vec())
        }
        other => Err(ScxError::InvalidFormat(format!(
            "{}: expected integers, found {other:?}",
            ds.name()
        ))),
    }
}

/// The DataType a matrix values dataset is read as. Integers of any width map
/// to I32 (signed) or U32 (unsigned): counts fit in 32 bits.
pub(crate) fn value_dtype(ds: &hdf5::Dataset) -> Result<DataType> {
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_) => DataType::F64,
        TypeDescriptor::Integer(_) => DataType::I32,
        TypeDescriptor::Unsigned(_) | TypeDescriptor::Boolean => DataType::U32,
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "{}: unsupported matrix value type {other:?}",
                ds.name()
            )))
        }
    })
}

/// Rows `rows` of a compressed-sparse group (`{group}/indices`, `data`),
/// given its full `indptr`, as a chunk with `n_cols` columns.
pub(crate) fn read_csr_rows(
    file: &File,
    group: &str,
    indptr: &[u64],
    rows: std::ops::Range<usize>,
    n_cols: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let nnz = indptr[rows.start] as usize..indptr[rows.end] as usize;
    let (indices, data) = if nnz.is_empty() {
        (Vec::new(), TypedVec::empty(dtype))
    } else {
        (
            read_u32_range(&file.dataset(&format!("{group}/indices"))?, nnz.clone())?,
            read_values(&file.dataset(&format!("{group}/data"))?, dtype, nnz)?,
        )
    };
    Ok(crate::sparse::csr_chunk(
        indptr, rows, n_cols, indices, data,
    ))
}

/// Read `range` of a values dataset as `dtype`.
pub(crate) fn read_values(
    ds: &hdf5::Dataset,
    dtype: DataType,
    range: std::ops::Range<usize>,
) -> Result<TypedVec> {
    Ok(match dtype {
        DataType::F32 => TypedVec::F32(ds.read_slice_1d::<f32, _>(s![range])?.to_vec()),
        DataType::F64 => TypedVec::F64(ds.read_slice_1d::<f64, _>(s![range])?.to_vec()),
        DataType::I32 => TypedVec::I32(ds.read_slice_1d::<i32, _>(s![range])?.to_vec()),
        DataType::U32 => TypedVec::U32(ds.read_slice_1d::<u32, _>(s![range])?.to_vec()),
    })
}
