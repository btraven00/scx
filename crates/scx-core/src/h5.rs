//! HDF5 building blocks shared by the format readers: typed reads of index
//! and value datasets that accept every integer width and signedness the
//! formats use.

use hdf5::types::{FloatSize, TypeDescriptor};
use hdf5::File;
use ndarray::s;
use std::path::Path;

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

// ---------------------------------------------------------------------------
// Generic tree walk, for `scx inspect` on HDF5 files of no known format
// ---------------------------------------------------------------------------

/// A node in the HDF5 file tree.
pub struct H5Node {
    pub name: String,
    pub kind: H5NodeKind,
}

pub enum H5NodeKind {
    Dataset {
        shape: Vec<usize>,
        dtype: String,
    },
    Group {
        children: Vec<H5Node>,
        /// Number of children that were omitted due to depth limit.
        truncated: usize,
    },
}

/// Walk the root of an HDF5 file up to `max_depth` levels deep.
pub fn walk_h5(path: &Path, max_depth: usize) -> Result<Vec<H5Node>> {
    let file = File::open(path)?;
    let root = file
        .group("/")
        .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
    walk_group(&file, &root, max_depth)
}

fn walk_group(file: &File, grp: &hdf5::Group, depth: usize) -> Result<Vec<H5Node>> {
    let names = grp.member_names().unwrap_or_default();
    let mut nodes = Vec::with_capacity(names.len());

    for name in &names {
        let full_path = {
            let grp_name = grp.name();
            if grp_name == "/" {
                format!("/{name}")
            } else {
                format!("{grp_name}/{name}")
            }
        };

        let is_group = file.group(&full_path).is_ok() && file.dataset(&full_path).is_err();

        let kind = if is_group {
            if depth == 0 {
                H5NodeKind::Group {
                    children: Vec::new(),
                    truncated: file
                        .group(&full_path)
                        .ok()
                        .and_then(|g| g.member_names().ok())
                        .map(|v| v.len())
                        .unwrap_or(0),
                }
            } else {
                let child_grp = file
                    .group(&full_path)
                    .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
                let children = walk_group(file, &child_grp, depth - 1)?;
                H5NodeKind::Group {
                    children,
                    truncated: 0,
                }
            }
        } else {
            match file.dataset(&full_path) {
                Ok(ds) => {
                    let shape = ds.shape();
                    let dtype = dtype_str(&ds);
                    H5NodeKind::Dataset { shape, dtype }
                }
                Err(_) => continue,
            }
        };

        nodes.push(H5Node {
            name: name.clone(),
            kind,
        });
    }

    nodes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(nodes)
}

fn dtype_str(ds: &hdf5::Dataset) -> String {
    match ds.dtype().and_then(|d| d.to_descriptor()) {
        Ok(TypeDescriptor::Float(s)) => format!("f{}", (s as usize) * 8),
        Ok(TypeDescriptor::Integer(s)) => format!("i{}", (s as usize) * 8),
        Ok(TypeDescriptor::Unsigned(s)) => format!("u{}", (s as usize) * 8),
        Ok(TypeDescriptor::Boolean) => "bool".into(),
        Ok(TypeDescriptor::VarLenUnicode) => "str".into(),
        Ok(TypeDescriptor::VarLenAscii) => "str".into(),
        Ok(TypeDescriptor::FixedAscii(n)) => format!("str[{n}]"),
        Ok(TypeDescriptor::FixedUnicode(n)) => format!("str[{n}]"),
        _ => "?".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_h5_tree_and_depth_truncation() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny/tiny_10x.h5");
        let nodes = walk_h5(&p, 3).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "matrix");
        let H5NodeKind::Group {
            children,
            truncated,
        } = &nodes[0].kind
        else {
            panic!("matrix should be a group");
        };
        assert_eq!(*truncated, 0);
        let names: Vec<&str> = children.iter().map(|n| n.name.as_str()).collect();
        for expected in ["barcodes", "data", "features", "indices", "indptr", "shape"] {
            assert!(names.contains(&expected), "missing {expected}: {names:?}");
        }

        // depth 0: the matrix group's children are omitted but counted.
        let shallow = walk_h5(&p, 0).unwrap();
        let H5NodeKind::Group {
            children,
            truncated,
        } = &shallow[0].kind
        else {
            panic!("matrix should be a group");
        };
        assert!(children.is_empty());
        assert!(*truncated > 0);
    }
}
