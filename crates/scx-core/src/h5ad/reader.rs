use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Dataset, File, Group};
use ndarray::{s, Array1, Array2};

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    h5_chunk,
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
    },
    stream::DatasetReader,
};

/// Streaming reader for the AnnData `.h5ad` format.
///
/// Spec: <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>
///
/// Supports both sparse (CSR) and dense X storage.
/// Files with CSC X must be converted first:
///   `adata.X = adata.X.tocsr(); adata.write_h5ad(path)`
pub struct H5AdReader {
    path: PathBuf,
    n_obs: usize,
    n_vars: usize,
    /// CSR row pointer array (n_obs + 1 entries). None when X is dense.
    indptr: Option<Vec<u64>>,
    chunk_size: usize,
    dtype: DataType,
    /// Base HDF5 path the matrix is read from: `"X"`, or `"layers/<name>"` when
    /// X is absent (or a layer was requested explicitly).
    x_path: String,
}

impl H5AdReader {
    /// Open with the default matrix source: `/X`, falling back to a layer when
    /// `/X` is absent (common for files written with `adata.X = None`).
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        Self::open_layer(path, chunk_size, None)
    }

    /// Open reading the matrix from `layers/<layer>` instead of `/X`. Passing
    /// `None` uses `/X`, auto-falling-back to a layer when `/X` is missing.
    pub fn open_layer<P: AsRef<Path>>(
        path: P,
        chunk_size: usize,
        layer: Option<&str>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        // Optional root encoding check — tolerate files without it
        if let Ok(root) = file.group("/") {
            if let Ok(enc) = read_str_attr_on_group(&root, "encoding-type") {
                if !enc.is_empty() && enc != "anndata" {
                    return Err(ScxError::InvalidFormat(format!(
                        "not an AnnData file: root encoding-type = '{enc}'"
                    )));
                }
            }
        }

        let base = resolve_matrix_path(&file, layer)?;

        // The matrix can be stored as a dense 2-D dataset or a sparse CSR group.
        let is_dense = file.dataset(&base).is_ok() && file.group(&base).is_err();

        let (n_obs, n_vars, indptr, dtype) = if is_dense {
            let ds = file.dataset(&base)?;
            let sh = ds.shape();
            if sh.len() != 2 {
                return Err(ScxError::InvalidFormat(format!("dense {base} must be 2-D")));
            }
            let dtype = match ds.dtype()?.to_descriptor()? {
                TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
                TypeDescriptor::Float(_) => DataType::F64,
                TypeDescriptor::Integer(_) => DataType::I32,
                _ => DataType::F32,
            };
            (sh[0], sh[1], None, dtype)
        } else {
            let grp = file.group(&base)?;

            if let Ok(enc) = read_str_attr_on_group(&grp, "encoding-type") {
                if enc == "csc_matrix" {
                    return Err(ScxError::InvalidFormat(format!(
                        "{base} is stored as CSC. Convert to CSR first: \
                         adata.X = adata.X.tocsr(); adata.write_h5ad(path)"
                    )));
                }
            }

            let shape_attr = grp
                .attr("shape")
                .map_err(|_| ScxError::InvalidFormat(format!("missing {base}/shape attribute")))?;
            let (n_obs, n_vars) = match shape_attr.dtype()?.to_descriptor()? {
                TypeDescriptor::Integer(IntSize::U8) => {
                    let s: Vec<i64> = shape_attr.read_1d::<i64>()?.to_vec();
                    (s[0] as usize, s[1] as usize)
                }
                _ => {
                    let s: Vec<i32> = shape_attr.read_1d::<i32>()?.to_vec();
                    (s[0] as usize, s[1] as usize)
                }
            };

            let indptr = ad_read_indptr(&file, &format!("{base}/indptr"))?;
            if indptr.len() != n_obs + 1 {
                return Err(ScxError::InvalidFormat(format!(
                    "{base}/indptr length {} != n_obs+1 {}",
                    indptr.len(),
                    n_obs + 1
                )));
            }
            let dtype = ad_detect_dtype(&file, &format!("{base}/data"))?;
            (n_obs, n_vars, Some(indptr), dtype)
        };

        Ok(Self {
            path,
            n_obs,
            n_vars,
            indptr,
            chunk_size,
            dtype,
            x_path: base,
        })
    }

    /// The HDF5 path the matrix is read from — `"X"` or `"layers/<name>"`.
    pub fn x_source(&self) -> &str {
        &self.x_path
    }
}

/// Decide which HDF5 path holds the count matrix.
///
/// - `Some(name)` → `layers/<name>` (errors if that layer is absent).
/// - `None` with `/X` present → `"X"`.
/// - `None` with `/X` absent → fall back to a layer: the sole layer if there's
///   exactly one, else one named `counts`/`X`, else an error listing choices.
fn resolve_matrix_path(file: &File, layer: Option<&str>) -> Result<String> {
    if let Some(name) = layer {
        let p = format!("layers/{name}");
        if file.group(&p).is_err() && file.dataset(&p).is_err() {
            return Err(ScxError::InvalidFormat(format!(
                "layer '{name}' not found (no /layers/{name})"
            )));
        }
        return Ok(p);
    }

    if file.dataset("X").is_ok() || file.group("X").is_ok() {
        return Ok("X".to_string());
    }

    // X absent — try to promote a layer.
    let layers = file
        .group("layers")
        .map_err(|_| ScxError::InvalidFormat("missing /X and no /layers to fall back to".into()))?;
    let names = layers.member_names().unwrap_or_default();
    match names.as_slice() {
        [] => Err(ScxError::InvalidFormat(
            "missing /X and /layers is empty".into(),
        )),
        [only] => Ok(format!("layers/{only}")),
        many => {
            let pick = many.iter().find(|n| *n == "counts" || *n == "X");
            match pick {
                Some(n) => Ok(format!("layers/{n}")),
                None => Err(ScxError::InvalidFormat(format!(
                    "missing /X; multiple layers present ({}). Pick one with --layer <name>",
                    many.join(", ")
                ))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------

fn read_str_attr_on_group(grp: &Group, name: &str) -> Result<String> {
    let attr = grp.attr(name)?;
    Ok(attr.read_scalar::<VarLenUnicode>()?.to_string())
}

fn read_str_attr_on_dataset(ds: &Dataset, name: &str) -> Result<String> {
    let attr = ds.attr(name)?;
    Ok(attr.read_scalar::<VarLenUnicode>()?.to_string())
}

fn ad_read_indptr(file: &File, path: &str) -> Result<Vec<u64>> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(IntSize::U8) => {
            ds.read_1d::<i64>()?.iter().map(|&x| x as u64).collect()
        }
        TypeDescriptor::Integer(_) => ds.read_1d::<i32>()?.iter().map(|&x| x as u64).collect(),
        TypeDescriptor::Float(_) => ds.read_1d::<f64>()?.iter().map(|&x| x as u64).collect(),
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unexpected indptr dtype {:?} at {path}",
                other
            )))
        }
    })
}

pub(super) fn ad_detect_dtype(file: &File, path: &str) -> Result<DataType> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_) => DataType::F64,
        TypeDescriptor::Integer(IntSize::U4) => DataType::I32,
        TypeDescriptor::Integer(IntSize::U8) => DataType::I32, // i64 → i32 (counts fit)
        TypeDescriptor::Unsigned(IntSize::U4) => DataType::U32,
        _ => DataType::F32,
    })
}

/// Read a row slice of a dense 2-D dataset and convert to a sparse CSR chunk.
fn ad_read_dense_chunk(
    path: &Path,
    base: &str,
    row_start: usize,
    row_end: usize,
    n_vars: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    ad_read_dense_chunk_with_dtype(&file, base, row_start, row_end, n_vars, dtype)
}

/// Read a row slice of an arbitrary dense 2-D dataset path and convert to a sparse CSR chunk.
/// The stored dtype is detected from the dataset itself.
fn ad_read_dense_chunk_at(
    path: &Path,
    ds_path: &str,
    row_start: usize,
    row_end: usize,
    n_vars: usize,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    let dtype = ad_detect_dtype(&file, ds_path)?;
    ad_read_dense_chunk_with_dtype(&file, ds_path, row_start, row_end, n_vars, dtype)
}

fn ad_read_dense_chunk_with_dtype(
    file: &File,
    ds_path: &str,
    row_start: usize,
    row_end: usize,
    n_vars: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let ds = file.dataset(ds_path)?;
    let nrows = row_end - row_start;
    let slice = ds.read_slice::<f64, _, _>(s![row_start..row_end, ..])?;
    let csr = dense_array2_to_csr(slice.view(), nrows, n_vars, dtype);
    Ok(MatrixChunk {
        row_offset: row_start,
        nrows,
        data: csr,
    })
}

/// Convert a dense 2-D array view to a CSR sparse matrix, skipping exact zeros.
fn dense_array2_to_csr(
    arr: ndarray::ArrayView2<f64>,
    nrows: usize,
    ncols: usize,
    dtype: DataType,
) -> SparseMatrixCSR {
    let mut indices: Vec<u32> = Vec::new();
    let mut data_f64: Vec<f64> = Vec::new();
    let mut indptr: Vec<u64> = Vec::with_capacity(nrows + 1);
    indptr.push(0);
    for row in arr.rows() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                indices.push(j as u32);
                data_f64.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    let data = match dtype {
        DataType::F32 => TypedVec::F32(data_f64.iter().map(|&x| x as f32).collect()),
        DataType::F64 => TypedVec::F64(data_f64),
        DataType::I32 => TypedVec::I32(data_f64.iter().map(|&x| x as i32).collect()),
        DataType::U32 => TypedVec::U32(data_f64.iter().map(|&x| x as u32).collect()),
    };
    SparseMatrixCSR {
        shape: (nrows, ncols),
        indptr,
        indices,
        data,
    }
}

fn ad_read_strings(file: &File, path: &str) -> Result<Vec<String>> {
    crate::h5_str::read_str_1d(&file.dataset(path)?)
}

/// Read a chunk [row_start, row_end) from a CSR matrix stored at /X/.
/// H5AD natively stores X as CSR, so this is a direct slice — no transpose.
/// Read `X/indices[a..b]` as `u32`. Uses the parallel-inflate fast path for
/// 4-byte, deflate-only-chunked datasets (the common case); otherwise the
/// normal HDF5 read. Column indices are non-negative, so a 4-byte little-endian
/// reinterpret is correct whether stored signed or unsigned.
fn read_x_indices(file: &File, base: &str, a: usize, b: usize) -> Result<Vec<u32>> {
    let ds = file.dataset(&format!("{base}/indices"))?;
    let stored_bytes = ds.dtype()?.size();
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {}
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unexpected {base}/indices dtype {:?}",
                other
            )))
        }
    }
    if stored_bytes == 4 {
        if let Some(plan) = h5_chunk::chunk_plan(&ds) {
            let bytes = h5_chunk::read_range_parallel(&ds, a, b, 4, plan)?;
            return Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect());
        }
    }
    Ok(ds
        .read_slice_1d::<i32, _>(s![a..b])?
        .iter()
        .map(|&x| x as u32)
        .collect())
}

/// Read `X/data[a..b]` as the requested `dtype`. Uses the parallel-inflate fast
/// path when the stored element layout matches the requested dtype and the
/// dataset is deflate-only-chunked; otherwise the normal HDF5 read (which also
/// handles any stored→requested type conversion).
fn read_x_data(file: &File, base: &str, a: usize, b: usize, dtype: DataType) -> Result<TypedVec> {
    let ds = file.dataset(&format!("{base}/data"))?;
    let (want_bytes, want_float) = match dtype {
        DataType::F32 => (4usize, true),
        DataType::F64 => (8, true),
        DataType::I32 | DataType::U32 => (4, false),
    };
    let stored_bytes = ds.dtype()?.size();
    let stored_float = matches!(ds.dtype()?.to_descriptor()?, TypeDescriptor::Float(_));
    if stored_bytes == want_bytes && stored_float == want_float {
        if let Some(plan) = h5_chunk::chunk_plan(&ds) {
            let raw = h5_chunk::read_range_parallel(&ds, a, b, want_bytes, plan)?;
            return Ok(bytes_to_typed(&raw, dtype));
        }
    }
    Ok(match dtype {
        DataType::F32 => TypedVec::F32(ds.read_slice_1d::<f32, _>(s![a..b])?.to_vec()),
        DataType::F64 => TypedVec::F64(ds.read_slice_1d::<f64, _>(s![a..b])?.to_vec()),
        DataType::I32 => TypedVec::I32(ds.read_slice_1d::<i32, _>(s![a..b])?.to_vec()),
        DataType::U32 => TypedVec::U32(ds.read_slice_1d::<u32, _>(s![a..b])?.to_vec()),
    })
}

/// Reinterpret little-endian raw bytes as a `TypedVec` of the given dtype.
fn bytes_to_typed(b: &[u8], dtype: DataType) -> TypedVec {
    match dtype {
        DataType::F32 => TypedVec::F32(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DataType::F64 => TypedVec::F64(
            b.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DataType::I32 => TypedVec::I32(
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DataType::U32 => TypedVec::U32(
            b.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
    }
}

fn ad_read_chunk(
    path: &Path,
    base: &str,
    indptr: &[u64],
    row_start: usize,
    row_end: usize,
    n_vars: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    let nrows = row_end - row_start;
    let nnz_start = indptr[row_start] as usize;
    let nnz_end = indptr[row_end] as usize;
    let nnz = nnz_end - nnz_start;

    let indices: Vec<u32> = if nnz > 0 {
        read_x_indices(&file, base, nnz_start, nnz_end)?
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        read_x_data(&file, base, nnz_start, nnz_end, dtype)?
    } else {
        TypedVec::F32(Vec::new())
    };

    // Normalise indptr to start from 0 for this chunk
    let chunk_indptr: Vec<u64> = indptr[row_start..=row_end]
        .iter()
        .map(|&p| p - indptr[row_start])
        .collect();

    Ok(MatrixChunk {
        row_offset: row_start,
        nrows,
        data: SparseMatrixCSR {
            shape: (nrows, n_vars),
            indptr: chunk_indptr,
            indices,
            data,
        },
    })
}

/// Read a dataframe group at `group_path` (e.g. "obs" or "var").
/// Returns (index, columns).
fn ad_read_dataframe(file: &File, group_path: &str) -> Result<(Vec<String>, Vec<Column>)> {
    let grp = file.group(group_path)?;

    // Index dataset name from _index attr; fall back to "index"
    let index_name = read_str_attr_on_group(&grp, "_index").unwrap_or_else(|_| "index".into());
    let index = ad_read_strings(file, &format!("{group_path}/{index_name}"))?;

    // Column order from attribute
    let col_names: Vec<String> = match grp.attr("column-order") {
        Err(_) => Vec::new(),
        Ok(attr) => {
            let raw: Array1<VarLenUnicode> = attr.read_1d().unwrap_or_default();
            raw.into_iter().map(|s| s.to_string()).collect()
        }
    };

    let mut columns = Vec::new();
    for col_name in col_names {
        let col_path = format!("{group_path}/{col_name}");
        // Groups are categorical; datasets are array/string-array
        let is_group = file.group(&col_path).is_ok() && file.dataset(&col_path).is_err();

        let col_data = if is_group {
            // Distinguish categorical (codes+categories) from nullable (values+mask)
            let has_codes = file.dataset(&format!("{col_path}/codes")).is_ok();
            let has_values = file.dataset(&format!("{col_path}/values")).is_ok();
            let result = if has_codes {
                ad_read_categorical(file, &col_path)
            } else if has_values {
                ad_read_nullable(file, &col_path)
            } else {
                Err(ScxError::InvalidFormat(format!(
                    "unknown group encoding at '{col_path}'"
                )))
            };
            match result {
                Ok(cd) => cd,
                Err(e) => {
                    tracing::warn!("skipping column '{col_name}': {e}");
                    continue;
                }
            }
        } else {
            match ad_read_column(file, &col_path) {
                Ok(cd) => cd,
                Err(e) => {
                    tracing::warn!("skipping column '{col_name}': {e}");
                    continue;
                }
            }
        };
        columns.push(Column {
            name: col_name,
            data: col_data,
        });
    }

    Ok((index, columns))
}

/// Read a single array or string-array dataset as ColumnData.
fn ad_read_column(file: &File, path: &str) -> Result<ColumnData> {
    let ds = file.dataset(path)?;
    // Prefer encoding-type attr; fall back to HDF5 dtype inspection
    let enc = read_str_attr_on_dataset(&ds, "encoding-type").unwrap_or_default();

    if enc == "string-array" {
        return Ok(ColumnData::String(ad_read_strings(file, path)?));
    }

    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => {
            let v: Vec<f32> = ds.read_1d::<f32>()?.to_vec();
            Ok(ColumnData::Float(v.into_iter().map(|x| x as f64).collect()))
        }
        TypeDescriptor::Float(_) => Ok(ColumnData::Float(ds.read_1d::<f64>()?.to_vec())),
        // Native HDF5 boolean type (anndata >= 0.10 uses this for bool columns)
        TypeDescriptor::Boolean => Ok(ColumnData::Bool(ds.read_1d::<bool>()?.to_vec())),
        // 1-byte ints encode bool columns: our writer emits unsigned u8 0/1;
        // some sources use signed i8. (AnnData >= 0.10 uses native Boolean above.)
        TypeDescriptor::Unsigned(IntSize::U1) => {
            let v: Vec<u8> = ds.read_1d::<u8>()?.to_vec();
            Ok(ColumnData::Bool(v.into_iter().map(|x| x != 0).collect()))
        }
        TypeDescriptor::Integer(IntSize::U1) => {
            let v: Vec<i8> = ds.read_1d::<i8>()?.to_vec();
            Ok(ColumnData::Bool(v.into_iter().map(|x| x != 0).collect()))
        }
        TypeDescriptor::Integer(_) => Ok(ColumnData::Int(ds.read_1d::<i32>()?.to_vec())),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            Ok(ColumnData::String(ad_read_strings(file, path)?))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported column dtype {:?} at '{path}'",
            other
        ))),
    }
}

/// Read a categorical group: codes (i8 or i16) + categories (string-array).
pub(super) fn ad_read_categorical(file: &File, grp_path: &str) -> Result<ColumnData> {
    let codes_path = format!("{grp_path}/codes");
    let codes_ds = file.dataset(&codes_path)?;
    let codes: Vec<u32> = match codes_ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(IntSize::U1) => codes_ds
            .read_1d::<i8>()?
            .iter()
            .map(|&x| x as u32)
            .collect(),
        TypeDescriptor::Integer(IntSize::U2) => codes_ds
            .read_1d::<i16>()?
            .iter()
            .map(|&x| x as u32)
            .collect(),
        TypeDescriptor::Integer(_) => codes_ds
            .read_1d::<i32>()?
            .iter()
            .map(|&x| x as u32)
            .collect(),
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unexpected categorical codes dtype {:?}",
                other
            )))
        }
    };

    let levels = ad_read_levels(file, &format!("{grp_path}/categories"))?;
    Ok(ColumnData::Categorical { codes, levels })
}

/// Read a categorical `categories` dataset as level labels. AnnData usually
/// stores these as strings, but pandas Categoricals with integer/float levels
/// (e.g. cluster ids stored as small uints) are equally valid. We coerce any
/// scalar level dtype to its string label so `ColumnData::Categorical` always
/// carries `Vec<String>` levels.
fn ad_read_levels(file: &File, path: &str) -> Result<Vec<String>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => ad_read_strings(file, path),
        TypeDescriptor::Integer(IntSize::U1) => {
            Ok(ds.read_1d::<i8>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            Ok(ds.read_1d::<i16>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            Ok(ds.read_1d::<i64>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Integer(_) => {
            Ok(ds.read_1d::<i32>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            Ok(ds.read_1d::<u8>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            Ok(ds.read_1d::<u16>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            Ok(ds.read_1d::<u64>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(_) => {
            Ok(ds.read_1d::<u32>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Float(FloatSize::U4) => {
            Ok(ds.read_1d::<f32>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Float(_) => {
            Ok(ds.read_1d::<f64>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Boolean => Ok(ds
            .read_1d::<bool>()?
            .iter()
            .map(|v| v.to_string())
            .collect()),
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported categorical level dtype {other:?} at '{path}'"
        ))),
    }
}

/// Read a nullable column group (values + mask) as ColumnData.
/// mask == 0 means valid, mask == 1 means NA.
/// Float/Int columns use NaN for NA; Bool columns use false.
fn ad_read_nullable(file: &File, grp_path: &str) -> Result<ColumnData> {
    let values_path = format!("{grp_path}/values");
    let mask_path = format!("{grp_path}/mask");

    let ds = file.dataset(&values_path)?;
    let mask: Vec<bool> = if let Ok(mds) = file.dataset(&mask_path) {
        match mds.dtype()?.to_descriptor()? {
            TypeDescriptor::Boolean => mds.read_1d::<bool>()?.to_vec(),
            TypeDescriptor::Integer(_) => mds.read_1d::<i8>()?.iter().map(|&x| x != 0).collect(),
            _ => vec![false; ds.shape().first().copied().unwrap_or(0)],
        }
    } else {
        vec![false; ds.shape().first().copied().unwrap_or(0)]
    };

    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => {
            let vals: Vec<f32> = ds.read_1d::<f32>()?.to_vec();
            Ok(ColumnData::Float(
                vals.iter()
                    .zip(&mask)
                    .map(|(&v, &na)| if na { f64::NAN } else { v as f64 })
                    .collect(),
            ))
        }
        TypeDescriptor::Float(_) => {
            let vals: Vec<f64> = ds.read_1d::<f64>()?.to_vec();
            Ok(ColumnData::Float(
                vals.iter()
                    .zip(&mask)
                    .map(|(&v, &na)| if na { f64::NAN } else { v })
                    .collect(),
            ))
        }
        TypeDescriptor::Integer(_) => {
            // Widen nullable int to f64 with NaN for NA
            let vals: Vec<i32> = ds.read_1d::<i32>()?.to_vec();
            Ok(ColumnData::Float(
                vals.iter()
                    .zip(&mask)
                    .map(|(&v, &na)| if na { f64::NAN } else { v as f64 })
                    .collect(),
            ))
        }
        TypeDescriptor::Boolean => {
            let vals: Vec<bool> = ds.read_1d::<bool>()?.to_vec();
            Ok(ColumnData::Bool(
                vals.iter()
                    .zip(&mask)
                    .map(|(&v, &na)| if na { false } else { v })
                    .collect(),
            ))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported nullable column dtype {:?} at '{grp_path}'",
            other
        ))),
    }
}

/// Read the obsm group as named dense matrices.
fn ad_read_obsm(path: &Path, n_obs: usize) -> Result<Embeddings> {
    let file = File::open(path)?;
    let grp = match file.group("obsm") {
        Ok(g) => g,
        Err(_) => return Ok(Embeddings::default()),
    };
    let mut map = HashMap::new();
    for name in grp.member_names().unwrap_or_default() {
        let ds_path = format!("obsm/{name}");
        let ds = match file.dataset(&ds_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let arr: Array2<f64> = match ds.read::<f64, ndarray::Ix2>() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("skipping obsm['{name}']: {e}");
                continue;
            }
        };
        // Guard against transposed storage (some writers store (k, n_obs))
        let arr = if arr.shape()[0] != n_obs && arr.shape()[1] == n_obs {
            arr.t().to_owned()
        } else {
            arr
        };
        let shape = (arr.shape()[0], arr.shape()[1]);
        map.insert(
            name,
            DenseMatrix {
                shape,
                data: arr.into_raw_vec_and_offset().0,
            },
        );
    }
    Ok(Embeddings { map })
}

/// Recursively walk an HDF5 group into a serde_json::Value tree.
fn ad_walk_group(file: &File, group_path: &str) -> Result<serde_json::Value> {
    let grp = file.group(group_path)?;
    let members = grp.member_names().unwrap_or_default();
    let mut map = serde_json::Map::new();
    for name in members {
        let child_path = format!("{group_path}/{name}");
        let is_group = file.group(&child_path).is_ok() && file.dataset(&child_path).is_err();
        let value = if is_group {
            ad_walk_group(file, &child_path).unwrap_or(serde_json::Value::Null)
        } else {
            ad_dataset_to_json(file, &child_path).unwrap_or(serde_json::Value::Null)
        };
        map.insert(name, value);
    }
    Ok(serde_json::Value::Object(map))
}

fn ad_dataset_to_json(file: &File, path: &str) -> Result<serde_json::Value> {
    let ds = file.dataset(path)?;
    let is_scalar = ds.ndim() == 0;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(_) => {
            if is_scalar {
                let v = ds.read_scalar::<f64>()?;
                Ok(serde_json::Value::from(v))
            } else {
                let v: Vec<f64> = ds.read_1d::<f64>()?.to_vec();
                Ok(serde_json::json!(v))
            }
        }
        TypeDescriptor::Integer(_) => {
            if is_scalar {
                let v = ds.read_scalar::<i64>()?;
                Ok(serde_json::Value::from(v))
            } else {
                let v: Vec<i64> = ds.read_1d::<i64>()?.to_vec();
                Ok(serde_json::json!(v))
            }
        }
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            if is_scalar {
                let s = match ds.dtype()?.to_descriptor()? {
                    TypeDescriptor::VarLenUnicode => ds.read_scalar::<VarLenUnicode>()?.to_string(),
                    _ => ds.read_scalar::<hdf5::types::VarLenAscii>()?.to_string(),
                };
                Ok(serde_json::Value::String(s))
            } else {
                let strings = ad_read_strings(file, path)?;
                if strings.len() == 1 {
                    Ok(serde_json::Value::String(
                        strings.into_iter().next().unwrap_or_default(),
                    ))
                } else {
                    Ok(serde_json::json!(strings))
                }
            }
        }
        _ => Ok(serde_json::Value::Null),
    }
}

/// Read the shape and indptr for an H5AD CSR sparse group — used to create a `SparseMatrixMeta`.
fn ad_read_sparse_meta(file: &File, name: &str, group_path: &str) -> Result<SparseMatrixMeta> {
    let grp = file.group(group_path)?;
    let shape_attr = grp.attr("shape")?;
    let (nrows, ncols) = match shape_attr.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(IntSize::U8) => {
            let s: Vec<i64> = shape_attr.read_1d::<i64>()?.to_vec();
            (s[0] as usize, s[1] as usize)
        }
        _ => {
            let s: Vec<i32> = shape_attr.read_1d::<i32>()?.to_vec();
            (s[0] as usize, s[1] as usize)
        }
    };
    let indptr = ad_read_indptr(file, &format!("{group_path}/indptr"))?;
    Ok(SparseMatrixMeta {
        name: name.to_string(),
        shape: (nrows, ncols),
        indptr,
    })
}

/// Read a row-slice of an H5AD CSR sparse group as a `MatrixChunk`.
fn ad_read_sparse_chunk(
    path: &Path,
    group_path: &str,
    meta: &SparseMatrixMeta,
    row_start: usize,
    row_end: usize,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    let (_, ncols) = meta.shape;
    let chunk_rows = row_end - row_start;

    let nnz_start = meta.indptr[row_start] as usize;
    let nnz_end = meta.indptr[row_end] as usize;
    let nnz = nnz_end - nnz_start;

    let indices: Vec<u32> = if nnz > 0 {
        let ds = file.dataset(&format!("{group_path}/indices"))?;
        match ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(_) => ds
                .read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?
                .iter()
                .map(|&x| x as u32)
                .collect(),
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "unexpected indices dtype {other:?} at {group_path}/indices"
                )))
            }
        }
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset(&format!("{group_path}/data"))?;
        match ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Float(FloatSize::U4) => {
                TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            TypeDescriptor::Float(_) => {
                TypedVec::F64(ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            TypeDescriptor::Integer(_) => {
                TypedVec::I32(ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            _ => TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec()),
        }
    } else {
        TypedVec::F32(Vec::new())
    };

    let csr_indptr: Vec<u64> = meta.indptr[row_start..=row_end]
        .iter()
        .map(|&p| p - meta.indptr[row_start])
        .collect();

    Ok(MatrixChunk {
        row_offset: row_start,
        nrows: chunk_rows,
        data: SparseMatrixCSR {
            shape: (chunk_rows, ncols),
            indptr: csr_indptr,
            indices,
            data,
        },
    })
}

// ---------------------------------------------------------------------------
// DatasetReader impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetReader for H5AdReader {
    fn x_indptr(&self) -> &[u64] {
        self.indptr.as_deref().unwrap_or(&[])
    }

    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        let file = File::open(&self.path)?;
        let (index, columns) = ad_read_dataframe(&file, "obs")?;
        Ok(ObsTable { index, columns })
    }

    async fn var(&mut self) -> Result<VarTable> {
        let file = File::open(&self.path)?;
        let (index, columns) = ad_read_dataframe(&file, "var")?;
        Ok(VarTable { index, columns })
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        ad_read_obsm(&self.path, self.n_obs)
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        let file = File::open(&self.path)?;
        match file.group("uns") {
            Err(_) => Ok(UnsTable::default()),
            Ok(_) => {
                let mut raw = ad_walk_group(&file, "uns")?;
                // scx_provenance is stored as a JSON string to preserve keys
                // containing "/" without HDF5 path-separator mangling.
                // Parse it back to an Object so callers get the expected shape.
                if let Some(obj) = raw.as_object_mut() {
                    if let Some(serde_json::Value::String(s)) = obj.get("scx_provenance") {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                            obj.insert("scx_provenance".to_string(), parsed);
                        }
                    }
                }
                Ok(UnsTable { raw })
            }
        }
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let file = File::open(&self.path)?;
        let grp = match file.group("layers") {
            Err(_) => return Ok(Vec::new()),
            Ok(g) => g,
        };
        let mut metas = Vec::new();
        for name in grp.member_names().unwrap_or_default() {
            let grp_path = format!("layers/{name}");
            // Dense layer: read shape from dataset dimensions, indptr unused for inspect.
            if let Ok(ds) = file.dataset(&grp_path) {
                if file.group(&grp_path).is_err() {
                    let shape = ds.shape();
                    if shape.len() == 2 {
                        metas.push(SparseMatrixMeta {
                            name: name.clone(),
                            shape: (shape[0], shape[1]),
                            indptr: Vec::new(),
                        });
                    } else {
                        tracing::warn!(
                            "skipping dense layer '{name}': unexpected rank {}",
                            shape.len()
                        );
                    }
                    continue;
                }
            }
            match ad_read_sparse_meta(&file, &name, &grp_path) {
                Ok(m) => metas.push(m),
                Err(e) => tracing::warn!("skipping layers['{name}']: {e}"),
            }
        }
        Ok(metas)
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let file = File::open(&self.path)?;
        let grp = match file.group("obsp") {
            Err(_) => return Ok(Vec::new()),
            Ok(g) => g,
        };
        let mut metas = Vec::new();
        for name in grp.member_names().unwrap_or_default() {
            match ad_read_sparse_meta(&file, &name, &format!("obsp/{name}")) {
                Ok(m) => metas.push(m),
                Err(e) => tracing::warn!("skipping obsp['{name}']: {e}"),
            }
        }
        Ok(metas)
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        let path = self.path.clone();
        let grp_path = format!("layers/{}", meta.name);
        let n_rows = meta.shape.0;
        let n_cols = meta.shape.1;
        let is_dense = meta.indptr.is_empty();
        Box::pin(stream::unfold(0usize, move |row_start| {
            let path = path.clone();
            let grp_path = grp_path.clone();
            async move {
                if row_start >= n_rows {
                    return None;
                }
                let row_end = (row_start + chunk_size).min(n_rows);
                let chunk = if is_dense {
                    ad_read_dense_chunk_at(&path, &grp_path, row_start, row_end, n_cols)
                } else {
                    ad_read_sparse_chunk(&path, &grp_path, meta, row_start, row_end)
                };
                Some((chunk, row_end))
            }
        }))
    }

    fn obsp_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        let path = self.path.clone();
        let grp_path = format!("obsp/{}", meta.name);
        let n_rows = meta.shape.0;
        Box::pin(stream::unfold(0usize, move |row_start| {
            let path = path.clone();
            let grp_path = grp_path.clone();
            async move {
                if row_start >= n_rows {
                    return None;
                }
                let row_end = (row_start + chunk_size).min(n_rows);
                let chunk = ad_read_sparse_chunk(&path, &grp_path, meta, row_start, row_end);
                Some((chunk, row_end))
            }
        }))
    }

    async fn varm(&mut self) -> Result<Varm> {
        let file = File::open(&self.path)?;
        let grp = match file.group("varm") {
            Err(_) => return Ok(Varm::default()),
            Ok(g) => g,
        };
        let mut map = HashMap::new();
        for name in grp.member_names().unwrap_or_default() {
            let ds = match file.dataset(&format!("varm/{name}")) {
                Ok(d) => d,
                Err(_) => continue,
            };
            match ds.read::<f64, ndarray::Ix2>() {
                Ok(arr) => {
                    let shape = (arr.shape()[0], arr.shape()[1]);
                    map.insert(
                        name,
                        DenseMatrix {
                            shape,
                            data: arr.into_raw_vec_and_offset().0,
                        },
                    );
                }
                Err(e) => tracing::warn!("skipping varm['{name}']: {e}"),
            }
        }
        Ok(Varm { map })
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let path = self.path.clone();
        let base = self.x_path.clone();
        let n_obs = self.n_obs;
        let n_vars = self.n_vars;
        let chunk_size = self.chunk_size;
        let dtype = self.dtype;

        match &self.indptr {
            Some(indptr) => {
                let indptr = indptr.clone();
                Box::pin(stream::unfold(0usize, move |row_start| {
                    let path = path.clone();
                    let base = base.clone();
                    let indptr = indptr.clone();
                    async move {
                        if row_start >= n_obs {
                            return None;
                        }
                        let row_end = (row_start + chunk_size).min(n_obs);
                        let chunk =
                            ad_read_chunk(&path, &base, &indptr, row_start, row_end, n_vars, dtype);
                        Some((chunk, row_end))
                    }
                }))
            }
            None => {
                // Dense X: read rows slice-by-slice and convert to sparse CSR
                Box::pin(stream::unfold(0usize, move |row_start| {
                    let path = path.clone();
                    let base = base.clone();
                    async move {
                        if row_start >= n_obs {
                            return None;
                        }
                        let row_end = (row_start + chunk_size).min(n_obs);
                        let chunk =
                            ad_read_dense_chunk(&path, &base, row_start, row_end, n_vars, dtype);
                        Some((chunk, row_end))
                    }
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
