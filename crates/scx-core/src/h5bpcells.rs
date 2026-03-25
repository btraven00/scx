//! HDF5 backend for BPCells-format matrices embedded in HDF5 files.
//!
//! BPCells stores exactly the same logical schema in HDF5 as in the directory
//! format, but each named array becomes an HDF5 dataset and the version string
//! is an HDF5 **attribute** on the group (not a dataset).
//!
//! References:
//!   - `arrayIO/hdf5.cpp`: `writeVersion` writes `"version"` attribute;
//!     `readVersion` reads it back.
//!   - `matrixIterators/StoredMatrix.h`: same dataset names as directory format.

use hdf5::{File, Group};
use hdf5::types::{VarLenAscii, VarLenUnicode};
use ndarray::Array1;

use crate::bpcells::{
    decode_d1z, decode_for, BpcellsDatasetReader, StorageOrder, ValStore,
};
use crate::dtype::DataType;
use crate::error::{Result, ScxError};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Read a 1-D uint32 dataset from an HDF5 group.
fn read_u32s(grp: &Group, name: &str) -> Result<Vec<u32>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<u32> = ds.read_1d().map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}"))
    })?;
    Ok(arr.to_vec())
}

/// Read a 1-D uint64 dataset from an HDF5 group.
fn read_u64s(grp: &Group, name: &str) -> Result<Vec<u64>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<u64> = ds.read_1d().map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}"))
    })?;
    Ok(arr.to_vec())
}

/// Read a 1-D float32 dataset.
fn read_f32s(grp: &Group, name: &str) -> Result<Vec<f32>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<f32> = ds.read_1d().map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}"))
    })?;
    Ok(arr.to_vec())
}

/// Read a 1-D float64 dataset.
fn read_f64s(grp: &Group, name: &str) -> Result<Vec<f64>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<f64> = ds.read_1d().map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}"))
    })?;
    Ok(arr.to_vec())
}

/// Read a 1-D string dataset (variable-length Unicode or ASCII).
fn read_strings(grp: &Group, name: &str) -> Result<Vec<String>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    if ds.size() == 0 {
        return Ok(vec![]);
    }
    match ds.dtype().and_then(|t| t.to_descriptor()) {
        Ok(hdf5::types::TypeDescriptor::VarLenAscii) => {
            let arr: Array1<VarLenAscii> = ds.read_1d().map_err(|e| {
                ScxError::InvalidFormat(format!("reading '{name}': {e}"))
            })?;
            Ok(arr.into_iter().map(|s| s.to_string()).collect())
        }
        _ => {
            let arr: Array1<VarLenUnicode> = ds.read_1d().map_err(|e| {
                ScxError::InvalidFormat(format!("reading '{name}': {e}"))
            })?;
            Ok(arr.into_iter().map(|s| s.to_string()).collect())
        }
    }
}

/// Read the scalar `"version"` attribute from the group.
///
/// BPCells R writes this as an H5S_SCALAR variable-length UTF-8 string.
/// Other tools may write it as a 1-D array of length 1. Try both.
pub fn read_version_attr(grp: &Group) -> Option<String> {
    let attr = grp.attr("version").ok()?;
    // Scalar (BPCells R / C++ native path)
    if let Ok(s) = attr.read_scalar::<VarLenUnicode>() {
        return Some(s.to_string());
    }
    if let Ok(s) = attr.read_scalar::<VarLenAscii>() {
        return Some(s.to_string());
    }
    // 1-D array fallback
    if let Ok(arr) = attr.read_1d::<VarLenUnicode>() {
        return arr.into_iter().next().map(|s| s.to_string());
    }
    if let Ok(arr) = attr.read_1d::<VarLenAscii>() {
        return arr.into_iter().next().map(|s| s.to_string());
    }
    None
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Probe an H5Seurat assay group: return the BPCells version string if present,
/// or `None` if this is a standard dgCMatrix group.
pub fn probe_bpcells_version(file: &File, group_path: &str) -> Option<String> {
    let grp = file.group(group_path).ok()?;
    let version = read_version_attr(&grp)?;
    if version.starts_with("packed-") || version.starts_with("unpacked-") {
        Some(version)
    } else {
        None
    }
}

/// Open a BPCells matrix stored inside an HDF5 group as a streaming
/// `BpcellsDatasetReader`.
///
/// `group_path` is an absolute path within the HDF5 file, e.g.
/// `"assays/RNA/counts"` or `"assays/RNA/layers/counts"`.
pub fn open_bpcells_h5(
    file: &File,
    group_path: &str,
    chunk_size: usize,
) -> Result<BpcellsDatasetReader> {
    let grp = file.group(group_path).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: can't open group '{group_path}': {e}"))
    })?;

    let version = read_version_attr(&grp).ok_or_else(|| {
        ScxError::InvalidFormat(format!(
            "BPCells HDF5: no 'version' attribute on group '{group_path}'"
        ))
    })?;

    // --- storage_order ---
    let order_vec = read_strings(&grp, "storage_order")?;
    let storage_order = match order_vec.first().map(String::as_str) {
        Some("col") => StorageOrder::Col,
        Some("row") => StorageOrder::Row,
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "BPCells HDF5: unknown storage_order: {other:?}"
            )))
        }
    };

    // --- shape: [nrow, ncol] ---
    let shape = read_u32s(&grp, "shape")?;
    if shape.len() < 2 {
        return Err(ScxError::InvalidFormat("BPCells HDF5: shape has < 2 elements".into()));
    }
    let (nrow, ncol) = (shape[0] as usize, shape[1] as usize);

    // --- idxptr (v2 = uint64; v1 = uint32) ---
    let idxptr: Vec<u64> = if version.ends_with("-v2") {
        read_u64s(&grp, "idxptr")?
    } else {
        // v1 stores idxptr as uint32
        read_u32s(&grp, "idxptr")?.into_iter().map(|v| v as u64).collect()
    };
    let total_nnz = *idxptr.last().unwrap_or(&0) as usize;

    // --- names ---
    let row_names = read_strings(&grp, "row_names").unwrap_or_default();
    let col_names = read_strings(&grp, "col_names").unwrap_or_default();

    // --- index + values depending on version ---
    let (index, values) = match version.as_str() {
        v if v.starts_with("packed-uint-") => {
            let idx_data   = read_u32s(&grp, "index_data")?;
            let idx_idx    = read_u32s(&grp, "index_idx")?;
            let idx_starts = read_u32s(&grp, "index_starts")?;
            let val_data   = read_u32s(&grp, "val_data")?;
            let val_idx    = read_u32s(&grp, "val_idx")?;
            let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
            let vals  = decode_for(&val_data, &val_idx, total_nnz);
            (index, ValStore::Uint32(vals))
        }
        v if v.starts_with("unpacked-uint-") => {
            let index = read_u32s(&grp, "index")?;
            let vals  = read_u32s(&grp, "val")?;
            (index, ValStore::Uint32(vals))
        }
        v if v.starts_with("packed-float-") => {
            let idx_data   = read_u32s(&grp, "index_data")?;
            let idx_idx    = read_u32s(&grp, "index_idx")?;
            let idx_starts = read_u32s(&grp, "index_starts")?;
            let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
            let vals  = read_f32s(&grp, "val")?;
            (index, ValStore::Float32(vals))
        }
        v if v.starts_with("packed-double-") => {
            let idx_data   = read_u32s(&grp, "index_data")?;
            let idx_idx    = read_u32s(&grp, "index_idx")?;
            let idx_starts = read_u32s(&grp, "index_starts")?;
            let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
            let vals  = read_f64s(&grp, "val")?;
            (index, ValStore::Float64(vals))
        }
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "BPCells HDF5: unsupported version '{other}'"
            )))
        }
    };

    // --- derive n_obs / n_vars / obs_names / var_names from storage_order ---
    let (n_obs, n_vars, obs_names, var_names) = match storage_order {
        StorageOrder::Col => (ncol, nrow, col_names, row_names),
        StorageOrder::Row => (nrow, ncol, row_names, col_names),
    };

    let dtype = match &values {
        ValStore::Uint32(_)  => DataType::U32,
        ValStore::Float32(_) => DataType::F32,
        ValStore::Float64(_) => DataType::F64,
    };

    Ok(BpcellsDatasetReader::from_parts(
        n_obs, n_vars, chunk_size, obs_names, var_names, idxptr, index, values, dtype,
    ))
}
