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

use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use hdf5::types::{VarLenAscii, VarLenUnicode};
use hdf5::{File, Group};
use ndarray::Array1;

use crate::bpcells::{
    decode_d1z, decode_for, encode_d1z, encode_for, BpcellsDatasetReader, StorageOrder, ValStore,
};
use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable, VarTable,
    Varm,
};
use crate::stream::DatasetWriter;
use ndarray::Array2;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Read a 1-D uint32 dataset from an HDF5 group.
fn read_u32s(grp: &Group, name: &str) -> Result<Vec<u32>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<u32> = ds
        .read_1d()
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}")))?;
    Ok(arr.to_vec())
}

/// Read a 1-D uint64 dataset from an HDF5 group.
fn read_u64s(grp: &Group, name: &str) -> Result<Vec<u64>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<u64> = ds
        .read_1d()
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}")))?;
    Ok(arr.to_vec())
}

/// Read a 1-D float32 dataset.
fn read_f32s(grp: &Group, name: &str) -> Result<Vec<f32>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<f32> = ds
        .read_1d()
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}")))?;
    Ok(arr.to_vec())
}

/// Read a 1-D float64 dataset.
fn read_f64s(grp: &Group, name: &str) -> Result<Vec<f64>> {
    let ds = grp.dataset(name).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: missing dataset '{name}': {e}"))
    })?;
    let arr: Array1<f64> = ds
        .read_1d()
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: reading '{name}': {e}")))?;
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
            let arr: Array1<VarLenAscii> = ds
                .read_1d()
                .map_err(|e| ScxError::InvalidFormat(format!("reading '{name}': {e}")))?;
            Ok(arr.into_iter().map(|s| s.to_string()).collect())
        }
        _ => {
            let arr: Array1<VarLenUnicode> = ds
                .read_1d()
                .map_err(|e| ScxError::InvalidFormat(format!("reading '{name}': {e}")))?;
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

fn seurat_write_strings_local(grp: &Group, name: &str, strings: &[String]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!(
                "BPCells HDF5: removing existing dataset '{name}': {e}"
            ))
        })?;
    }
    let vals: Vec<VarLenUnicode> = strings
        .iter()
        .map(|s| {
            <VarLenUnicode as std::str::FromStr>::from_str(s).map_err(|e| {
                ScxError::InvalidFormat(format!(
                    "BPCells HDF5: invalid UTF-8 string for '{name}': {e}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ds = grp
        .new_dataset::<VarLenUnicode>()
        .shape(vals.len())
        .create(name)?;
    ds.write(&Array1::from_vec(vals))?;
    Ok(())
}

fn seurat_write_col_local(grp: &Group, name: &str, data: &ColumnData) -> Result<()> {
    match data {
        ColumnData::Float(v) => {
            let ds = grp.new_dataset::<f64>().shape(v.len()).create(name)?;
            ds.write(&Array1::from_vec(v.clone()))?;
        }
        ColumnData::Int(v) => {
            let ds = grp.new_dataset::<i32>().shape(v.len()).create(name)?;
            ds.write(&Array1::from_vec(v.clone()))?;
        }
        ColumnData::Bool(v) => {
            let vi: Vec<i32> = v.iter().map(|&b| b as i32).collect();
            let ds = grp.new_dataset::<i32>().shape(vi.len()).create(name)?;
            ds.write(&Array1::from_vec(vi))?;
        }
        ColumnData::String(v) => {
            seurat_write_strings_local(grp, name, v)?;
        }
        ColumnData::Categorical { codes, levels } => {
            let col_grp = grp.create_group(name)?;
            let values: Vec<i32> = codes.iter().map(|&c| c as i32 + 1).collect();
            let ds = col_grp
                .new_dataset::<i32>()
                .shape(values.len())
                .create("values")?;
            ds.write(&Array1::from_vec(values))?;
            seurat_write_strings_local(&col_grp, "levels", levels)?;
        }
    }
    Ok(())
}

fn seurat_write_meta_cols_local(grp: &Group, columns: &[Column]) -> Result<()> {
    let logical_names: Vec<VarLenUnicode> = columns
        .iter()
        .filter(|c| matches!(c.data, ColumnData::Bool(_)))
        .map(|c| <VarLenUnicode as std::str::FromStr>::from_str(&c.name).unwrap_or_default())
        .collect();
    if !logical_names.is_empty() {
        let attr = grp
            .new_attr::<VarLenUnicode>()
            .shape(logical_names.len())
            .create("logicals")?;
        attr.write(&Array1::from_vec(logical_names))?;
    }

    let colnames: Vec<VarLenUnicode> = columns
        .iter()
        .map(|c| <VarLenUnicode as std::str::FromStr>::from_str(&c.name).unwrap_or_default())
        .collect();
    let col_attr = grp
        .new_attr::<VarLenUnicode>()
        .shape(colnames.len())
        .create("colnames")?;
    col_attr.write(&Array1::from_vec(colnames))?;

    if grp.name() == "/meta.data" {
        let class_vals = vec![VarLenUnicode::from_str("data.frame").unwrap_or_default()];
        let class_attr = grp
            .new_attr::<VarLenUnicode>()
            .shape(class_vals.len())
            .create("_class")?;
        class_attr.write(&Array1::from_vec(class_vals))?;
    }

    for col in columns {
        seurat_write_col_local(grp, &col.name, &col.data)?;
    }
    Ok(())
}

fn seurat_write_json_value_local(
    parent: &Group,
    name: &str,
    value: &serde_json::Value,
) -> Result<()> {
    match value {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::Bool(b) => {
            let ds = parent.new_dataset::<i32>().shape(()).create(name)?;
            ds.write_scalar(&(*b as i32))?;
            Ok(())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let ds = parent.new_dataset::<i64>().shape(()).create(name)?;
                ds.write_scalar(&i)?;
            } else if let Some(u) = n.as_u64() {
                let ds = parent.new_dataset::<u64>().shape(()).create(name)?;
                ds.write_scalar(&u)?;
            } else if let Some(f) = n.as_f64() {
                let ds = parent.new_dataset::<f64>().shape(()).create(name)?;
                ds.write_scalar(&f)?;
            }
            Ok(())
        }
        serde_json::Value::String(s) => {
            let ds = parent
                .new_dataset::<VarLenUnicode>()
                .shape(1)
                .create(name)?;
            let vals = vec![VarLenUnicode::from_str(s).unwrap_or_default()];
            ds.write(&Array1::from_vec(vals))?;
            Ok(())
        }
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                let ds = parent.new_dataset::<f64>().shape(0).create(name)?;
                ds.write(&Array1::from_vec(Vec::<f64>::new()))?;
                return Ok(());
            }

            if arr
                .iter()
                .all(|v| matches!(v, serde_json::Value::Number(_)))
            {
                let vals: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                let ds = parent.new_dataset::<f64>().shape(vals.len()).create(name)?;
                ds.write(&Array1::from_vec(vals))?;
                return Ok(());
            }

            if arr
                .iter()
                .all(|v| matches!(v, serde_json::Value::String(_)))
            {
                let vals: Vec<VarLenUnicode> = arr
                    .iter()
                    .map(|v| {
                        VarLenUnicode::from_str(v.as_str().unwrap_or_default()).unwrap_or_default()
                    })
                    .collect();
                let ds = parent
                    .new_dataset::<VarLenUnicode>()
                    .shape(vals.len())
                    .create(name)?;
                ds.write(&Array1::from_vec(vals))?;
                return Ok(());
            }

            let grp = parent.create_group(name)?;
            for (i, elem) in arr.iter().enumerate() {
                seurat_write_json_value_local(&grp, &i.to_string(), elem)?;
            }
            Ok(())
        }
        serde_json::Value::Object(map) => {
            let grp = parent.create_group(name)?;
            for (k, v) in map {
                seurat_write_json_value_local(&grp, k, v)?;
            }
            Ok(())
        }
    }
}

fn seurat_write_uns_local(file: &File, uns: &UnsTable) -> Result<()> {
    if uns.raw.is_null() {
        return Ok(());
    }

    let misc = match file.group("misc") {
        Ok(g) => g,
        Err(_) => file.create_group("misc")?,
    };

    match &uns.raw {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                seurat_write_json_value_local(&misc, k, v)?;
            }
        }
        other => {
            seurat_write_json_value_local(&misc, "value", other)?;
        }
    }

    Ok(())
}

/// Write a 1-D uint32 dataset into an HDF5 group, replacing any existing one.
fn write_u32s(grp: &Group, name: &str, values: &[u32]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!(
                "BPCells HDF5: removing existing dataset '{name}': {e}"
            ))
        })?;
    }
    grp.new_dataset_builder()
        .with_data(values)
        .create(name)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating '{name}': {e}")))?;
    Ok(())
}

/// Write a 1-D uint64 dataset into an HDF5 group, replacing any existing one.
fn write_u64s(grp: &Group, name: &str, values: &[u64]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!(
                "BPCells HDF5: removing existing dataset '{name}': {e}"
            ))
        })?;
    }
    grp.new_dataset_builder()
        .with_data(values)
        .create(name)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating '{name}': {e}")))?;
    Ok(())
}

/// Write a 1-D float32 dataset into an HDF5 group, replacing any existing one.
fn write_f32s(grp: &Group, name: &str, values: &[f32]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!(
                "BPCells HDF5: removing existing dataset '{name}': {e}"
            ))
        })?;
    }
    grp.new_dataset_builder()
        .with_data(values)
        .create(name)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating '{name}': {e}")))?;
    Ok(())
}

/// Write a 1-D float64 dataset into an HDF5 group, replacing any existing one.
fn write_f64s(grp: &Group, name: &str, values: &[f64]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!(
                "BPCells HDF5: removing existing dataset '{name}': {e}"
            ))
        })?;
    }
    grp.new_dataset_builder()
        .with_data(values)
        .create(name)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating '{name}': {e}")))?;
    Ok(())
}

/// Write a 1-D UTF-8 string dataset into an HDF5 group, replacing any existing one.
fn write_strings(grp: &Group, name: &str, values: &[String]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!(
                "BPCells HDF5: removing existing dataset '{name}': {e}"
            ))
        })?;
    }
    let vals: Vec<VarLenUnicode> = values
        .iter()
        .map(|s| {
            <VarLenUnicode as std::str::FromStr>::from_str(s).map_err(|e| {
                ScxError::InvalidFormat(format!(
                    "BPCells HDF5: invalid UTF-8 string for '{name}': {e}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    grp.new_dataset_builder()
        .with_data(&vals)
        .create(name)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating '{name}': {e}")))?;
    Ok(())
}

/// Write the scalar `version` attribute on a BPCells HDF5 group.
fn write_version_attr(grp: &Group, version: &str) -> Result<()> {
    if grp.attr("version").is_ok() {
        grp.attr("version")
            .and_then(|a| {
                a.write_scalar(
                    &<VarLenUnicode as std::str::FromStr>::from_str(version).map_err(|e| {
                        hdf5::Error::Internal(format!("invalid version string: {e}"))
                    })?,
                )
            })
            .map_err(|e| {
                ScxError::InvalidFormat(format!("BPCells HDF5: writing 'version' attr: {e}"))
            })?;
        return Ok(());
    }

    let v = <VarLenUnicode as std::str::FromStr>::from_str(version).map_err(|e| {
        ScxError::InvalidFormat(format!("BPCells HDF5: invalid version string: {e}"))
    })?;
    grp.new_attr::<VarLenUnicode>()
        .shape(())
        .create("version")
        .and_then(|a| a.write_scalar(&v))
        .map_err(|e| {
            ScxError::InvalidFormat(format!("BPCells HDF5: creating 'version' attr: {e}"))
        })?;
    Ok(())
}

/// Write a packed BPCells matrix into an HDF5 group.
///
/// BPCells groups columns into "runs" of 128. Each run's nonzeros are encoded
/// independently so chunk boundaries always align with run boundaries.
/// `index_idx_offsets[r]` and `val_idx_offsets[r]` mark where run r's slice
/// begins within the concatenated `index_idx` / `val_idx` arrays.
///
/// `shape` is stored as `[nrow, ncol]` in BPCells convention.
#[allow(clippy::too_many_arguments)]
pub fn write_bpcells_h5(
    file: &File,
    group_path: &str,
    storage_order: StorageOrder,
    nrow: usize,
    ncol: usize,
    row_names: &[String],
    col_names: &[String],
    idxptr: &[u64],
    index: &[u32],
    values: &ValStore,
) -> Result<()> {
    let grp = match file.group(group_path) {
        Ok(g) => g,
        Err(_) => file.create_group(group_path).map_err(|e| {
            ScxError::InvalidFormat(format!("BPCells HDF5: creating group '{group_path}': {e}"))
        })?,
    };

    let storage = match storage_order {
        StorageOrder::Col => "col".to_string(),
        StorageOrder::Row => "row".to_string(),
    };
    write_strings(&grp, "storage_order", &[storage])?;
    write_u32s(&grp, "shape", &[nrow as u32, ncol as u32])?;
    write_u64s(&grp, "idxptr", idxptr)?;
    write_strings(&grp, "row_names", row_names)?;
    write_strings(&grp, "col_names", col_names)?;

    let n_runs = ncol.div_ceil(128);

    // Encode indices and values per-run so chunk boundaries align with run
    // boundaries. This is required by BPCells: runs must be independently
    // decodable.
    let mut all_index_data: Vec<u32> = Vec::new();
    let mut all_index_idx: Vec<u32> = Vec::new();
    let mut all_index_starts: Vec<u32> = Vec::new();
    let mut index_idx_offsets: Vec<u64> = Vec::with_capacity(n_runs + 1);
    index_idx_offsets.push(0);

    // For Uint32 values only (floats have no compression index).
    let mut all_val_data: Vec<u32> = Vec::new();
    let mut all_val_idx: Vec<u32> = Vec::new();
    let mut val_idx_offsets: Vec<u64> = Vec::with_capacity(n_runs + 1);
    val_idx_offsets.push(0);

    for r in 0..n_runs {
        let col_start = r * 128;
        let col_end = ((r + 1) * 128).min(ncol);
        let nnz_start = idxptr[col_start] as usize;
        let nnz_end = idxptr[col_end] as usize;

        // Encode this run's row indices.
        let (run_idx_data, mut run_idx_idx, run_idx_starts) =
            encode_d1z(&index[nnz_start..nnz_end]);

        // Adjust run-local idx offsets to be global (offset by data written so far).
        let idx_data_offset = all_index_data.len() as u32;
        for v in &mut run_idx_idx {
            *v += idx_data_offset;
        }

        all_index_data.extend_from_slice(&run_idx_data);
        all_index_idx.extend_from_slice(&run_idx_idx); // includes per-run sentinel
        all_index_starts.extend_from_slice(&run_idx_starts);
        index_idx_offsets.push(all_index_idx.len() as u64);

        // Encode this run's values (Uint32 only; floats written flat later).
        if let ValStore::Uint32(v) = values {
            let (run_val_data, mut run_val_idx) = encode_for(&v[nnz_start..nnz_end]);
            let val_data_offset = all_val_data.len() as u32;
            for vv in &mut run_val_idx {
                *vv += val_data_offset;
            }
            all_val_data.extend_from_slice(&run_val_data);
            all_val_idx.extend_from_slice(&run_val_idx);
            val_idx_offsets.push(all_val_idx.len() as u64);
        }
    }

    write_u32s(&grp, "index_data", &all_index_data)?;
    write_u32s(&grp, "index_idx", &all_index_idx)?;
    write_u64s(&grp, "index_idx_offsets", &index_idx_offsets)?;
    write_u32s(&grp, "index_starts", &all_index_starts)?;

    match values {
        ValStore::Uint32(_) => {
            write_version_attr(&grp, "packed-uint-matrix-v2")?;
            write_u32s(&grp, "val_data", &all_val_data)?;
            write_u32s(&grp, "val_idx", &all_val_idx)?;
            write_u64s(&grp, "val_idx_offsets", &val_idx_offsets)?;
        }
        ValStore::Float32(v) => {
            write_version_attr(&grp, "packed-float-matrix-v2")?;
            write_f32s(&grp, "val", v)?;
        }
        ValStore::Float64(v) => {
            write_version_attr(&grp, "packed-double-matrix-v2")?;
            write_f64s(&grp, "val", v)?;
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
enum TypedVal {
    F32(f32),
    F64(f64),
    U32(u32),
}

impl TypedVal {
    fn from_typed_vec_at(values: &TypedVec, idx: usize) -> Result<Self> {
        match values {
            TypedVec::F32(v) => Ok(Self::F32(v[idx])),
            TypedVec::F64(v) => Ok(Self::F64(v[idx])),
            TypedVec::U32(v) => Ok(Self::U32(v[idx])),
            TypedVec::I32(_) => Err(ScxError::InvalidFormat(
                "BPCells writer does not support I32 matrices".into(),
            )),
        }
    }
}

#[derive(Debug)]
struct BpcellsCscData {
    idxptr: Vec<u64>,
    index: Vec<u32>,
    values: ValStore,
}

#[derive(Debug)]
struct BpcellsCscAccumulator {
    /// Number of outer-dimension elements (= n_obs for X/layers, = n_obs for obsp).
    /// This is the length of the idxptr array minus 1.
    n_outer: usize,
    /// (obs_idx, var_idx, val) — sorted by (obs, var) to produce obs-indexed CSC.
    entries: Vec<(u32, u32, TypedVal)>,
}

impl BpcellsCscAccumulator {
    fn new(n_outer: usize) -> Self {
        Self {
            n_outer,
            entries: Vec::new(),
        }
    }

    fn push_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let csr = &chunk.data;
        if csr.indices.len() != csr.data.len() {
            return Err(ScxError::InvalidFormat(
                "BPCells writer: CSR indices/data length mismatch".into(),
            ));
        }

        for local_row in 0..chunk.nrows {
            let global_row = chunk.row_offset + local_row;
            let start = csr.indptr[local_row] as usize;
            let end = csr.indptr[local_row + 1] as usize;
            for ptr in start..end {
                let col = csr.indices[ptr];
                let val = TypedVal::from_typed_vec_at(&csr.data, ptr)?;
                // outer = obs (cell), inner = var (gene) — matches BPCells CSC convention
                self.entries.push((global_row as u32, col, val));
            }
        }

        Ok(())
    }

    fn into_csc(mut self) -> Result<BpcellsCscData> {
        // Sort by (obs, var) so entries are grouped by obs — producing obs-indexed CSC.
        self.entries
            .sort_unstable_by_key(|(obs, var, _)| (*obs, *var));

        let mut idxptr = vec![0u64; self.n_outer + 1];
        let mut index = Vec::with_capacity(self.entries.len());

        let first_kind = self.entries.first().map(|e| match e.2 {
            TypedVal::F32(_) => DataType::F32,
            TypedVal::F64(_) => DataType::F64,
            TypedVal::U32(_) => DataType::U32,
        });

        match first_kind {
            None => Ok(BpcellsCscData {
                idxptr,
                index,
                values: ValStore::Uint32(Vec::new()),
            }),
            Some(DataType::F32) => {
                let mut vals = Vec::with_capacity(self.entries.len());
                for (obs, var, v) in self.entries {
                    idxptr[obs as usize + 1] += 1;
                    index.push(var);
                    match v {
                        TypedVal::F32(x) => vals.push(x),
                        _ => {
                            return Err(ScxError::InvalidFormat(
                                "BPCells writer: mixed value types in accumulator".into(),
                            ))
                        }
                    }
                }
                for i in 0..self.n_outer {
                    idxptr[i + 1] += idxptr[i];
                }
                Ok(BpcellsCscData {
                    idxptr,
                    index,
                    values: ValStore::Float32(vals),
                })
            }
            Some(DataType::F64) => {
                let mut vals = Vec::with_capacity(self.entries.len());
                for (obs, var, v) in self.entries {
                    idxptr[obs as usize + 1] += 1;
                    index.push(var);
                    match v {
                        TypedVal::F64(x) => vals.push(x),
                        _ => {
                            return Err(ScxError::InvalidFormat(
                                "BPCells writer: mixed value types in accumulator".into(),
                            ))
                        }
                    }
                }
                for i in 0..self.n_outer {
                    idxptr[i + 1] += idxptr[i];
                }
                Ok(BpcellsCscData {
                    idxptr,
                    index,
                    values: ValStore::Float64(vals),
                })
            }
            Some(DataType::U32) => {
                let mut vals = Vec::with_capacity(self.entries.len());
                for (obs, var, v) in self.entries {
                    idxptr[obs as usize + 1] += 1;
                    index.push(var);
                    match v {
                        TypedVal::U32(x) => vals.push(x),
                        _ => {
                            return Err(ScxError::InvalidFormat(
                                "BPCells writer: mixed value types in accumulator".into(),
                            ))
                        }
                    }
                }
                for i in 0..self.n_outer {
                    idxptr[i + 1] += idxptr[i];
                }
                Ok(BpcellsCscData {
                    idxptr,
                    index,
                    values: ValStore::Uint32(vals),
                })
            }
            Some(DataType::I32) => Err(ScxError::InvalidFormat(
                "BPCells writer does not support I32 matrices".into(),
            )),
        }
    }
}

struct PendingSparseMatrix {
    group_prefix: String,
    name: String,
    shape: (usize, usize),
    accumulator: BpcellsCscAccumulator,
}

pub struct BpcellsH5Writer {
    file: File,
    assay: String,
    layer: String,
    n_obs: usize,
    n_vars: usize,
    obs: Option<ObsTable>,
    var: Option<VarTable>,
    obsm: Option<Embeddings>,
    uns: Option<UnsTable>,
    varm: Option<Varm>,
    accumulator: BpcellsCscAccumulator,
    sparse_state: Option<PendingSparseMatrix>,
}

impl BpcellsH5Writer {
    #[allow(clippy::too_many_arguments)]
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        _dtype: DataType,
        assay: Option<&str>,
        layer: Option<&str>,
        project: Option<&str>,
        seuratdisk_compat: bool,
    ) -> Result<Self> {
        let assay = assay.unwrap_or("RNA").to_string();
        let layer = layer.unwrap_or("counts").to_string();
        let project = project.unwrap_or("SeuratProject");
        let file = File::create(path.as_ref())?;

        if seuratdisk_compat {
            let root = file.group("/")?;
            for (name, value) in [
                ("version", "3.1.5.9900"),
                ("active.assay", assay.as_str()),
                ("project", project),
            ] {
                let v = VarLenUnicode::from_str(value).unwrap_or_default();
                root.new_attr::<VarLenUnicode>()
                    .create(name)?
                    .write_scalar(&v)?;
            }
            for grp in &[
                "commands",
                "graphs",
                "images",
                "misc",
                "neighbors",
                "reductions",
                "tools",
            ] {
                file.create_group(grp)?;
            }
        }

        if file.group("assays").is_err() {
            file.create_group("assays")?;
        }
        let assay_grp = if file.group(&format!("assays/{assay}")).is_err() {
            file.create_group(&format!("assays/{assay}"))?
        } else {
            file.group(&format!("assays/{assay}"))?
        };
        if seuratdisk_compat {
            let key =
                VarLenUnicode::from_str(&format!("{}_", assay.to_lowercase())).unwrap_or_default();
            assay_grp
                .new_attr::<VarLenUnicode>()
                .create("key")?
                .write_scalar(&key)?;
        }

        Ok(Self {
            file,
            assay,
            layer,
            n_obs,
            n_vars,
            obs: None,
            var: None,
            obsm: None,
            uns: None,
            varm: None,
            accumulator: BpcellsCscAccumulator::new(n_obs),
            sparse_state: None,
        })
    }
}

#[async_trait]
impl DatasetWriter for BpcellsH5Writer {
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()> {
        self.obs = Some(obs.clone());
        Ok(())
    }

    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        self.var = Some(var.clone());
        Ok(())
    }

    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()> {
        self.obsm = Some(obsm.clone());
        Ok(())
    }

    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()> {
        self.uns = Some(uns.clone());
        Ok(())
    }

    async fn write_varm(&mut self, varm: &Varm) -> Result<()> {
        self.varm = Some(varm.clone());
        Ok(())
    }

    async fn begin_sparse(
        &mut self,
        group_prefix: &str,
        name: &str,
        meta: &SparseMatrixMeta,
    ) -> Result<()> {
        if self.sparse_state.is_some() {
            return Err(ScxError::InvalidFormat(
                "BPCells writer: begin_sparse called while another sparse matrix is open".into(),
            ));
        }
        if group_prefix != "layers" && group_prefix != "obsp" {
            return Err(ScxError::InvalidFormat(
                "BPCells writer does not yet support auxiliary sparse matrices outside assay layers and obsp".into(),
            ));
        }

        self.sparse_state = Some(PendingSparseMatrix {
            group_prefix: group_prefix.to_string(),
            name: name.to_string(),
            shape: meta.shape,
            accumulator: BpcellsCscAccumulator::new(meta.shape.0),
        });
        Ok(())
    }

    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let state = self.sparse_state.as_mut().ok_or_else(|| {
            ScxError::InvalidFormat(
                "BPCells writer: write_sparse_chunk called without begin_sparse".into(),
            )
        })?;
        state.accumulator.push_chunk(chunk)
    }

    async fn end_sparse(&mut self) -> Result<()> {
        let state = self.sparse_state.take().ok_or_else(|| {
            ScxError::InvalidFormat("BPCells writer: end_sparse called without begin_sparse".into())
        })?;

        let csc = state.accumulator.into_csc()?;
        let obs = self.obs.as_ref().ok_or_else(|| {
            ScxError::InvalidFormat("BPCells writer end_sparse called before write_obs".into())
        })?;
        let var = self.var.as_ref().ok_or_else(|| {
            ScxError::InvalidFormat("BPCells writer end_sparse called before write_var".into())
        })?;

        let (group_path, nrow, ncol, row_names, col_names) = match state.group_prefix.as_str() {
            "layers" => (
                format!("assays/{}/{}", self.assay, state.name),
                state.shape.1,
                state.shape.0,
                var.index.clone(),
                obs.index.clone(),
            ),
            "obsp" => (
                format!("graphs/{}", state.name),
                state.shape.1,
                state.shape.0,
                obs.index.clone(),
                obs.index.clone(),
            ),
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "BPCells writer: unsupported sparse group prefix '{other}'"
                )))
            }
        };

        write_bpcells_h5(
            &self.file,
            &group_path,
            StorageOrder::Col,
            nrow,
            ncol,
            &row_names,
            &col_names,
            &csc.idxptr,
            &csc.index,
            &csc.values,
        )?;
        Ok(())
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        self.accumulator.push_chunk(chunk)
    }

    async fn finalize(&mut self) -> Result<()> {
        let accumulator = std::mem::replace(
            &mut self.accumulator,
            BpcellsCscAccumulator::new(self.n_obs),
        );
        let csc = accumulator.into_csc()?;
        let group_path = format!("assays/{}/{}", self.assay, self.layer);

        let obs = self.obs.as_ref().ok_or_else(|| {
            ScxError::InvalidFormat("BPCells writer finalize called before write_obs".into())
        })?;
        let var = self.var.as_ref().ok_or_else(|| {
            ScxError::InvalidFormat("BPCells writer finalize called before write_var".into())
        })?;

        let root = self.file.group("/")?;
        seurat_write_strings_local(&root, "cell.names", &obs.index)?;
        let meta_grp = match self.file.group("meta.data") {
            Ok(g) => g,
            Err(_) => self.file.create_group("meta.data")?,
        };
        seurat_write_meta_cols_local(&meta_grp, &obs.columns)?;

        let assay_grp = self.file.group(&format!("assays/{}", self.assay))?;
        seurat_write_strings_local(&assay_grp, "features", &var.index)?;
        if !var.columns.is_empty() {
            let mf_grp = match assay_grp.group("meta.features") {
                Ok(g) => g,
                Err(_) => assay_grp.create_group("meta.features")?,
            };
            seurat_write_meta_cols_local(&mf_grp, &var.columns)?;
        }

        // Always create reductions — SeuratDisk requires it even when empty.
        let reds_grp = Some(match self.file.group("reductions") {
            Ok(g) => g,
            Err(_) => self.file.create_group("reductions")?,
        });

        if let (Some(reds_grp), Some(obsm)) = (&reds_grp, &self.obsm) {
            for (key, mat) in &obsm.map {
                let red_name = key.strip_prefix("X_").unwrap_or(key.as_str());
                let red_grp = match reds_grp.group(red_name) {
                    Ok(g) => g,
                    Err(_) => reds_grp.create_group(red_name)?,
                };

                let (n_obs, n_comps) = mat.shape;
                let arr_t = Array2::from_shape_vec((n_obs, n_comps), mat.data.clone())
                    .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
                let ds = red_grp
                    .new_dataset::<f64>()
                    .shape((n_obs, n_comps))
                    .create("cell.embeddings")?;
                ds.write(&arr_t)?;

                let key = if red_name.eq_ignore_ascii_case("pca") {
                    "PC_"
                } else if red_name.eq_ignore_ascii_case("umap") {
                    "UMAP_"
                } else {
                    ""
                };
                let assay_attr = vec![VarLenUnicode::from_str(&self.assay).unwrap_or_default()];
                let key_attr = vec![VarLenUnicode::from_str(key).unwrap_or_default()];
                let attr = red_grp
                    .new_attr::<VarLenUnicode>()
                    .shape(assay_attr.len())
                    .create("active.assay")?;
                attr.write(&Array1::from_vec(assay_attr))?;
                let attr = red_grp
                    .new_attr::<VarLenUnicode>()
                    .shape(key_attr.len())
                    .create("key")?;
                attr.write(&Array1::from_vec(key_attr))?;
            }
        }

        if let (Some(reds_grp), Some(varm)) = (&reds_grp, &self.varm) {
            for (key, mat) in &varm.map {
                let red_name = key.strip_prefix("X_").unwrap_or(key.as_str());
                let red_grp = match reds_grp.group(red_name) {
                    Ok(g) => g,
                    Err(_) => reds_grp.create_group(red_name)?,
                };

                let (n_vars, k) = mat.shape;
                let mut buf = vec![0.0f64; n_vars * k];
                for i in 0..n_vars {
                    for j in 0..k {
                        buf[j * n_vars + i] = mat.data[i * k + j];
                    }
                }
                let arr_t = Array2::from_shape_vec((k, n_vars), buf)
                    .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
                let ds = red_grp
                    .new_dataset::<f64>()
                    .shape((k, n_vars))
                    .create("feature.loadings")?;
                ds.write(&arr_t)?;

                if red_grp.attr("active.assay").is_err() {
                    let assay_attr = vec![VarLenUnicode::from_str(&self.assay).unwrap_or_default()];
                    let attr = red_grp
                        .new_attr::<VarLenUnicode>()
                        .shape(assay_attr.len())
                        .create("active.assay")?;
                    attr.write(&Array1::from_vec(assay_attr))?;
                }
                if red_grp.attr("key").is_err() {
                    let key_attr = vec![VarLenUnicode::from_str("").unwrap_or_default()];
                    let attr = red_grp
                        .new_attr::<VarLenUnicode>()
                        .shape(key_attr.len())
                        .create("key")?;
                    attr.write(&Array1::from_vec(key_attr))?;
                }
            }
        }

        if let Some(uns) = &self.uns {
            seurat_write_uns_local(&self.file, uns)?;
        }

        write_bpcells_h5(
            &self.file,
            &group_path,
            StorageOrder::Col,
            self.n_vars,
            self.n_obs,
            &var.index,
            &obs.index,
            &csc.idxptr,
            &csc.index,
            &csc.values,
        )?;
        self.file.flush()?;
        Ok(())
    }
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
        ScxError::InvalidFormat(format!(
            "BPCells HDF5: can't open group '{group_path}': {e}"
        ))
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
        return Err(ScxError::InvalidFormat(
            "BPCells HDF5: shape has < 2 elements".into(),
        ));
    }
    let (nrow, ncol) = (shape[0] as usize, shape[1] as usize);

    // --- idxptr (v2 = uint64; v1 = uint32) ---
    let idxptr: Vec<u64> = if version.ends_with("-v2") {
        read_u64s(&grp, "idxptr")?
    } else {
        // v1 stores idxptr as uint32
        read_u32s(&grp, "idxptr")?
            .into_iter()
            .map(|v| v as u64)
            .collect()
    };
    let total_nnz = *idxptr.last().unwrap_or(&0) as usize;

    // --- names ---
    let row_names = read_strings(&grp, "row_names").unwrap_or_default();
    let col_names = read_strings(&grp, "col_names").unwrap_or_default();

    // --- index + values depending on version ---
    let (index, values) = match version.as_str() {
        v if v.starts_with("packed-uint-") => {
            let idx_data = read_u32s(&grp, "index_data")?;
            let idx_idx = read_u32s(&grp, "index_idx")?;
            let idx_starts = read_u32s(&grp, "index_starts")?;
            let val_data = read_u32s(&grp, "val_data")?;
            let val_idx = read_u32s(&grp, "val_idx")?;
            let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
            let vals = decode_for(&val_data, &val_idx, total_nnz);
            (index, ValStore::Uint32(vals))
        }
        v if v.starts_with("unpacked-uint-") => {
            let index = read_u32s(&grp, "index")?;
            let vals = read_u32s(&grp, "val")?;
            (index, ValStore::Uint32(vals))
        }
        v if v.starts_with("packed-float-") => {
            let idx_data = read_u32s(&grp, "index_data")?;
            let idx_idx = read_u32s(&grp, "index_idx")?;
            let idx_starts = read_u32s(&grp, "index_starts")?;
            let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
            let vals = read_f32s(&grp, "val")?;
            (index, ValStore::Float32(vals))
        }
        v if v.starts_with("packed-double-") => {
            let idx_data = read_u32s(&grp, "index_data")?;
            let idx_idx = read_u32s(&grp, "index_idx")?;
            let idx_starts = read_u32s(&grp, "index_starts")?;
            let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
            let vals = read_f64s(&grp, "val")?;
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
        ValStore::Uint32(_) => DataType::U32,
        ValStore::Float32(_) => DataType::F32,
        ValStore::Float64(_) => DataType::F64,
    };

    Ok(BpcellsDatasetReader::from_parts(
        n_obs, n_vars, chunk_size, obs_names, var_names, idxptr, index, values, dtype,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpcells::{bp128_pack, bp128_unpack};

    #[test]
    fn bp128_pack_roundtrip_all_bit_widths() {
        for b in 0u8..=32 {
            let mask = if b == 32 {
                u32::MAX
            } else if b == 0 {
                0
            } else {
                (1u32 << b) - 1
            };
            let mut vals = [0u32; 128];
            for (i, v) in vals.iter_mut().enumerate() {
                let base = ((i as u32).wrapping_mul(2654435761)).rotate_left((i % 31) as u32);
                *v = base & mask;
            }
            let packed = bp128_pack(b, &vals);
            let unpacked = bp128_unpack(b, &packed);
            assert_eq!(unpacked, vals, "failed roundtrip for b={b}");
        }
    }

    #[test]
    fn encode_for_roundtrip_various_lengths() {
        for len in [1usize, 127, 128, 129, 256, 10_000] {
            let values: Vec<u32> = (0..len).map(|i| ((i as u32) % 97) + 1).collect();
            let (data, idx) = encode_for(&values);
            let decoded = decode_for(&data, &idx, values.len());
            assert_eq!(decoded, values, "failed FOR roundtrip for len={len}");
        }
    }

    #[test]
    fn encode_d1z_roundtrip_various_lengths() {
        for len in [1usize, 127, 128, 129, 256, 10_000] {
            let mut cur = 0u32;
            let values: Vec<u32> = (0..len)
                .map(|i| {
                    cur = cur.wrapping_add(((i % 5) as u32) + 1);
                    cur
                })
                .collect();
            let (data, idx, starts) = encode_d1z(&values);
            let decoded = decode_d1z(&data, &idx, &starts, values.len());
            assert_eq!(decoded, values, "failed D1Z roundtrip for len={len}");
        }
    }

    #[test]
    fn write_then_open_packed_uint_h5_group() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_bpcells_uint.h5");
        let file = File::create(&path).unwrap();

        let row_names = vec!["g1".to_string(), "g2".to_string(), "g3".to_string()];
        let col_names = vec!["c1".to_string(), "c2".to_string()];
        let idxptr = vec![0u64, 2, 4];
        let index = vec![0u32, 2, 1, 2];
        let values = ValStore::Uint32(vec![5u32, 7, 11, 13]);

        write_bpcells_h5(
            &file,
            "matrix",
            StorageOrder::Col,
            3,
            2,
            &row_names,
            &col_names,
            &idxptr,
            &index,
            &values,
        )
        .unwrap();

        let reopened = open_bpcells_h5(&file, "matrix", 2).unwrap();
        assert_eq!(reopened.n_obs, 2);
        assert_eq!(reopened.n_vars, 3);
        assert_eq!(
            read_version_attr(&file.group("matrix").unwrap()).as_deref(),
            Some("packed-uint-matrix-v2")
        );
    }

    #[test]
    fn write_then_open_packed_float_h5_group() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_bpcells_float.h5");
        let file = File::create(&path).unwrap();

        let row_names = vec!["g1".to_string(), "g2".to_string()];
        let col_names = vec!["c1".to_string(), "c2".to_string(), "c3".to_string()];
        let idxptr = vec![0u64, 1, 2, 3];
        let index = vec![0u32, 1, 0];
        let values = ValStore::Float32(vec![1.5f32, 2.5, 3.5]);

        write_bpcells_h5(
            &file,
            "matrix",
            StorageOrder::Col,
            2,
            3,
            &row_names,
            &col_names,
            &idxptr,
            &index,
            &values,
        )
        .unwrap();

        let reopened = open_bpcells_h5(&file, "matrix", 2).unwrap();
        assert_eq!(reopened.n_obs, 3);
        assert_eq!(reopened.n_vars, 2);
        assert_eq!(
            read_version_attr(&file.group("matrix").unwrap()).as_deref(),
            Some("packed-float-matrix-v2")
        );
    }

    #[tokio::test]
    async fn bpcells_writer_preserves_obsm_varm_uns_layers_and_obsp_for_h5seurat_reader() {
        use crate::dtype::TypedVec;
        use crate::h5seurat::H5SeuratReader;
        use crate::ir::{
            DenseMatrix, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta, UnsTable,
            VarTable, Varm,
        };
        use crate::stream::{DatasetReader, DatasetWriter};
        use futures::StreamExt;

        let n_obs = 3usize;
        let n_vars = 4usize;

        let obs = ObsTable {
            index: vec!["c0".into(), "c1".into(), "c2".into()],
            columns: vec![],
        };
        let var = VarTable {
            index: vec!["g0".into(), "g1".into(), "g2".into(), "g3".into()],
            columns: vec![],
        };

        let mut obsm = crate::ir::Embeddings::default();
        obsm.map.insert(
            "X_pca".into(),
            DenseMatrix {
                shape: (n_obs, 2),
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let mut varm = Varm::default();
        let varm_mat = DenseMatrix {
            shape: (n_vars, 2),
            data: vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        };
        varm.map.insert("X_pca".into(), varm_mat.clone());

        let uns = UnsTable {
            raw: serde_json::json!({
                "weights": [0.1, 0.2, 0.3],
                "title": "bp test",
                "nested": {
                    "alpha": 7,
                    "beta": [1.0, 2.0]
                }
            }),
        };

        let x_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr: vec![0, 2, 3, 5],
                indices: vec![0, 2, 1, 0, 3],
                data: TypedVec::U32(vec![1, 2, 3, 4, 5]),
            },
        };

        let layer_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr: vec![0, 1, 3, 4],
                indices: vec![1, 0, 2, 3],
                data: TypedVec::U32(vec![9, 8, 7, 6]),
            },
        };
        let layer_meta = SparseMatrixMeta {
            name: "data".into(),
            shape: (n_obs, n_vars),
            indptr: layer_chunk.data.indptr.clone(),
        };

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer =
            BpcellsH5Writer::create(&path, n_obs, n_vars, DataType::U32, None, None, None, false)
                .unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&obsm).await.unwrap();
        writer.write_uns(&uns).await.unwrap();
        writer.write_varm(&varm).await.unwrap();
        let obsp_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_obs),
                indptr: vec![0, 1, 2, 3],
                indices: vec![1, 2, 0],
                data: TypedVec::U32(vec![21, 22, 23]),
            },
        };
        let obsp_meta = SparseMatrixMeta {
            name: "knn".into(),
            shape: (n_obs, n_obs),
            indptr: obsp_chunk.data.indptr.clone(),
        };

        writer
            .begin_sparse("layers", "data", &layer_meta)
            .await
            .unwrap();
        writer.write_sparse_chunk(&layer_chunk).await.unwrap();
        writer.end_sparse().await.unwrap();
        writer
            .begin_sparse("obsp", "knn", &obsp_meta)
            .await
            .unwrap();
        writer.write_sparse_chunk(&obsp_chunk).await.unwrap();
        writer.end_sparse().await.unwrap();
        writer.write_x_chunk(&x_chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        let mut reader = H5SeuratReader::open(&path, 2, None, None).unwrap();

        let rt_obsm = reader.obsm().await.unwrap();
        assert!(rt_obsm.map.contains_key("X_pca"), "obsm['X_pca'] missing");
        let rt_pca = &rt_obsm.map["X_pca"];
        assert_eq!(rt_pca.shape, (n_obs, 2));
        for (a, b) in rt_pca.data.iter().zip(obsm.map["X_pca"].data.iter()) {
            assert!((a - b).abs() < 1e-10, "obsm data mismatch: {a} vs {b}");
        }

        let rt_varm = reader.varm().await.unwrap();
        assert!(rt_varm.map.contains_key("X_pca"), "varm['X_pca'] missing");
        let rt_loadings = &rt_varm.map["X_pca"];
        assert_eq!(rt_loadings.shape, varm_mat.shape);
        for (a, b) in rt_loadings.data.iter().zip(varm_mat.data.iter()) {
            assert!((a - b).abs() < 1e-10, "varm data mismatch: {a} vs {b}");
        }

        let rt_uns = reader.uns().await.unwrap();
        assert!(rt_uns.raw.is_object(), "uns.raw should be a JSON object");
        assert_eq!(rt_uns.raw["weights"], serde_json::json!([0.1, 0.2, 0.3]));
        assert_eq!(rt_uns.raw["title"], serde_json::json!("bp test"));
        assert_eq!(rt_uns.raw["nested"]["alpha"], serde_json::json!(7));
        assert_eq!(rt_uns.raw["nested"]["beta"], serde_json::json!([1.0, 2.0]));

        let layer_metas = reader.layer_metas().await.unwrap();
        let lm = layer_metas
            .iter()
            .find(|m| m.name == "data")
            .expect("layers['data'] missing");
        assert_eq!(lm.shape, layer_meta.shape);

        let seen_nnz = {
            let mut stream = reader.layer_stream(lm, 10);
            let mut seen_nnz = 0usize;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.unwrap();
                seen_nnz += chunk.data.indices.len();
                match chunk.data.data {
                    TypedVec::U32(_) => {}
                    other => panic!("unexpected layer dtype: {:?}", other),
                }
            }
            seen_nnz
        };
        assert!(seen_nnz > 0, "layer stream should yield non-empty chunks");

        let obsp_metas = reader.obsp_metas().await.unwrap();
        let om = obsp_metas
            .iter()
            .find(|m| m.name == "knn")
            .expect("obsp['knn'] missing");
        assert_eq!(om.shape, obsp_meta.shape);

        let mut stream = reader.obsp_stream(om, 10);
        let mut seen_nnz = 0usize;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            seen_nnz += chunk.data.indices.len();
            match chunk.data.data {
                TypedVec::U32(_) => {}
                other => panic!("unexpected obsp dtype: {:?}", other),
            }
        }
        assert!(seen_nnz > 0, "obsp stream should yield non-empty chunks");
    }

    /// Write a known 3×4 CSR matrix via BpcellsH5Writer, read it back via
    /// H5SeuratReader, and assert the streamed values are bit-exact.
    ///
    /// Matrix (obs × vars, row-major):
    ///   row 0: [1, 0, 2, 0]
    ///   row 1: [0, 3, 0, 0]
    ///   row 2: [4, 0, 0, 5]
    #[tokio::test]
    async fn bpcells_writer_x_stream_exact_values() {
        use crate::dtype::TypedVec;
        use crate::h5seurat::H5SeuratReader;
        use crate::ir::{MatrixChunk, ObsTable, SparseMatrixCSR, VarTable};
        use crate::stream::{DatasetReader, DatasetWriter};
        use futures::StreamExt;

        let n_obs = 3usize;
        let n_vars = 4usize;

        let obs = ObsTable {
            index: vec!["c0".into(), "c1".into(), "c2".into()],
            columns: vec![],
        };
        let var = VarTable {
            index: vec!["g0".into(), "g1".into(), "g2".into(), "g3".into()],
            columns: vec![],
        };

        // CSR: row 0→(col0=1,col2=2), row1→(col1=3), row2→(col0=4,col3=5)
        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr: vec![0, 2, 3, 5],
                indices: vec![0, 2, 1, 0, 3],
                data: TypedVec::U32(vec![1, 2, 3, 4, 5]),
            },
        };

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer =
            BpcellsH5Writer::create(&path, n_obs, n_vars, DataType::U32, None, None, None, false)
                .unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        // Expected dense [obs × vars], row-major
        let expected: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0];

        let mut reader = H5SeuratReader::open(&path, 1000, None, None).unwrap();
        let (rt_n_obs, rt_n_vars) = reader.shape();
        assert_eq!(rt_n_obs, n_obs, "n_obs mismatch");
        assert_eq!(rt_n_vars, n_vars, "n_vars mismatch");

        let mut dense = vec![0.0f64; n_obs * n_vars];
        let mut stream = reader.x_stream();
        while let Some(c) = stream.next().await {
            let c = c.unwrap();
            let csr = &c.data;
            for row in 0..c.nrows {
                let obs_i = c.row_offset + row;
                for k in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
                    let var_i = csr.indices[k] as usize;
                    dense[obs_i * n_vars + var_i] = match &csr.data {
                        TypedVec::U32(v) => v[k] as f64,
                        TypedVec::F32(v) => v[k] as f64,
                        TypedVec::F64(v) => v[k],
                        TypedVec::I32(v) => v[k] as f64,
                    };
                }
            }
        }
        assert_eq!(dense, expected, "X stream values do not match written data");
    }

    /// Write obs/var with non-trivial metadata columns and verify every column
    /// survives the BpcellsH5Writer → H5SeuratReader round-trip.
    #[tokio::test]
    async fn bpcells_writer_obs_var_metadata_roundtrip() {
        use crate::dtype::TypedVec;
        use crate::h5seurat::H5SeuratReader;
        use crate::ir::{Column, ColumnData, MatrixChunk, ObsTable, SparseMatrixCSR, VarTable};
        use crate::stream::{DatasetReader, DatasetWriter};

        let n_obs = 3usize;
        let n_vars = 2usize;

        let obs = ObsTable {
            index: vec!["cell_A".into(), "cell_B".into(), "cell_C".into()],
            columns: vec![
                Column {
                    name: "n_counts".into(),
                    data: ColumnData::Float(vec![100.0, 200.0, 300.0]),
                },
                Column {
                    name: "is_doublet".into(),
                    data: ColumnData::Bool(vec![false, true, false]),
                },
                Column {
                    name: "cell_type".into(),
                    data: ColumnData::Categorical {
                        codes: vec![0, 1, 0],
                        levels: vec!["T cell".into(), "B cell".into()],
                    },
                },
            ],
        };
        let var = VarTable {
            index: vec!["GeneA".into(), "GeneB".into()],
            columns: vec![Column {
                name: "highly_variable".into(),
                data: ColumnData::Bool(vec![true, false]),
            }],
        };

        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr: vec![0, 1, 2, 2],
                indices: vec![0, 1],
                data: TypedVec::U32(vec![5, 7]),
            },
        };

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer =
            BpcellsH5Writer::create(&path, n_obs, n_vars, DataType::U32, None, None, None, false)
                .unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        let mut reader = H5SeuratReader::open(&path, 1000, None, None).unwrap();

        // Cell names
        let rt_obs = reader.obs().await.unwrap();
        assert_eq!(rt_obs.index, obs.index, "cell names mismatch");

        // n_counts float column
        let nc = rt_obs
            .columns
            .iter()
            .find(|c| c.name == "n_counts")
            .expect("n_counts column missing");
        if let ColumnData::Float(vals) = &nc.data {
            assert_eq!(vals.len(), n_obs);
            for (a, b) in vals.iter().zip([100.0f64, 200.0, 300.0].iter()) {
                assert!((a - b).abs() < 1e-10, "n_counts mismatch: {a} vs {b}");
            }
        } else {
            panic!("n_counts wrong type: {:?}", nc.data);
        }

        // is_doublet bool column
        let id = rt_obs
            .columns
            .iter()
            .find(|c| c.name == "is_doublet")
            .expect("is_doublet column missing");
        if let ColumnData::Bool(vals) = &id.data {
            assert_eq!(vals, &[false, true, false], "is_doublet mismatch");
        } else {
            panic!("is_doublet wrong type: {:?}", id.data);
        }

        // cell_type categorical column
        let ct = rt_obs
            .columns
            .iter()
            .find(|c| c.name == "cell_type")
            .expect("cell_type column missing");
        if let ColumnData::Categorical { codes, levels } = &ct.data {
            assert_eq!(codes, &[0u32, 1, 0], "cell_type codes mismatch");
            assert_eq!(levels, &["T cell", "B cell"], "cell_type levels mismatch");
        } else {
            panic!("cell_type wrong type: {:?}", ct.data);
        }

        // Gene names
        let rt_var = reader.var().await.unwrap();
        assert_eq!(rt_var.index, var.index, "gene names mismatch");
    }

    /// Cell and gene names written by BpcellsH5Writer must appear in the
    /// correct HDF5 locations expected by Seurat v5:
    ///   /cell.names          — root-level cell barcodes
    ///   /assays/RNA/features — gene feature names
    #[tokio::test]
    async fn bpcells_writer_cell_gene_names_in_seurat_locations() {
        use crate::dtype::TypedVec;
        use crate::ir::{MatrixChunk, ObsTable, SparseMatrixCSR, VarTable};
        use crate::stream::DatasetWriter;
        use hdf5::types::VarLenUnicode;

        let n_obs = 2usize;
        let n_vars = 3usize;

        let obs = ObsTable {
            index: vec!["ACGT-1".into(), "TGCA-2".into()],
            columns: vec![],
        };
        let var = VarTable {
            index: vec!["CD3D".into(), "CD3E".into(), "GAPDH".into()],
            columns: vec![],
        };

        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr: vec![0, 1, 1],
                indices: vec![0],
                data: TypedVec::U32(vec![3]),
            },
        };

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer =
            BpcellsH5Writer::create(&path, n_obs, n_vars, DataType::U32, None, None, None, false)
                .unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        // Verify directly in the HDF5 file — same checks Seurat v5 would perform
        let file = hdf5::File::open(&path).unwrap();

        let cell_names: Vec<VarLenUnicode> = file
            .dataset("cell.names")
            .expect("/cell.names missing")
            .read_1d::<VarLenUnicode>()
            .unwrap()
            .to_vec();
        let cell_strs: Vec<&str> = cell_names.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            cell_strs,
            ["ACGT-1", "TGCA-2"],
            "/cell.names content mismatch"
        );

        let features: Vec<VarLenUnicode> = file
            .dataset("assays/RNA/features")
            .expect("/assays/RNA/features missing")
            .read_1d::<VarLenUnicode>()
            .unwrap()
            .to_vec();
        let feat_strs: Vec<&str> = features.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            feat_strs,
            ["CD3D", "CD3E", "GAPDH"],
            "/assays/RNA/features content mismatch"
        );

        // BPCells version attribute must be set so Seurat v5 routes to BPCells backend
        let counts_grp = file
            .group("assays/RNA/counts")
            .expect("/assays/RNA/counts group missing");
        let version_attr = counts_grp
            .attr("version")
            .expect("version attribute missing on /assays/RNA/counts");
        let version: VarLenUnicode = version_attr.read_scalar().unwrap();
        assert!(
            version.as_str().starts_with("packed-"),
            "version attr should be a packed-* BPCells type, got: {}",
            version.as_str()
        );
    }
}
