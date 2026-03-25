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

use async_trait::async_trait;
use hdf5::{File, Group};
use hdf5::types::{VarLenAscii, VarLenUnicode};
use ndarray::Array1;

use crate::bpcells::{
    decode_d1z, decode_for, encode_d1z, encode_for, BpcellsDatasetReader, StorageOrder, ValStore,
};
use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable, VarTable, Varm};
use crate::stream::DatasetWriter;

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

fn seurat_write_strings_local(grp: &Group, name: &str, strings: &[String]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!("BPCells HDF5: removing existing dataset '{name}': {e}"))
        })?;
    }
    let vals: Vec<VarLenUnicode> = strings
        .iter()
        .map(|s| {
            <VarLenUnicode as std::str::FromStr>::from_str(s)
                .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: invalid UTF-8 string for '{name}': {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    grp.new_dataset_builder()
        .with_data(&vals)
        .create(name)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating '{name}': {e}")))?;
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
            let ds = col_grp.new_dataset::<i32>().shape(values.len()).create("values")?;
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
        .map(|c| {
            <VarLenUnicode as std::str::FromStr>::from_str(&c.name)
                .unwrap_or_default()
        })
        .collect();
    if !logical_names.is_empty() {
        let attr = grp.new_attr::<VarLenUnicode>()
            .shape(logical_names.len())
            .create("logicals")?;
        attr.write(&Array1::from_vec(logical_names))?;
    }
    for col in columns {
        seurat_write_col_local(grp, &col.name, &col.data)?;
    }
    Ok(())
}

/// Write a 1-D uint32 dataset into an HDF5 group, replacing any existing one.
fn write_u32s(grp: &Group, name: &str, values: &[u32]) -> Result<()> {
    if grp.link_exists(name) {
        grp.unlink(name).map_err(|e| {
            ScxError::InvalidFormat(format!("BPCells HDF5: removing existing dataset '{name}': {e}"))
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
            ScxError::InvalidFormat(format!("BPCells HDF5: removing existing dataset '{name}': {e}"))
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
            ScxError::InvalidFormat(format!("BPCells HDF5: removing existing dataset '{name}': {e}"))
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
            ScxError::InvalidFormat(format!("BPCells HDF5: removing existing dataset '{name}': {e}"))
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
            ScxError::InvalidFormat(format!("BPCells HDF5: removing existing dataset '{name}': {e}"))
        })?;
    }
    let vals: Vec<VarLenUnicode> = values
        .iter()
        .map(|s| {
            <VarLenUnicode as std::str::FromStr>::from_str(s)
                .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: invalid UTF-8 string for '{name}': {e}")))
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
            .and_then(|a| a.write_scalar(
                &<VarLenUnicode as std::str::FromStr>::from_str(version)
                    .map_err(|e| hdf5::Error::Internal(format!("invalid version string: {e}")))?,
            ))
            .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: writing 'version' attr: {e}")))?;
        return Ok(());
    }

    let v = <VarLenUnicode as std::str::FromStr>::from_str(version)
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: invalid version string: {e}")))?;
    grp.new_attr::<VarLenUnicode>()
        .shape(())
        .create("version")
        .and_then(|a| a.write_scalar(&v))
        .map_err(|e| ScxError::InvalidFormat(format!("BPCells HDF5: creating 'version' attr: {e}")))?;
    Ok(())
}

/// Write a packed BPCells matrix into an HDF5 group.
///
/// `shape` is stored as `[nrow, ncol]` in BPCells convention.
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
    let shape = vec![nrow as u32, ncol as u32];
    let storage_vec = vec![storage];

    write_strings(&grp, "storage_order", &storage_vec)?;
    write_u32s(&grp, "shape", &shape)?;
    write_u64s(&grp, "idxptr", idxptr)?;
    write_strings(&grp, "row_names", row_names)?;
    write_strings(&grp, "col_names", col_names)?;

    let (index_data, index_idx, index_starts) = encode_d1z(index);
    write_u32s(&grp, "index_data", &index_data)?;
    write_u32s(&grp, "index_idx", &index_idx)?;
    write_u64s(&grp, "index_idx_offsets", &vec![0u64; index_starts.len()])?;
    write_u32s(&grp, "index_starts", &index_starts)?;

    match values {
        ValStore::Uint32(v) => {
            let (val_data, val_idx) = encode_for(v);
            write_version_attr(&grp, "packed-uint-matrix-v2")?;
            write_u32s(&grp, "val_data", &val_data)?;
            write_u32s(&grp, "val_idx", &val_idx)?;
            write_u64s(&grp, "val_idx_offsets", &vec![0u64; val_idx.len().saturating_sub(1)])?;
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
    n_vars: usize,
    entries: Vec<(u32, u32, TypedVal)>,
}

impl BpcellsCscAccumulator {
    fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
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
                self.entries.push((col, global_row as u32, val));
            }
        }

        Ok(())
    }

    fn into_csc(mut self) -> Result<BpcellsCscData> {
        self.entries
            .sort_unstable_by_key(|(col, row, _)| (*col, *row));

        let mut idxptr = vec![0u64; self.n_vars + 1];
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
                for (col, row, v) in self.entries {
                    idxptr[col as usize + 1] += 1;
                    index.push(row);
                    match v {
                        TypedVal::F32(x) => vals.push(x),
                        _ => {
                            return Err(ScxError::InvalidFormat(
                                "BPCells writer: mixed value types in accumulator".into(),
                            ))
                        }
                    }
                }
                for i in 0..self.n_vars {
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
                for (col, row, v) in self.entries {
                    idxptr[col as usize + 1] += 1;
                    index.push(row);
                    match v {
                        TypedVal::F64(x) => vals.push(x),
                        _ => {
                            return Err(ScxError::InvalidFormat(
                                "BPCells writer: mixed value types in accumulator".into(),
                            ))
                        }
                    }
                }
                for i in 0..self.n_vars {
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
                for (col, row, v) in self.entries {
                    idxptr[col as usize + 1] += 1;
                    index.push(row);
                    match v {
                        TypedVal::U32(x) => vals.push(x),
                        _ => {
                            return Err(ScxError::InvalidFormat(
                                "BPCells writer: mixed value types in accumulator".into(),
                            ))
                        }
                    }
                }
                for i in 0..self.n_vars {
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

pub struct BpcellsH5Writer {
    file: File,
    assay: String,
    layer: String,
    n_obs: usize,
    n_vars: usize,
    obs: Option<ObsTable>,
    var: Option<VarTable>,
    accumulator: BpcellsCscAccumulator,
}

impl BpcellsH5Writer {
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        _dtype: DataType,
        assay: Option<&str>,
        layer: Option<&str>,
    ) -> Result<Self> {
        let assay = assay.unwrap_or("RNA").to_string();
        let layer = layer.unwrap_or("counts").to_string();
        let file = File::create(path.as_ref())?;

        if file.group("assays").is_err() {
            file.create_group("assays")?;
        }
        if file.group(&format!("assays/{assay}")).is_err() {
            file.create_group(&format!("assays/{assay}"))?;
        }

        Ok(Self {
            file,
            assay,
            layer,
            n_obs,
            n_vars,
            obs: None,
            var: None,
            accumulator: BpcellsCscAccumulator::new(n_vars),
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

    async fn write_obsm(&mut self, _obsm: &Embeddings) -> Result<()> {
        Ok(())
    }

    async fn write_uns(&mut self, _uns: &UnsTable) -> Result<()> {
        Ok(())
    }

    async fn write_varm(&mut self, _varm: &Varm) -> Result<()> {
        Ok(())
    }

    async fn begin_sparse(&mut self, _group_prefix: &str, _name: &str, _meta: &SparseMatrixMeta) -> Result<()> {
        Err(ScxError::InvalidFormat(
            "BPCells writer does not yet support auxiliary sparse matrices".into(),
        ))
    }

    async fn write_sparse_chunk(&mut self, _chunk: &MatrixChunk) -> Result<()> {
        Err(ScxError::InvalidFormat(
            "BPCells writer does not yet support auxiliary sparse matrices".into(),
        ))
    }

    async fn end_sparse(&mut self) -> Result<()> {
        Err(ScxError::InvalidFormat(
            "BPCells writer does not yet support auxiliary sparse matrices".into(),
        ))
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        self.accumulator.push_chunk(chunk)
    }

    async fn finalize(&mut self) -> Result<()> {
        let accumulator = std::mem::replace(
            &mut self.accumulator,
            BpcellsCscAccumulator::new(self.n_vars),
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
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpcells::{bp128_pack, bp128_unpack};

    #[test]
    fn bp128_pack_roundtrip_all_bit_widths() {
        for b in 0u8..=32 {
            let mask = if b == 32 { u32::MAX } else if b == 0 { 0 } else { (1u32 << b) - 1 };
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
        assert_eq!(read_version_attr(&file.group("matrix").unwrap()).as_deref(), Some("packed-uint-matrix-v2"));
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
        assert_eq!(read_version_attr(&file.group("matrix").unwrap()).as_deref(), Some("packed-float-matrix-v2"));
    }
}
