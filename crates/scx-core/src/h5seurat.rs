use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::str::FromStr;

use serde_json;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::{File, Group, SimpleExtents};
use hdf5::types::{TypeDescriptor, FloatSize, VarLenUnicode};
use ndarray::{s, Array1, Array2};

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable,
         SparseMatrixCSR, SparseMatrixMeta, UnsTable, VarTable, Varm},
    stream::{DatasetReader, DatasetWriter},
};

/// Reader for the SeuratDisk H5Seurat format (Seurat v3/v4).
///
/// Schema layout (assay = "RNA", layer = "counts"):
///   /cell.names                     string (ncells,)
///   /assays/RNA/features            string (ngenes,)
///   /assays/RNA/counts/
///     data                          float64 (nnz,)   — raw count values
///     indices                       int32   (nnz,)   — 0-based row (gene) indices
///     indptr                        int32   (ncells+1,) — column (cell) pointers
///     attr:dims                     int32   [ngenes, ncells]
///   /meta.data/
///     <numeric_col>                 float64 (ncells,)
///     <factor_col>/
///       values                      int32 1-indexed codes (ncells,)
///       levels                      string (nlevels,)
///   /reductions/<name>/
///     cell.embeddings               float64 (n_components, ncells) — stored column-major
///
/// The sparse matrix is CSC (gene-major). We stream it as cell-major CSR chunks.
enum XBackend {
    DgCMatrix { indptr: Vec<u64>, dtype: DataType },
    BpCells,
}

pub struct H5SeuratReader {
    path: PathBuf,
    assay: String,
    layer: String,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    x_backend: XBackend,
}

impl H5SeuratReader {
    /// Open an H5Seurat file.
    ///
    /// `assay` defaults to `"RNA"`, `layer` defaults to `"counts"`.
    pub fn open<P: AsRef<Path>>(
        path: P,
        chunk_size: usize,
        assay: Option<&str>,
        layer: Option<&str>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let assay = assay.unwrap_or("RNA").to_string();
        let layer = layer.unwrap_or("counts").to_string();

        let file = File::open(&path)?;
        let dims_path = format!("assays/{assay}/{layer}");
        let dims_grp  = file.group(&dims_path)?;

        // Standard dgCMatrix groups carry a `dims` attribute [ngenes, ncells].
        // BPCells-backed groups instead store a `shape` dataset [nrow, ncol],
        // which for Seurat counts means [ngenes, ncells].
        let (n_vars, n_obs) = if let Ok(dims_attr) = dims_grp.attr("dims") {
            let dims: Vec<i32> = dims_attr.read_1d::<i32>()?.to_vec();
            if dims.len() < 2 {
                return Err(ScxError::InvalidFormat("dims must have 2 elements".into()));
            }
            (dims[0] as usize, dims[1] as usize)
        } else if crate::h5bpcells::probe_bpcells_version(&file, &dims_path).is_some() {
            let shape_ds = file.dataset(&format!("{dims_path}/shape")).map_err(|_| {
                ScxError::InvalidFormat(format!(
                    "missing 'shape' dataset on BPCells group {dims_path}"
                ))
            })?;
            let shape: Vec<u32> = shape_ds.read_1d::<u32>()?.to_vec();
            if shape.len() < 2 {
                return Err(ScxError::InvalidFormat("shape must have 2 elements".into()));
            }
            (shape[0] as usize, shape[1] as usize)
        } else {
            return Err(ScxError::InvalidFormat(format!(
                "missing 'dims' attribute on {dims_path}"
            )));
        };

        let x_backend = if crate::h5bpcells::probe_bpcells_version(&file, &dims_path).is_some() {
            XBackend::BpCells
        } else {
            let indptr_ds_path = format!("{dims_path}/indptr");
            let indptr = read_indptr_from(&file, &indptr_ds_path)?;
            if indptr.len() != n_obs + 1 {
                return Err(ScxError::InvalidFormat(format!(
                    "indptr length {} != n_obs+1 {}",
                    indptr.len(), n_obs + 1
                )));
            }

            let data_ds_path = format!("{dims_path}/data");
            let dtype = detect_dtype(&file, &data_ds_path)?;
            XBackend::DgCMatrix { indptr, dtype }
        };

        Ok(Self { path, assay, layer, n_obs, n_vars, chunk_size, x_backend })
    }
}

// ---------------------------------------------------------------------------
// Sync helpers
// ---------------------------------------------------------------------------

fn read_indptr_from(file: &File, path: &str) -> Result<Vec<u64>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(_) => {
            Ok(ds.read_1d::<f64>()?.iter().map(|&x| x as u64).collect())
        }
        TypeDescriptor::Integer(_) => {
            Ok(ds.read_1d::<i32>()?.iter().map(|&x| x as u64).collect())
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unexpected indptr type at {path}: {:?}", other
        ))),
    }
}

fn read_indices_at(file: &File, path: &str, start: usize, end: usize) -> Result<Vec<u32>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) => {
            Ok(ds.read_slice_1d::<i32, _>(s![start..end])?.iter().map(|&x| x as u32).collect())
        }
        _ => Err(ScxError::InvalidFormat(format!("unexpected indices type at {path}"))),
    }
}

fn detect_dtype(file: &File, path: &str) -> Result<DataType> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_)             => DataType::F64,
        TypeDescriptor::Integer(_)           => DataType::I32,
        _                                    => DataType::F32,
    })
}

fn read_strings(file: &File, path: &str) -> Result<Vec<String>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => {
            let raw: ndarray::Array1<VarLenUnicode> = ds.read_1d()?;
            Ok(raw.into_iter().map(|s| s.to_string()).collect())
        }
        TypeDescriptor::VarLenAscii => {
            let raw: ndarray::Array1<hdf5::types::VarLenAscii> = ds.read_1d()?;
            Ok(raw.into_iter().map(|s| s.to_string()).collect())
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported string type {:?} at '{path}'", other
        ))),
    }
}

fn read_chunk_sync(
    path: &Path,
    assay: &str,
    layer: &str,
    indptr: &[u64],
    cell_start: usize,
    cell_end: usize,
    n_vars: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    let chunk_cells = cell_end - cell_start;
    let nnz_start = indptr[cell_start] as usize;
    let nnz_end   = indptr[cell_end]   as usize;
    let nnz = nnz_end - nnz_start;

    let base = format!("assays/{assay}/{layer}");

    let gene_indices: Vec<u32> = if nnz > 0 {
        read_indices_at(&file, &format!("{base}/indices"), nnz_start, nnz_end)?
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset(&format!("{base}/data"))?;
        match dtype {
            DataType::F32 => TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::F64 => TypedVec::F64(ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::I32 => TypedVec::I32(ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::U32 => TypedVec::U32(ds.read_slice_1d::<u32, _>(s![nnz_start..nnz_end])?.to_vec()),
        }
    } else {
        TypedVec::F32(Vec::new())
    };

    // CSC column pointers → CSR row pointers (same data, zero-copy reinterpretation).
    let csr_indptr: Vec<u64> = indptr[cell_start..=cell_end]
        .iter()
        .map(|&p| p - indptr[cell_start])
        .collect();

    Ok(MatrixChunk {
        row_offset: cell_start,
        nrows: chunk_cells,
        data: SparseMatrixCSR {
            shape: (chunk_cells, n_vars),
            indptr: csr_indptr,
            indices: gene_indices,
            data,
        },
    })
}

fn read_obs_sync(path: &Path) -> Result<ObsTable> {
    let file = File::open(path)?;
    let index = read_strings(&file, "cell.names")?;

    // Collect which columns are logical (encoded 0/1/2)
    let logicals: std::collections::HashSet<String> = {
        let grp = file.group("meta.data")?;
        if let Ok(attr) = grp.attr("logicals") {
            let raw: ndarray::Array1<VarLenUnicode> = attr.read_1d().unwrap_or_default();
            raw.into_iter().map(|s| s.to_string()).collect()
        } else {
            std::collections::HashSet::new()
        }
    };

    let meta_grp = file.group("meta.data")?;
    let members = meta_grp.member_names()?;
    let mut columns = Vec::new();

    for name in &members {
        // Each member is either a dataset (numeric/logical/string) or a group (factor)
        let is_group = file.group(&format!("meta.data/{name}")).is_ok()
            && file.dataset(&format!("meta.data/{name}")).is_err();

        let col_data = if is_group {
            // Factor: group with values (1-indexed int) + levels (string)
            match read_factor_column(&file, &format!("meta.data/{name}")) {
                Ok(cd) => cd,
                Err(e) => { tracing::warn!("skipping factor column '{name}': {e}"); continue; }
            }
        } else if logicals.contains(name.as_str()) {
            // Logical: int32 (0=F, 1=T, 2=NA) → Bool (NA → false for now)
            let ds = file.dataset(&format!("meta.data/{name}"))?;
            let vals: Vec<i32> = ds.read_1d::<i32>()?.to_vec();
            ColumnData::Bool(vals.into_iter().map(|v| v == 1).collect())
        } else {
            // Numeric or string dataset
            match read_meta_column(&file, &format!("meta.data/{name}")) {
                Ok(cd) => cd,
                Err(e) => { tracing::warn!("skipping obs column '{name}': {e}"); continue; }
            }
        };

        columns.push(Column { name: name.clone(), data: col_data });
    }

    Ok(ObsTable { index, columns })
}

fn read_factor_column(file: &File, grp_path: &str) -> Result<ColumnData> {
    let values_path = format!("{grp_path}/values");
    let levels_path = format!("{grp_path}/levels");
    let codes: Vec<u32> = file.dataset(&values_path)?
        .read_1d::<i32>()?
        .iter()
        .map(|&v| (v - 1).max(0) as u32)   // 1-indexed → 0-indexed
        .collect();
    let levels = read_strings(file, &levels_path)?;
    Ok(ColumnData::Categorical { codes, levels })
}

fn read_meta_column(file: &File, ds_path: &str) -> Result<ColumnData> {
    let ds = file.dataset(ds_path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => {
            let v: Vec<f32> = ds.read_1d::<f32>()?.to_vec();
            Ok(ColumnData::Float(v.into_iter().map(|x| x as f64).collect()))
        }
        TypeDescriptor::Float(_) => {
            Ok(ColumnData::Float(ds.read_1d::<f64>()?.to_vec()))
        }
        TypeDescriptor::Integer(_) => {
            Ok(ColumnData::Int(ds.read_1d::<i32>()?.to_vec()))
        }
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            Ok(ColumnData::String(read_strings(file, ds_path)?))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported column type {:?} at {ds_path}", other
        ))),
    }
}

fn read_var_sync(path: &Path, assay: &str) -> Result<VarTable> {
    let file = File::open(path)?;
    let index = read_strings(&file, &format!("assays/{assay}/features"))?;

    let mf_grp_path = format!("assays/{assay}/meta.features");
    let columns = match file.group(&mf_grp_path) {
        Err(_) => Vec::new(),
        Ok(grp) => {
            // Which columns are logical (0/1/2 encoded)
            let logicals: std::collections::HashSet<String> = {
                if let Ok(attr) = grp.attr("logicals") {
                    let raw: ndarray::Array1<VarLenUnicode> = attr.read_1d().unwrap_or_default();
                    raw.into_iter().map(|s| s.to_string()).collect()
                } else {
                    std::collections::HashSet::new()
                }
            };

            let mut cols = Vec::new();
            for name in grp.member_names().unwrap_or_default() {
                let ds_path = format!("{mf_grp_path}/{name}");
                let is_group = file.group(&ds_path).is_ok()
                    && file.dataset(&ds_path).is_err();
                let col_data = if is_group {
                    match read_factor_column(&file, &ds_path) {
                        Ok(cd) => cd,
                        Err(e) => { tracing::warn!("skipping var factor '{name}': {e}"); continue; }
                    }
                } else if logicals.contains(name.as_str()) {
                    let ds = match file.dataset(&ds_path) {
                        Ok(d) => d,
                        Err(e) => { tracing::warn!("skipping var logical '{name}': {e}"); continue; }
                    };
                    let vals: Vec<i32> = ds.read_1d::<i32>()?.to_vec();
                    ColumnData::Bool(vals.into_iter().map(|v| v == 1).collect())
                } else {
                    match read_meta_column(&file, &ds_path) {
                        Ok(cd) => cd,
                        Err(e) => { tracing::warn!("skipping var column '{name}': {e}"); continue; }
                    }
                };
                cols.push(Column { name, data: col_data });
            }
            cols
        }
    };

    Ok(VarTable { index, columns })
}

fn read_obsm_sync(path: &Path, n_obs: usize) -> Result<Embeddings> {
    let file = File::open(path)?;
    let reds_grp = match file.group("reductions") {
        Ok(g)  => g,
        Err(_) => return Ok(Embeddings::default()),
    };
    let mut map = HashMap::new();
    for red_name in reds_grp.member_names()? {
        let ds_path = format!("reductions/{red_name}/cell.embeddings");
        let ds = match file.dataset(&ds_path) {
            Ok(d)  => d,
            Err(_) => continue,
        };
        let arr: ndarray::Array2<f64> = match ds.read::<f64, ndarray::Ix2>() {
            Ok(a)  => a,
            Err(e) => { tracing::warn!("skipping reduction '{red_name}': {e}"); continue; }
        };
        // SeuratDisk stores cell.embeddings column-major as (n_components, n_obs).
        // After HDF5 read in row-major, we get shape (n_components, n_obs) unless
        // the writer already transposed. Detect and fix.
        let arr = if arr.shape()[0] != n_obs && arr.shape()[1] == n_obs {
            arr.t().as_standard_layout().into_owned()
        } else {
            arr
        };
        // Map reduction name to AnnData obsm key convention
        let obsm_key = format!("X_{}", red_name.to_lowercase());
        let shape = (arr.shape()[0], arr.shape()[1]);
        map.insert(obsm_key, DenseMatrix { shape, data: arr.into_raw_vec() });
    }
    Ok(Embeddings { map })
}

// ---------------------------------------------------------------------------
// uns helpers — walk misc/ into a serde_json::Value tree
// ---------------------------------------------------------------------------

/// Recursively walk an HDF5 group into a JSON object.
/// Unreadable or unsupported nodes are silently replaced with `null`.
fn seurat_walk_group(file: &File, group_path: &str) -> serde_json::Value {
    let grp = match file.group(group_path) {
        Ok(g)  => g,
        Err(_) => return serde_json::Value::Null,
    };
    let members = grp.member_names().unwrap_or_default();
    let mut map = serde_json::Map::new();
    for name in members {
        let child = format!("{group_path}/{name}");
        let is_grp = file.group(&child).is_ok() && file.dataset(&child).is_err();
        let value  = if is_grp {
            seurat_walk_group(file, &child)
        } else {
            seurat_ds_to_json(file, &child).unwrap_or(serde_json::Value::Null)
        };
        map.insert(name, value);
    }
    serde_json::Value::Object(map)
}

fn seurat_ds_to_json(file: &File, path: &str) -> Result<serde_json::Value> {
    let ds        = file.dataset(path)?;
    let is_scalar = ds.ndim() == 0;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(_) => {
            if is_scalar {
                Ok(serde_json::Value::from(ds.read_scalar::<f64>()?))
            } else {
                let v: Vec<f64> = ds.read_1d::<f64>()?.to_vec();
                Ok(serde_json::json!(v))
            }
        }
        TypeDescriptor::Integer(_) => {
            if is_scalar {
                Ok(serde_json::Value::from(ds.read_scalar::<i64>()?))
            } else {
                let v: Vec<i64> = ds.read_1d::<i64>()?.to_vec();
                Ok(serde_json::json!(v))
            }
        }
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            let strings = read_strings(file, path)?;
            if is_scalar || strings.len() == 1 {
                Ok(serde_json::Value::String(
                    strings.into_iter().next().unwrap_or_default()
                ))
            } else {
                Ok(serde_json::json!(strings))
            }
        }
        _ => Ok(serde_json::Value::Null),
    }
}

fn read_uns_sync(path: &Path) -> Result<UnsTable> {
    let file = File::open(path)?;
    if file.group("misc").is_err() {
        return Ok(UnsTable::default());
    }
    Ok(UnsTable { raw: seurat_walk_group(&file, "misc") })
}

// ---------------------------------------------------------------------------
// Slot parity helpers
// ---------------------------------------------------------------------------

/// Classify an H5Seurat sparse group as either classic dgCMatrix storage or
/// BPCells-backed storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SparseGroupKind {
    DgCMatrix,
    BpCells,
}

fn detect_sparse_group_kind(file: &File, group_path: &str) -> Option<SparseGroupKind> {
    if file.group(group_path).is_err() {
        return None;
    }
    if crate::h5bpcells::probe_bpcells_version(file, group_path).is_some() {
        return Some(SparseGroupKind::BpCells);
    }
    if file.dataset(&format!("{group_path}/indptr")).is_ok() {
        return Some(SparseGroupKind::DgCMatrix);
    }
    None
}

/// Read the shape and indptr for a single H5Seurat sparse group.
/// Supports both classic dgCMatrix groups (`dims` + `indptr`) and
/// BPCells-backed groups (`shape` + `idxptr` with `version` attr).
fn seurat_read_sparse_meta(file: &File, name: &str, group_path: &str) -> Result<SparseMatrixMeta> {
    let grp = file.group(group_path)?;

    if matches!(detect_sparse_group_kind(file, group_path), Some(SparseGroupKind::BpCells)) {
        let shape_ds = file.dataset(&format!("{group_path}/shape"))?;
        let shape: Vec<u32> = shape_ds.read_1d::<u32>()?.to_vec();
        if shape.len() < 2 {
            return Err(ScxError::InvalidFormat(format!(
                "BPCells layer/group '{group_path}' has shape with < 2 elements"
            )));
        }
        let nrows = shape[1] as usize;
        let ncols = shape[0] as usize;
        let idxptr_ds = file.dataset(&format!("{group_path}/idxptr"))?;
        let indptr: Vec<u64> = match idxptr_ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {
                if idxptr_ds.dtype()?.size() == 8 {
                    idxptr_ds.read_1d::<u64>()?.to_vec()
                } else {
                    idxptr_ds.read_1d::<u32>()?.iter().map(|&x| x as u64).collect()
                }
            }
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "unexpected BPCells idxptr type at {group_path}: {:?}", other
                )))
            }
        };
        return Ok(SparseMatrixMeta {
            name: name.to_string(),
            shape: (nrows, ncols),
            indptr,
        });
    }

    let dims_attr = grp.attr("dims")?;
    let dims: Vec<i32> = dims_attr.read_1d::<i32>()?.to_vec();
    // H5Seurat dims attr is [n_rows, n_cols] where columns = CSC dimension
    let (nrows, ncols) = (dims[0] as usize, dims[1] as usize);
    let indptr = read_indptr_from(file, &format!("{group_path}/indptr"))?;
    Ok(SparseMatrixMeta { name: name.to_string(), shape: (nrows, ncols), indptr })
}

/// Read a row-slice of an H5Seurat CSC sparse group as a CSR `MatrixChunk`.
fn seurat_read_sparse_chunk(
    path: &Path,
    group_path: &str,
    meta: &SparseMatrixMeta,
    row_start: usize,
    row_end: usize,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    let (_nrows, ncols) = meta.shape;
    let chunk_rows = row_end - row_start;

    let nnz_start = meta.indptr[row_start] as usize;
    let nnz_end   = meta.indptr[row_end]   as usize;
    let nnz = nnz_end - nnz_start;

    let indices: Vec<u32> = if nnz > 0 {
        read_indices_at(&file, &format!("{group_path}/indices"), nnz_start, nnz_end)?
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset(&format!("{group_path}/data"))?;
        match ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Float(FloatSize::U4) => TypedVec::F32(
                ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec()),
            TypeDescriptor::Float(_) => TypedVec::F64(
                ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec()),
            TypeDescriptor::Integer(_) => TypedVec::I32(
                ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec()),
            _ => TypedVec::F32(
                ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec()),
        }
    } else {
        TypedVec::F32(Vec::new())
    };

    // CSC column pointers → CSR row pointers (zero-based within chunk).
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

fn read_layer_metas_sync(path: &Path, assay: &str, primary_layer: &str) -> Result<Vec<SparseMatrixMeta>> {
    let file = File::open(path)?;
    let assay_grp = match file.group(&format!("assays/{assay}")) {
        Err(_) => return Ok(Vec::new()),
        Ok(g)  => g,
    };
    let mut metas = Vec::new();
    for name in assay_grp.member_names().unwrap_or_default() {
        if name == primary_layer { continue; }
        let grp_path = format!("assays/{assay}/{name}");

        if detect_sparse_group_kind(&file, &grp_path).is_none() { continue; }

        match seurat_read_sparse_meta(&file, &name, &grp_path) {
            Ok(m)  => metas.push(m),
            Err(e) => tracing::warn!("skipping assay layer '{name}': {e}"),
        }
    }
    Ok(metas)
}

fn read_obsp_metas_sync(path: &Path) -> Result<Vec<SparseMatrixMeta>> {
    let file = File::open(path)?;
    let grp = match file.group("graphs") {
        Err(_) => return Ok(Vec::new()),
        Ok(g)  => g,
    };
    let mut metas = Vec::new();
    for name in grp.member_names().unwrap_or_default() {
        let grp_path = format!("graphs/{name}");
        if detect_sparse_group_kind(&file, &grp_path).is_none() { continue; }
        match seurat_read_sparse_meta(&file, &name, &grp_path) {
            Ok(m)  => metas.push(m),
            Err(e) => tracing::warn!("skipping graph '{name}': {e}"),
        }
    }
    Ok(metas)
}

fn read_varm_sync(path: &Path, n_vars: usize) -> Result<Varm> {
    let file = File::open(path)?;
    let reds_grp = match file.group("reductions") {
        Err(_) => return Ok(Varm::default()),
        Ok(g)  => g,
    };
    let mut map = HashMap::new();
    for red_name in reds_grp.member_names().unwrap_or_default() {
        let ds_path = format!("reductions/{red_name}/feature.loadings");
        let ds = match file.dataset(&ds_path) {
            Ok(d)  => d,
            Err(_) => continue,
        };
        let arr: ndarray::Array2<f64> = match ds.read::<f64, ndarray::Ix2>() {
            Ok(a)  => a,
            Err(e) => { tracing::warn!("skipping varm '{red_name}': {e}"); continue; }
        };
        // feature.loadings stored as (k, n_vars) — transpose to (n_vars, k)
        let arr = if arr.shape()[1] == n_vars && arr.shape()[0] != n_vars {
            arr.t().as_standard_layout().into_owned()
        } else {
            arr
        };
        let shape = (arr.shape()[0], arr.shape()[1]);
        let varm_key = format!("X_{}", red_name.to_lowercase());
        map.insert(varm_key, DenseMatrix { shape, data: arr.into_raw_vec() });
    }
    Ok(Varm { map })
}

// ---------------------------------------------------------------------------
// Seurat v5 / BPCells routing
// ---------------------------------------------------------------------------

/// Candidate group paths to probe within an H5Seurat file, in priority order.
///
/// Seurat v4/v3: `assays/{assay}/{layer}` (dgCMatrix or BPCells)
/// Seurat v5:    `assays/{assay}/layers/{layer}` (BPCells or future formats)
fn candidate_group_paths(assay: &str, layer: &str) -> Vec<String> {
    vec![
        format!("assays/{assay}/{layer}"),
        format!("assays/{assay}/layers/{layer}"),
    ]
}

/// Open an H5Seurat file, automatically routing to `BpcellsDatasetReader`
/// when the matrix group carries a BPCells `version` attribute.
///
/// Falls back to the standard `H5SeuratReader` (dgCMatrix path) otherwise.
pub fn open_h5seurat<P: AsRef<Path>>(
    path: P,
    chunk_size: usize,
    assay: Option<&str>,
    layer: Option<&str>,
) -> Result<Box<dyn crate::stream::DatasetReader + Send>> {
    let path = path.as_ref();
    let assay = assay.unwrap_or("RNA");
    let layer = layer.unwrap_or("counts");
    Ok(Box::new(H5SeuratReader::open(path, chunk_size, Some(assay), Some(layer))?))
}

// ---------------------------------------------------------------------------
// DatasetReader impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetReader for H5SeuratReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        match &self.x_backend {
            XBackend::DgCMatrix { dtype, .. } => *dtype,
            XBackend::BpCells => DataType::F64,
        }
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        read_obs_sync(&self.path)
    }

    async fn var(&mut self) -> Result<VarTable> {
        read_var_sync(&self.path, &self.assay)
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        read_obsm_sync(&self.path, self.n_obs)
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        read_uns_sync(&self.path)
    }

    async fn varm(&mut self) -> Result<Varm> {
        read_varm_sync(&self.path, self.n_vars)
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        read_layer_metas_sync(&self.path, &self.assay, &self.layer)
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        read_obsp_metas_sync(&self.path)
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        let path      = self.path.clone();
        let assay     = self.assay.clone();
        let grp_path  = format!("assays/{}/{}", assay, meta.name);
        let n_rows    = meta.shape.0;

        let is_bpcells = {
            let file = File::open(&path);
            match file {
                Ok(file) => matches!(detect_sparse_group_kind(&file, &grp_path), Some(SparseGroupKind::BpCells)),
                Err(_) => false,
            }
        };

        if is_bpcells {
            let bp_reader = {
                let file = match File::open(&path) {
                    Ok(file) => file,
                    Err(e) => {
                        return Box::pin(stream::once(async move { Err(ScxError::from(e)) }));
                    }
                };
                match crate::h5bpcells::open_bpcells_h5(&file, &grp_path, chunk_size) {
                    Ok(reader) => reader,
                    Err(e) => return Box::pin(stream::once(async move { Err(e) })),
                }
            };

            Box::pin(stream::unfold(0usize, move |row_start| {
                let reader = bp_reader.clone();
                async move {
                    if row_start >= n_rows { return None; }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk = reader.read_chunk(row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        } else {
            Box::pin(stream::unfold(0usize, move |row_start| {
                let path     = path.clone();
                let grp_path = grp_path.clone();
                async move {
                    if row_start >= n_rows { return None; }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk = seurat_read_sparse_chunk(&path, &grp_path, meta, row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        }
    }

    fn obsp_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        let path     = self.path.clone();
        let grp_path = format!("graphs/{}", meta.name);
        let n_rows   = meta.shape.0;

        let is_bpcells = {
            let file = File::open(&path);
            match file {
                Ok(file) => matches!(detect_sparse_group_kind(&file, &grp_path), Some(SparseGroupKind::BpCells)),
                Err(_) => false,
            }
        };

        if is_bpcells {
            let bp_reader = {
                let file = match File::open(&path) {
                    Ok(file) => file,
                    Err(e) => {
                        return Box::pin(stream::once(async move { Err(ScxError::from(e)) }));
                    }
                };
                match crate::h5bpcells::open_bpcells_h5(&file, &grp_path, chunk_size) {
                    Ok(reader) => reader,
                    Err(e) => return Box::pin(stream::once(async move { Err(e) })),
                }
            };

            Box::pin(stream::unfold(0usize, move |row_start| {
                let reader = bp_reader.clone();
                async move {
                    if row_start >= n_rows { return None; }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk = reader.read_chunk(row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        } else {
            Box::pin(stream::unfold(0usize, move |row_start| {
                let path     = path.clone();
                let grp_path = grp_path.clone();
                async move {
                    if row_start >= n_rows { return None; }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk = seurat_read_sparse_chunk(&path, &grp_path, meta, row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        }
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        match &self.x_backend {
            XBackend::DgCMatrix { indptr, dtype } => {
                let path       = self.path.clone();
                let assay      = self.assay.clone();
                let layer      = self.layer.clone();
                let n_obs      = self.n_obs;
                let n_vars     = self.n_vars;
                let chunk_size = self.chunk_size;
                let indptr     = indptr.clone();
                let dtype      = *dtype;

                Box::pin(stream::unfold(0usize, move |cell_start| {
                    let path   = path.clone();
                    let assay  = assay.clone();
                    let layer  = layer.clone();
                    let indptr = indptr.clone();
                    async move {
                        if cell_start >= n_obs { return None; }
                        let cell_end = (cell_start + chunk_size).min(n_obs);
                        let chunk = read_chunk_sync(
                            &path, &assay, &layer, &indptr,
                            cell_start, cell_end, n_vars, dtype,
                        );
                        Some((chunk, cell_end))
                    }
                }))
            }
            XBackend::BpCells => {
                let path       = self.path.clone();
                let assay      = self.assay.clone();
                let layer      = self.layer.clone();
                let n_obs      = self.n_obs;
                let chunk_size = self.chunk_size;

                let bp_reader = {
                    let file = File::open(&path).expect("failed to open H5Seurat file for BPCells backend");
                    let grp_path = candidate_group_paths(&assay, &layer)
                        .into_iter()
                        .find(|p| file.group(p).is_ok())
                        .expect("missing assay/layer group for BPCells backend");
                    crate::h5bpcells::open_bpcells_h5(&file, &grp_path, chunk_size)
                        .expect("failed to open BPCells matrix backend")
                };

                Box::pin(stream::unfold(0usize, move |cell_start| {
                    let reader = bp_reader.clone();
                    async move {
                        if cell_start >= n_obs { return None; }
                        let cell_end = (cell_start + chunk_size).min(n_obs);
                        let chunk = reader.read_chunk(cell_start, cell_end);
                        Some((chunk, cell_end))
                    }
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// H5SeuratWriter
// ---------------------------------------------------------------------------

const SEURAT_CHUNK_ELEMS: usize = 65_536;

/// Streaming writer for the SeuratDisk H5Seurat format (Seurat v3/v4).
///
/// Schema written (mirrors what `H5SeuratReader` expects):
///   /cell.names                          VarLenUnicode (n_obs,)
///   /assays/{assay}/features             VarLenUnicode (n_vars,)
///   /assays/{assay}/{layer}/
///     data                               typed        (nnz,)
///     indices                            i32          (nnz,)  — gene indices
///     indptr                             i32/i64      (n_obs+1,) — cell pointers
///     attr:dims                          i32[2]       [n_vars, n_obs]
///   /meta.data/
///     attr:logicals    string array — names of Bool columns
///     <float_col>      float64 (n_obs,)
///     <int_col>        int32   (n_obs,)
///     <bool_col>       int32   (n_obs,)   0=F 1=T
///     <str_col>        VarLenUnicode (n_obs,)
///     <factor_col>/    group
///       values         int32 (n_obs,)  — 1-indexed codes
///       levels         VarLenUnicode (n_levels,)
///   /assays/{assay}/meta.features/       (omitted when var.columns is empty)
///   /reductions/{name}/
///     cell.embeddings  float64 (n_comps, n_obs)  — transposed from IR (n_obs, n_comps)
///
/// Call order: write_obs → write_var → write_obsm → write_uns → write_x_chunk* → finalize.
/// The first four may arrive in any order; chunks must arrive in cell order.
/// State kept while streaming a single named sparse matrix (layer or obsp).
struct SparseWriteState {
    /// HDF5 group path being written (e.g. "assays/RNA/data" or "graphs/nn").
    group_path: String,
    /// Accumulated CSR indptr across written chunks.
    indptr: Vec<u64>,
    /// Shape of the matrix: (nrows, ncols).
    shape: (usize, usize),
}

pub struct H5SeuratWriter {
    file: File,
    assay: String,
    layer: String,
    n_obs: usize,
    n_vars: usize,
    dtype: DataType,
    /// Accumulated cell indptr (n_obs + 1 entries when finalized).
    x_indptr: Vec<u64>,
    /// State for the currently open streaming sparse matrix, if any.
    sparse_state: Option<SparseWriteState>,
}

impl H5SeuratWriter {
    /// Create a new H5Seurat file for writing.
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
        assay: Option<&str>,
        layer: Option<&str>,
        project: Option<&str>,
    ) -> Result<Self> {
        let assay   = assay.unwrap_or("RNA").to_string();
        let layer   = layer.unwrap_or("counts").to_string();
        let project = project.unwrap_or("SeuratProject");

        let file = File::create(path.as_ref())?;

        // Root-level attributes required by SeuratDisk::LoadH5Seurat.
        let root = file.group("/")?;
        for (name, value) in [
            ("version",      "3.1.5.9900"),
            ("active.assay", assay.as_str()),
            ("project",      project),
        ] {
            let v = VarLenUnicode::from_str(value).unwrap_or_default();
            root.new_attr::<VarLenUnicode>().create(name)?.write_scalar(&v)?;
        }

        // Pre-create the group hierarchy needed before the resizable datasets.
        file.create_group("assays")?;
        file.create_group(&format!("assays/{assay}"))?;
        file.create_group(&format!("assays/{assay}/{layer}"))?;
        // All top-level groups required by SeuratDisk::LoadH5Seurat, even when empty.
        for grp in &["commands", "graphs", "images", "misc", "neighbors", "reductions", "tools"] {
            file.create_group(grp)?;
        }

        // Resizable datasets for streaming x-chunk writes.
        let data_path    = format!("assays/{assay}/{layer}/data");
        let indices_path = format!("assays/{assay}/{layer}/indices");
        match dtype {
            DataType::F32 => seurat_init_resizable::<f32>(&file, &data_path)?,
            DataType::F64 => seurat_init_resizable::<f64>(&file, &data_path)?,
            DataType::I32 => seurat_init_resizable::<i32>(&file, &data_path)?,
            DataType::U32 => seurat_init_resizable::<u32>(&file, &data_path)?,
        }
        seurat_init_resizable::<i32>(&file, &indices_path)?;

        Ok(Self { file, assay, layer, n_obs, n_vars, dtype, x_indptr: vec![0u64], sparse_state: None })
    }
}

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

fn seurat_init_resizable<T: hdf5::H5Type>(file: &File, path: &str) -> Result<()> {
    file.new_dataset::<T>()
        .chunk(SEURAT_CHUNK_ELEMS)
        .shape(SimpleExtents::resizable([0usize]))
        .create(path)?;
    Ok(())
}

fn seurat_write_strings(grp: &Group, name: &str, strings: &[String]) -> Result<()> {
    let vals: Vec<VarLenUnicode> = strings
        .iter()
        .map(|s| VarLenUnicode::from_str(s).unwrap_or_default())
        .collect();
    let ds = grp.new_dataset::<VarLenUnicode>().shape(vals.len()).create(name)?;
    ds.write(&Array1::from_vec(vals))?;
    Ok(())
}

/// Write all metadata columns into `grp`.  Also writes the `logicals` attribute
/// listing the names of Bool columns (R's integer-encoded logicals convention).
fn seurat_write_meta_cols(grp: &Group, columns: &[Column]) -> Result<()> {
    let logical_names: Vec<VarLenUnicode> = columns
        .iter()
        .filter(|c| matches!(c.data, ColumnData::Bool(_)))
        .map(|c| VarLenUnicode::from_str(&c.name).unwrap_or_default())
        .collect();
    if !logical_names.is_empty() {
        let attr = grp.new_attr::<VarLenUnicode>()
            .shape(logical_names.len())
            .create("logicals")?;
        attr.write(&Array1::from_vec(logical_names))?;
    }
    for col in columns {
        seurat_write_col(grp, &col.name, &col.data)?;
    }
    Ok(())
}



fn seurat_write_col(grp: &Group, name: &str, data: &ColumnData) -> Result<()> {
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
            // Stored as int32 (0/1); column name is tracked in the `logicals` attr
            let vi: Vec<i32> = v.iter().map(|&b| b as i32).collect();
            let ds = grp.new_dataset::<i32>().shape(vi.len()).create(name)?;
            ds.write(&Array1::from_vec(vi))?;
        }
        ColumnData::String(v) => {
            seurat_write_strings(grp, name, v)?;
        }
        ColumnData::Categorical { codes, levels } => {
            let col_grp = grp.create_group(name)?;
            // 0-indexed codes → 1-indexed values (R dgCMatrix convention)
            let values: Vec<i32> = codes.iter().map(|&c| c as i32 + 1).collect();
            let ds = col_grp.new_dataset::<i32>().shape(values.len()).create("values")?;
            ds.write(&Array1::from_vec(values))?;
            seurat_write_strings(&col_grp, "levels", levels)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DatasetWriter impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetWriter for H5SeuratWriter {
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()> {
        // /cell.names — root-level cell barcode array
        let root = self.file.group("/")?;
        seurat_write_strings(&root, "cell.names", &obs.index)?;

        // /meta.data/ — always created even when obs.columns is empty
        let meta_grp = self.file.create_group("meta.data")?;
        seurat_write_meta_cols(&meta_grp, &obs.columns)?;

        Ok(())
    }

    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        // /assays/{assay}/features
        let assay_grp = self.file.group(&format!("assays/{}", self.assay))?;
        seurat_write_strings(&assay_grp, "features", &var.index)?;

        // /assays/{assay}/meta.features/ — only when var has columns
        if !var.columns.is_empty() {
            let mf_grp = assay_grp.create_group("meta.features")?;
            seurat_write_meta_cols(&mf_grp, &var.columns)?;
        }

        Ok(())
    }

    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()> {
        // Always create the reductions group — SeuratDisk requires it even when empty.
        let reds_grp = self.file.create_group("reductions")?;
        if obsm.map.is_empty() {
            return Ok(());
        }
        for (key, mat) in &obsm.map {
            let red_name = key.strip_prefix("X_").unwrap_or(key.as_str());
            let red_grp  = reds_grp.create_group(red_name)?;

            let (n_obs, n_comps) = mat.shape;
            // IR: (n_obs, n_comps) row-major → H5Seurat: (n_comps, n_obs)
            // Build a new C-contiguous (n_comps, n_obs) array.
            // Avoid .t().to_owned(): hdf5-rs 0.8 rejects non-standard-layout inputs.
            let mut buf = vec![0.0f64; n_obs * n_comps];
            for j in 0..n_obs {
                for i in 0..n_comps {
                    buf[i * n_obs + j] = mat.data[j * n_comps + i];
                }
            }
            let arr_t = Array2::from_shape_vec((n_comps, n_obs), buf)
                .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
            let ds = red_grp
                .new_dataset::<f64>()
                .shape((n_comps, n_obs))
                .create("cell.embeddings")?;
            ds.write(&arr_t)?;
        }
        Ok(())
    }

    async fn write_uns(&mut self, _uns: &UnsTable) -> Result<()> {
        Ok(()) // H5Seurat has no uns equivalent
    }

    async fn begin_sparse(
        &mut self,
        group_prefix: &str,
        name: &str,
        meta: &SparseMatrixMeta,
    ) -> Result<()> {
        // Determine the HDF5 group path for this sparse matrix.
        let group_path = match group_prefix {
            "layers" => {
                // Ensure parent exists.
                if self.file.group(&format!("assays/{}", self.assay)).is_err() {
                    self.file.create_group(&format!("assays/{}", self.assay))?;
                }
                format!("assays/{}/{}", self.assay, name)
            }
            "obsp" => {
                if self.file.group("graphs").is_err() {
                    self.file.create_group("graphs")?;
                }
                format!("graphs/{name}")
            }
            other => format!("{other}/{name}"),
        };

        self.file.create_group(&group_path)?;

        // Pre-create resizable data/indices datasets so chunks can be appended.
        seurat_init_resizable::<f64>(&self.file, &format!("{group_path}/data"))?;
        seurat_init_resizable::<i32>(&self.file, &format!("{group_path}/indices"))?;

        self.sparse_state = Some(SparseWriteState {
            group_path,
            indptr: vec![0u64],
            shape: meta.shape,
        });
        Ok(())
    }

    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let state = self.sparse_state.as_mut()
            .ok_or_else(|| ScxError::InvalidFormat("write_sparse_chunk called without begin_sparse".into()))?;

        let csr = &chunk.data;
        let nnz = csr.indices.len();

        if nnz > 0 {
            let data_ds = self.file.dataset(&format!("{}/data", state.group_path))?;
            let old_len = data_ds.shape()[0];
            let new_len = old_len + nnz;
            data_ds.resize(new_len)?;
            let vals: Vec<f64> = csr.data.to_f64();
            data_ds.write_slice(&Array1::from_vec(vals), s![old_len..new_len])?;

            let idx_ds = self.file.dataset(&format!("{}/indices", state.group_path))?;
            idx_ds.resize(new_len)?;
            let genes_i32: Vec<i32> = csr.indices.iter().map(|&x| x as i32).collect();
            idx_ds.write_slice(&Array1::from_vec(genes_i32), s![old_len..new_len])?;
        }

        // Accumulate indptr.
        let base = *state.indptr.last().unwrap();
        for i in 1..=chunk.nrows {
            state.indptr.push(base + csr.indptr[i]);
        }
        Ok(())
    }

    async fn end_sparse(&mut self) -> Result<()> {
        let state = self.sparse_state.take()
            .ok_or_else(|| ScxError::InvalidFormat("end_sparse called without begin_sparse".into()))?;

        let grp = self.file.group(&state.group_path)?;

        // Write indptr.
        let max_ptr = state.indptr.iter().copied().max().unwrap_or(0);
        if max_ptr > i32::MAX as u64 {
            let v: Vec<i64> = state.indptr.iter().map(|&x| x as i64).collect();
            let ds = grp.new_dataset::<i64>().shape(v.len()).create("indptr")?;
            ds.write(&Array1::from_vec(v))?;
        } else {
            let v: Vec<i32> = state.indptr.iter().map(|&x| x as i32).collect();
            let ds = grp.new_dataset::<i32>().shape(v.len()).create("indptr")?;
            ds.write(&Array1::from_vec(v))?;
        }

        // Write dims attribute: [nrows, ncols].
        let (nrows, ncols) = state.shape;
        let dims = vec![nrows as i32, ncols as i32];
        let attr = grp.new_attr::<i32>().shape(2).create("dims")?;
        attr.write(&Array1::from_vec(dims))?;

        Ok(())
    }

    async fn write_varm(&mut self, varm: &Varm) -> Result<()> {
        if varm.map.is_empty() {
            return Ok(());
        }
        // reductions/ may already exist (write_obsm creates it)
        let reds_grp = match self.file.group("reductions") {
            Ok(g)  => g,
            Err(_) => self.file.create_group("reductions")?,
        };
        for (key, mat) in &varm.map {
            let red_name = key.strip_prefix("X_").unwrap_or(key.as_str());
            // reduction sub-group may already exist from write_obsm
            let red_grp = match reds_grp.group(red_name) {
                Ok(g)  => g,
                Err(_) => reds_grp.create_group(red_name)?,
            };
            let (n_vars, k) = mat.shape;
            // IR: (n_vars, k) row-major → H5Seurat: (k, n_vars)
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
        }
        Ok(())
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let csr = &chunk.data;
        let nnz = csr.indices.len();

        if nnz > 0 {
            let data_path    = format!("assays/{}/{}/data",    self.assay, self.layer);
            let indices_path = format!("assays/{}/{}/indices", self.assay, self.layer);

            // Append values
            let data_ds = self.file.dataset(&data_path)?;
            let old_len = data_ds.shape()[0];
            let new_len = old_len + nnz;
            data_ds.resize(new_len)?;
            match self.dtype {
                DataType::F32 => {
                    let v: Vec<f32> = csr.data.to_f64().into_iter().map(|x| x as f32).collect();
                    data_ds.write_slice(&Array1::from_vec(v), s![old_len..new_len])?;
                }
                DataType::F64 => {
                    data_ds.write_slice(&Array1::from_vec(csr.data.to_f64()), s![old_len..new_len])?;
                }
                DataType::I32 => {
                    let v: Vec<i32> = csr.data.to_f64().into_iter().map(|x| x as i32).collect();
                    data_ds.write_slice(&Array1::from_vec(v), s![old_len..new_len])?;
                }
                DataType::U32 => {
                    let v: Vec<u32> = csr.data.to_f64().into_iter().map(|x| x as u32).collect();
                    data_ds.write_slice(&Array1::from_vec(v), s![old_len..new_len])?;
                }
            }

            // Append gene indices as i32
            let idx_ds = self.file.dataset(&indices_path)?;
            let old_idx = idx_ds.shape()[0];
            idx_ds.resize(new_len)?;
            let genes_i32: Vec<i32> = csr.indices.iter().map(|&x| x as i32).collect();
            idx_ds.write_slice(&Array1::from_vec(genes_i32), s![old_idx..new_len])?;
        }

        // Accumulate cell-level indptr
        let base = *self.x_indptr.last().unwrap();
        for i in 1..=chunk.nrows {
            self.x_indptr.push(base + csr.indptr[i]);
        }

        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        let layer_path = format!("assays/{}/{}", self.assay, self.layer);

        // Write indptr (i32 if nnz fits, i64 otherwise)
        let max_ptr     = self.x_indptr.iter().copied().max().unwrap_or(0);
        let indptr_path = format!("{layer_path}/indptr");
        if max_ptr > i32::MAX as u64 {
            let v: Vec<i64> = self.x_indptr.iter().map(|&x| x as i64).collect();
            let ds = self.file.new_dataset::<i64>().shape(v.len()).create(indptr_path.as_str())?;
            ds.write(&Array1::from_vec(v))?;
        } else {
            let v: Vec<i32> = self.x_indptr.iter().map(|&x| x as i32).collect();
            let ds = self.file.new_dataset::<i32>().shape(v.len()).create(indptr_path.as_str())?;
            ds.write(&Array1::from_vec(v))?;
        }

        // dims attribute: [n_vars, n_obs]
        let layer_grp = self.file.group(&layer_path)?;
        let dims = vec![self.n_vars as i32, self.n_obs as i32];
        let attr = layer_grp.new_attr::<i32>().shape(2).create("dims")?;
        attr.write(&Array1::from_vec(dims))?;

        tracing::info!(
            n_obs  = self.n_obs,
            n_vars = self.n_vars,
            nnz    = self.x_indptr.last().copied().unwrap_or(0),
            assay  = %self.assay,
            layer  = %self.layer,
            "h5seurat finalized"
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    const GOLDEN: &str = "../../tests/golden/pbmc3k.h5seurat";
    const NORMAN_FIXTURE: &str = "../../tests/fixtures/norman_subset.h5ad";

    fn golden_exists() -> bool {
        std::path::Path::new(GOLDEN).exists()
    }

    fn norman_exists() -> bool {
        std::path::Path::new(NORMAN_FIXTURE).exists()
    }

    #[test]
    fn test_detect_sparse_group_kind_bpcells_layer() {
        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path();

        let file = File::create(path).unwrap();
        file.create_group("assays").unwrap();
        file.create_group("assays/RNA").unwrap();
        let grp = file.create_group("assays/RNA/data").unwrap();

        let version = VarLenUnicode::from_str("packed-uint-matrix-v2").unwrap_or_default();
        let attr = grp.new_attr::<VarLenUnicode>().shape(()).create("version").unwrap();
        attr.write_scalar(&version).unwrap();

        let shape = Array1::from_vec(vec![4u32, 3u32]);
        grp.new_dataset::<u32>().shape(shape.len()).create("shape").unwrap()
            .write(&shape).unwrap();

        let idxptr = Array1::from_vec(vec![0u64, 1, 3, 4, 4]);
        grp.new_dataset::<u64>().shape(idxptr.len()).create("idxptr").unwrap()
            .write(&idxptr).unwrap();

        drop(file);

        let file = File::open(path).unwrap();
        assert_eq!(
            detect_sparse_group_kind(&file, "assays/RNA/data"),
            Some(SparseGroupKind::BpCells)
        );
    }

    #[tokio::test]
    async fn test_open_shape() {
        if !golden_exists() { return; }
        let reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
        let (n_obs, n_vars) = reader.shape();
        assert_eq!(n_obs,  2700,  "expected 2700 cells");
        assert_eq!(n_vars, 13714, "expected 13714 genes");
    }

    #[tokio::test]
    async fn test_obs() {
        if !golden_exists() { return; }
        let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
        let obs = reader.obs().await.unwrap();
        assert_eq!(obs.index.len(), 2700);
        assert!(!obs.columns.is_empty());
        assert!(obs.columns.iter().any(|c| c.name == "nCount_RNA"));
    }

    #[tokio::test]
    async fn test_var() {
        if !golden_exists() { return; }
        let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
        let var = reader.var().await.unwrap();
        assert_eq!(var.index.len(), 13714);
        assert!(!var.columns.is_empty(), "expected meta.features columns");
        assert!(var.columns.iter().any(|c| c.name == "vf.vst.mean"));
        assert!(var.columns.iter().any(|c| c.name == "vf.vst.variable"));
        // vf.vst.variable should be Bool
        let hvg_col = var.columns.iter().find(|c| c.name == "vf.vst.variable").unwrap();
        assert!(matches!(hvg_col.data, crate::ir::ColumnData::Bool(_)));
    }

    #[tokio::test]
    async fn test_obsm() {
        if !golden_exists() { return; }
        let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
        let obsm = reader.obsm().await.unwrap();
        assert!(obsm.map.contains_key("X_pca"),  "missing X_pca");
        assert!(obsm.map.contains_key("X_umap"), "missing X_umap");
        assert_eq!(obsm.map["X_pca"].shape,  (2700, 30));
        assert_eq!(obsm.map["X_umap"].shape, (2700, 2));
    }

    #[tokio::test]
    async fn test_stream_coverage() {
        if !golden_exists() { return; }
        let mut reader = H5SeuratReader::open(GOLDEN, 1000, None, None).unwrap();
        let mut total_cells = 0usize;
        let mut total_nnz   = 0usize;
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            total_cells += chunk.nrows;
            total_nnz   += chunk.data.indices.len();
        }
        assert_eq!(total_cells, 2700);
        assert_eq!(total_nnz,   2282976);
    }

    #[tokio::test]
    async fn test_h5seurat_roundtrip() {
        if !golden_exists() { return; }

        let mut reader = H5SeuratReader::open(GOLDEN, 500, None, None).unwrap();
        let (n_obs, n_vars) = reader.shape();

        let obs  = reader.obs().await.unwrap();
        let var  = reader.var().await.unwrap();
        let obsm = reader.obsm().await.unwrap();
        let uns  = reader.uns().await.unwrap();

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let out = tmp.path().to_path_buf();

        let mut writer = H5SeuratWriter::create(&out, n_obs, n_vars, DataType::F32, None, None, None).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&obsm).await.unwrap();
        writer.write_uns(&uns).await.unwrap();

        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            writer.write_x_chunk(&chunk.unwrap()).await.unwrap();
        }
        writer.finalize().await.unwrap();
        drop(writer);

        // Re-open and verify with H5SeuratReader
        let mut rt = H5SeuratReader::open(&out, 500, None, None).unwrap();
        assert_eq!(rt.shape(), (n_obs, n_vars));

        let rt_obs = rt.obs().await.unwrap();
        assert_eq!(rt_obs.index.len(), n_obs);
        assert_eq!(rt_obs.index[0], obs.index[0]);
        assert_eq!(rt_obs.columns.len(), obs.columns.len());

        let rt_var = rt.var().await.unwrap();
        assert_eq!(rt_var.index.len(), n_vars);
        assert_eq!(rt_var.index[0], var.index[0]);

        let rt_obsm = rt.obsm().await.unwrap();
        assert!(rt_obsm.map.contains_key("X_pca"), "X_pca missing after roundtrip");
        assert_eq!(rt_obsm.map["X_pca"].shape, obsm.map["X_pca"].shape);

        let mut total_nnz = 0usize;
        let mut stream = rt.x_stream();
        while let Some(chunk) = stream.next().await {
            total_nnz += chunk.unwrap().data.indices.len();
        }
        assert_eq!(total_nnz, 2282976, "nnz changed after H5Seurat roundtrip");
    }

    // -----------------------------------------------------------------------
    // data-layer test (synthetic 3×4 matrix written with layer="data")
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_data_layer() {
        // Write a 3-cell × 4-gene matrix under the "data" layer and read it back.
        let n_obs = 3usize;
        let n_vars = 4usize;

        let obs = ObsTable {
            index: (0..n_obs).map(|i| format!("c{i}")).collect(),
            columns: vec![],
        };
        let var = VarTable {
            index: (0..n_vars).map(|i| format!("g{i}")).collect(),
            columns: vec![],
        };
        // Non-zeros at (0,0),(0,2),(1,1),(1,3),(2,0),(2,3) → nnz = 6
        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr:  vec![0, 2, 4, 6],
                indices: vec![0, 2, 1, 3, 0, 3],
                data:    TypedVec::F32(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]),
            },
        };

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let out = tmp.path().to_path_buf();

        let mut writer = H5SeuratWriter::create(&out, n_obs, n_vars, DataType::F32, None, Some("data"), None).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        let mut reader = H5SeuratReader::open(&out, 100, None, Some("data")).unwrap();
        assert_eq!(reader.shape(), (n_obs, n_vars));

        let mut total_nnz = 0usize;
        let mut stream = reader.x_stream();
        while let Some(c) = stream.next().await {
            total_nnz += c.unwrap().data.indices.len();
        }
        assert_eq!(total_nnz, 6, "nnz mismatch for data layer");
    }

    // -----------------------------------------------------------------------
    // uns / misc pass-through test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_uns_misc_passthrough() {
        // Create a minimal valid H5Seurat with H5SeuratWriter, then inject a
        // misc/ group.  Verify that uns() surfaces it as JSON.
        let obs = ObsTable { index: vec!["c0".into()], columns: vec![] };
        let var = VarTable { index: vec!["g0".into()], columns: vec![] };
        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: 1,
            data: SparseMatrixCSR {
                shape: (1, 1),
                indptr:  vec![0, 1],
                indices: vec![0],
                data:    TypedVec::F32(vec![1.0]),
            },
        };

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer = H5SeuratWriter::create(&path, 1, 1, DataType::F32, None, None, None).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();
        writer.write_x_chunk(&chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        // Inject misc/ with a scalar dataset and a numeric array
        {
            let file = File::open_rw(&path).unwrap();
            let misc  = file.create_group("misc").unwrap();
            let ds    = misc.new_dataset::<f64>().shape(3).create("weights").unwrap();
            ds.write(&Array1::from_vec(vec![0.1f64, 0.2, 0.3])).unwrap();
        }

        let mut reader = H5SeuratReader::open(&path, 100, None, None).unwrap();
        let uns = reader.uns().await.unwrap();

        assert!(uns.raw.is_object(), "uns.raw should be a JSON object");
        assert!(uns.raw.get("weights").is_some(), "misc/weights missing from uns");
        assert_eq!(uns.raw["weights"], serde_json::json!([0.1, 0.2, 0.3]));
    }

    // -----------------------------------------------------------------------
    // Slot parity: layers, obsp, varm write → read roundtrip (synthetic)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_slot_parity_roundtrip() {
        // 3 cells × 4 genes synthetic dataset.
        let n_obs  = 3usize;
        let n_vars = 4usize;

        let obs = ObsTable { index: vec!["c0".into(), "c1".into(), "c2".into()], columns: vec![] };
        let var = VarTable {
            index: vec!["g0".into(), "g1".into(), "g2".into(), "g3".into()],
            columns: vec![],
        };

        // X: sparse 3×4, 4 non-zeros
        let x_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape:   (n_obs, n_vars),
                indptr:  vec![0, 2, 3, 4],
                indices: vec![0, 2, 1, 3],
                data:    TypedVec::F32(vec![1.0, 2.0, 3.0, 4.0]),
            },
        };

        // layers["data"]: sparse chunk (n_vars × n_obs stored as CSR in H5Seurat convention)
        let layer_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_vars,
            data: SparseMatrixCSR {
                shape:   (n_vars, n_obs),
                indptr:  vec![0, 1, 2, 3, 3],
                indices: vec![1, 0, 2],
                data:    TypedVec::F32(vec![10.0, 20.0, 30.0]),
            },
        };
        let layer_meta = SparseMatrixMeta {
            name:   "data".into(),
            shape:  (n_vars, n_obs),
            indptr: vec![0, 1, 2, 3, 3],
        };

        // obsp["knn"]: 3×3 cell-cell graph
        let obsp_chunk = MatrixChunk {
            row_offset: 0,
            nrows: n_obs,
            data: SparseMatrixCSR {
                shape:   (n_obs, n_obs),
                indptr:  vec![0, 1, 2, 3],
                indices: vec![1, 2, 0],
                data:    TypedVec::F32(vec![0.5, 0.6, 0.7]),
            },
        };
        let obsp_meta = SparseMatrixMeta {
            name:   "knn".into(),
            shape:  (n_obs, n_obs),
            indptr: vec![0, 1, 2, 3],
        };

        // varm["X_pca"]: 4 genes × 2 PCs
        let varm_mat = DenseMatrix {
            shape: (n_vars, 2),
            data:  vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let mut varm = Varm::default();
        varm.map.insert("X_pca".into(), varm_mat.clone());

        let tmp = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer = H5SeuratWriter::create(&path, n_obs, n_vars, DataType::F32, None, None, None).unwrap();
        writer.write_obs(&obs).await.unwrap();
        writer.write_var(&var).await.unwrap();
        writer.write_obsm(&Embeddings::default()).await.unwrap();
        writer.write_uns(&UnsTable::default()).await.unwrap();

        // Stream layer "data"
        writer.begin_sparse("layers", "data", &layer_meta).await.unwrap();
        writer.write_sparse_chunk(&layer_chunk).await.unwrap();
        writer.end_sparse().await.unwrap();

        // Stream obsp "knn"
        writer.begin_sparse("obsp", "knn", &obsp_meta).await.unwrap();
        writer.write_sparse_chunk(&obsp_chunk).await.unwrap();
        writer.end_sparse().await.unwrap();

        writer.write_varm(&varm).await.unwrap();
        writer.write_x_chunk(&x_chunk).await.unwrap();
        writer.finalize().await.unwrap();
        drop(writer);

        // Re-read and verify
        let mut reader = H5SeuratReader::open(&path, 100, None, None).unwrap();
        assert_eq!(reader.shape(), (n_obs, n_vars));

        let layer_metas = reader.layer_metas().await.unwrap();
        assert!(layer_metas.iter().any(|m| m.name == "data"), "layers['data'] missing");
        // Stream the layer in its own block so the borrow on `reader` ends before
        // the next `&mut self` call.
        let all_indices: Vec<u32> = {
            let lm = layer_metas.iter().find(|m| m.name == "data").unwrap();
            assert_eq!(lm.shape,  layer_meta.shape);
            assert_eq!(lm.indptr, layer_meta.indptr);
            let mut indices = Vec::new();
            let mut stream = reader.layer_stream(lm, 100);
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.unwrap();
                indices.extend_from_slice(&chunk.data.indices);
            }
            indices
        };
        assert_eq!(all_indices, layer_chunk.data.indices);

        let obsp_metas = reader.obsp_metas().await.unwrap();
        assert!(obsp_metas.iter().any(|m| m.name == "knn"), "obsp['knn'] missing");
        let om = obsp_metas.iter().find(|m| m.name == "knn").unwrap();
        assert_eq!(om.shape, obsp_meta.shape);

        let rt_varm = reader.varm().await.unwrap();
        assert!(rt_varm.map.contains_key("X_pca"), "varm['X_pca'] missing");
        let rt_pca = &rt_varm.map["X_pca"];
        assert_eq!(rt_pca.shape, varm_mat.shape);
        for (a, b) in rt_pca.data.iter().zip(varm_mat.data.iter()) {
            assert!((a - b).abs() < 1e-10, "varm data mismatch: {a} vs {b}");
        }
    }

    // --- Norman obs round-trip: H5AD → H5Seurat → read back ---

    #[tokio::test]
    async fn test_norman_obs_roundtrip() {
        use crate::h5ad::H5AdReader;
        use tempfile::NamedTempFile;

        if !norman_exists() { return; }

        // Read the Norman subset H5AD.
        let fixture = std::path::Path::new(NORMAN_FIXTURE);
        let mut src = H5AdReader::open(fixture, 500).unwrap();
        let (n_obs, n_vars) = src.shape();
        let src_obs  = src.obs().await.unwrap();
        let src_var  = src.var().await.unwrap();
        let src_obsm = src.obsm().await.unwrap();
        let src_uns  = src.uns().await.unwrap();

        let src_col_names: Vec<&str> = src_obs.columns.iter().map(|c| c.name.as_str()).collect();
        eprintln!("source obs columns: {src_col_names:?}");

        // Convert to H5Seurat.
        let tmp = NamedTempFile::with_suffix(".h5seurat").unwrap();
        let out = tmp.path().to_path_buf();

        let mut writer = H5SeuratWriter::create(&out, n_obs, n_vars, src.dtype(), None, None, None).unwrap();
        writer.write_obs(&src_obs).await.unwrap();
        writer.write_var(&src_var).await.unwrap();
        writer.write_obsm(&src_obsm).await.unwrap();
        writer.write_uns(&src_uns).await.unwrap();
        {
            let mut stream = src.x_stream();
            while let Some(chunk) = stream.next().await {
                writer.write_x_chunk(&chunk.unwrap()).await.unwrap();
            }
        }
        // Skip layers — the "counts" layer would collide with the X path in H5Seurat.
        // Obs fidelity is what this test exercises.
        writer.finalize().await.unwrap();
        drop(writer);

        // Read back via H5SeuratReader and check obs fidelity.
        let mut rt = H5SeuratReader::open(&out, 500, None, None).unwrap();
        assert_eq!(rt.shape(), (n_obs, n_vars), "shape mismatch after round-trip");

        let rt_obs = rt.obs().await.unwrap();
        assert_eq!(rt_obs.index.len(), n_obs, "obs index length mismatch");

        // Every source column must survive the round-trip.
        for src_col in &src_obs.columns {
            let rt_col = rt_obs.columns.iter().find(|c| c.name == src_col.name)
                .unwrap_or_else(|| panic!("obs column '{}' missing after round-trip", src_col.name));

            // Dtype class must be preserved.
            let src_kind = std::mem::discriminant(&src_col.data);
            let rt_kind  = std::mem::discriminant(&rt_col.data);
            assert_eq!(src_kind, rt_kind,
                "obs column '{}' changed dtype after round-trip", src_col.name);
        }

        eprintln!("norman obs round-trip OK: {n_obs} cells, {} columns",
                  rt_obs.columns.len());
    }
}
