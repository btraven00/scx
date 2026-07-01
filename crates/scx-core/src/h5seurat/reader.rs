use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use serde_json;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::types::{FloatSize, TypeDescriptor, VarLenUnicode};
use hdf5::File;
use ndarray::s;

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
    },
    stream::DatasetReader,
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
        // The requested assay may not exist (e.g. multimodal references default
        // to "SCT", not "RNA"); fall back to the file's active/first assay.
        let assay = resolve_assay(&file, &assay)?;
        let dims_path = format!("assays/{assay}/{layer}");
        let dims_grp = file
            .group(&dims_path)
            .map_err(|_| missing_layer_err(&file, &assay, &layer))?;

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
                    indptr.len(),
                    n_obs + 1
                )));
            }

            let data_ds_path = format!("{dims_path}/data");
            let dtype = detect_dtype(&file, &data_ds_path)?;
            XBackend::DgCMatrix { indptr, dtype }
        };

        Ok(Self {
            path,
            assay,
            layer,
            n_obs,
            n_vars,
            chunk_size,
            x_backend,
        })
    }
}

// ---------------------------------------------------------------------------
// Sync helpers
// ---------------------------------------------------------------------------

/// Names of the assays present under `/assays`.
fn list_assays(file: &File) -> Vec<String> {
    file.group("assays")
        .and_then(|g| g.member_names())
        .unwrap_or_default()
}

/// The file's active assay (root `active.assay` attribute), if any. SeuratDisk
/// may store it as variable- or fixed-length ASCII/UTF-8, so try each.
fn active_assay(file: &File) -> Option<String> {
    let attr = file.group("/").ok()?.attr("active.assay").ok()?;
    if let Ok(s) = attr.read_scalar::<VarLenUnicode>() {
        return Some(s.to_string());
    }
    if let Ok(s) = attr.read_scalar::<hdf5::types::VarLenAscii>() {
        return Some(s.to_string());
    }
    if let Ok(s) = attr.read_scalar::<hdf5::types::FixedAscii<64>>() {
        return Some(s.to_string());
    }
    None
}

/// Conventional main-expression assays, preferred over ancillary ones (e.g. the
/// antibody `ADT` assay) when the file has no readable active assay.
const PREFERRED_ASSAYS: &[&str] = &["SCT", "RNA", "originalexp", "spliced"];

/// Resolve which assay to read: the requested one if present, else the file's
/// active assay, else a conventional main assay, else the first. Errors only
/// when there are no assays.
fn resolve_assay(file: &File, requested: &str) -> Result<String> {
    if file.group(&format!("assays/{requested}")).is_ok() {
        return Ok(requested.to_string());
    }
    let available = list_assays(file);
    if available.is_empty() {
        return Err(ScxError::InvalidFormat(
            "no assays found under /assays in this H5Seurat file".into(),
        ));
    }
    let chosen = active_assay(file)
        .filter(|a| available.iter().any(|x| x == a))
        .or_else(|| {
            PREFERRED_ASSAYS
                .iter()
                .find(|p| available.iter().any(|a| a == *p))
                .map(|p| p.to_string())
        })
        .unwrap_or_else(|| available[0].clone());
    tracing::warn!(
        requested = %requested,
        chosen = %chosen,
        available = ?available,
        "assay '{requested}' not found; using assay '{chosen}' — pass --assay to override"
    );
    Ok(chosen)
}

/// A helpful error for a missing layer: name the assay and list what's there.
fn missing_layer_err(file: &File, assay: &str, layer: &str) -> ScxError {
    let layers = file
        .group(&format!("assays/{assay}"))
        .and_then(|g| g.member_names())
        .unwrap_or_default();
    ScxError::InvalidFormat(format!(
        "layer '{layer}' not found in assay '{assay}' (looked for assays/{assay}/{layer}); \
         available layers: {layers:?} — pass --layer to choose one"
    ))
}

fn read_indptr_from(file: &File, path: &str) -> Result<Vec<u64>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(_) => Ok(ds.read_1d::<f64>()?.iter().map(|&x| x as u64).collect()),
        TypeDescriptor::Integer(_) => Ok(ds.read_1d::<i32>()?.iter().map(|&x| x as u64).collect()),
        other => Err(ScxError::InvalidFormat(format!(
            "unexpected indptr type at {path}: {:?}",
            other
        ))),
    }
}

fn read_indices_at(file: &File, path: &str, start: usize, end: usize) -> Result<Vec<u32>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) => Ok(ds
            .read_slice_1d::<i32, _>(s![start..end])?
            .iter()
            .map(|&x| x as u32)
            .collect()),
        _ => Err(ScxError::InvalidFormat(format!(
            "unexpected indices type at {path}"
        ))),
    }
}

fn detect_dtype(file: &File, path: &str) -> Result<DataType> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_) => DataType::F64,
        TypeDescriptor::Integer(_) => DataType::I32,
        _ => DataType::F32,
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
            "unsupported string type {:?} at '{path}'",
            other
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
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
    let nnz_end = indptr[cell_end] as usize;
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
            DataType::F32 => {
                TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            DataType::F64 => {
                TypedVec::F64(ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            DataType::I32 => {
                TypedVec::I32(ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            DataType::U32 => {
                TypedVec::U32(ds.read_slice_1d::<u32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
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
                Err(e) => {
                    tracing::warn!("skipping factor column '{name}': {e}");
                    continue;
                }
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
                Err(e) => {
                    tracing::warn!("skipping obs column '{name}': {e}");
                    continue;
                }
            }
        };

        columns.push(Column {
            name: name.clone(),
            data: col_data,
        });
    }

    Ok(ObsTable { index, columns })
}

fn read_factor_column(file: &File, grp_path: &str) -> Result<ColumnData> {
    let values_path = format!("{grp_path}/values");
    let levels_path = format!("{grp_path}/levels");
    let codes: Vec<u32> = file
        .dataset(&values_path)?
        .read_1d::<i32>()?
        .iter()
        .map(|&v| (v - 1).max(0) as u32) // 1-indexed → 0-indexed
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
        TypeDescriptor::Float(_) => Ok(ColumnData::Float(ds.read_1d::<f64>()?.to_vec())),
        TypeDescriptor::Integer(_) => Ok(ColumnData::Int(ds.read_1d::<i32>()?.to_vec())),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            Ok(ColumnData::String(read_strings(file, ds_path)?))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported column type {:?} at {ds_path}",
            other
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
                let is_group = file.group(&ds_path).is_ok() && file.dataset(&ds_path).is_err();
                let col_data = if is_group {
                    match read_factor_column(&file, &ds_path) {
                        Ok(cd) => cd,
                        Err(e) => {
                            tracing::warn!("skipping var factor '{name}': {e}");
                            continue;
                        }
                    }
                } else if logicals.contains(name.as_str()) {
                    let ds = match file.dataset(&ds_path) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!("skipping var logical '{name}': {e}");
                            continue;
                        }
                    };
                    let vals: Vec<i32> = ds.read_1d::<i32>()?.to_vec();
                    ColumnData::Bool(vals.into_iter().map(|v| v == 1).collect())
                } else {
                    match read_meta_column(&file, &ds_path) {
                        Ok(cd) => cd,
                        Err(e) => {
                            tracing::warn!("skipping var column '{name}': {e}");
                            continue;
                        }
                    }
                };
                cols.push(Column {
                    name,
                    data: col_data,
                });
            }
            cols
        }
    };

    Ok(VarTable { index, columns })
}

fn read_obsm_sync(path: &Path, n_obs: usize) -> Result<Embeddings> {
    let file = File::open(path)?;
    let reds_grp = match file.group("reductions") {
        Ok(g) => g,
        Err(_) => return Ok(Embeddings::default()),
    };
    let mut map = HashMap::new();
    for red_name in reds_grp.member_names()? {
        let ds_path = format!("reductions/{red_name}/cell.embeddings");
        let ds = match file.dataset(&ds_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let arr: ndarray::Array2<f64> = match ds.read::<f64, ndarray::Ix2>() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("skipping reduction '{red_name}': {e}");
                continue;
            }
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
        map.insert(
            obsm_key,
            DenseMatrix {
                shape,
                data: arr.into_raw_vec_and_offset().0,
            },
        );
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
        Ok(g) => g,
        Err(_) => return serde_json::Value::Null,
    };
    let members = grp.member_names().unwrap_or_default();
    let mut map = serde_json::Map::new();
    for name in members {
        let child = format!("{group_path}/{name}");
        let is_grp = file.group(&child).is_ok() && file.dataset(&child).is_err();
        let value = if is_grp {
            seurat_walk_group(file, &child)
        } else {
            seurat_ds_to_json(file, &child).unwrap_or(serde_json::Value::Null)
        };
        map.insert(name, value);
    }
    serde_json::Value::Object(map)
}

fn seurat_ds_to_json(file: &File, path: &str) -> Result<serde_json::Value> {
    let ds = file.dataset(path)?;
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
                    strings.into_iter().next().unwrap_or_default(),
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
    Ok(UnsTable {
        raw: seurat_walk_group(&file, "misc"),
    })
}

// ---------------------------------------------------------------------------
// Slot parity helpers
// ---------------------------------------------------------------------------

/// Classify an H5Seurat sparse group as either classic dgCMatrix storage or
/// BPCells-backed storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SparseGroupKind {
    DgCMatrix,
    BpCells,
}

pub(super) fn detect_sparse_group_kind(file: &File, group_path: &str) -> Option<SparseGroupKind> {
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

    if matches!(
        detect_sparse_group_kind(file, group_path),
        Some(SparseGroupKind::BpCells)
    ) {
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
                    idxptr_ds
                        .read_1d::<u32>()?
                        .iter()
                        .map(|&x| x as u64)
                        .collect()
                }
            }
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "unexpected BPCells idxptr type at {group_path}: {:?}",
                    other
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
    Ok(SparseMatrixMeta {
        name: name.to_string(),
        shape: (nrows, ncols),
        indptr,
    })
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
    let nnz_end = meta.indptr[row_end] as usize;
    let nnz = nnz_end - nnz_start;

    let indices: Vec<u32> = if nnz > 0 {
        read_indices_at(&file, &format!("{group_path}/indices"), nnz_start, nnz_end)?
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

fn read_layer_metas_sync(
    path: &Path,
    assay: &str,
    primary_layer: &str,
) -> Result<Vec<SparseMatrixMeta>> {
    let file = File::open(path)?;
    let assay_grp = match file.group(&format!("assays/{assay}")) {
        Err(_) => return Ok(Vec::new()),
        Ok(g) => g,
    };
    let mut metas = Vec::new();
    for name in assay_grp.member_names().unwrap_or_default() {
        if name == primary_layer {
            continue;
        }
        let grp_path = format!("assays/{assay}/{name}");

        if detect_sparse_group_kind(&file, &grp_path).is_none() {
            continue;
        }

        match seurat_read_sparse_meta(&file, &name, &grp_path) {
            Ok(m) => metas.push(m),
            Err(e) => tracing::warn!("skipping assay layer '{name}': {e}"),
        }
    }
    Ok(metas)
}

fn read_obsp_metas_sync(path: &Path) -> Result<Vec<SparseMatrixMeta>> {
    let file = File::open(path)?;
    let grp = match file.group("graphs") {
        Err(_) => return Ok(Vec::new()),
        Ok(g) => g,
    };
    let mut metas = Vec::new();
    for name in grp.member_names().unwrap_or_default() {
        let grp_path = format!("graphs/{name}");
        if detect_sparse_group_kind(&file, &grp_path).is_none() {
            continue;
        }
        match seurat_read_sparse_meta(&file, &name, &grp_path) {
            Ok(m) => metas.push(m),
            Err(e) => tracing::warn!("skipping graph '{name}': {e}"),
        }
    }
    Ok(metas)
}

fn read_varm_sync(path: &Path, n_vars: usize) -> Result<Varm> {
    let file = File::open(path)?;
    let reds_grp = match file.group("reductions") {
        Err(_) => return Ok(Varm::default()),
        Ok(g) => g,
    };
    let mut map = HashMap::new();
    for red_name in reds_grp.member_names().unwrap_or_default() {
        let ds_path = format!("reductions/{red_name}/feature.loadings");
        let ds = match file.dataset(&ds_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let arr: ndarray::Array2<f64> = match ds.read::<f64, ndarray::Ix2>() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("skipping varm '{red_name}': {e}");
                continue;
            }
        };
        // feature.loadings stored as (k, n_vars) — transpose to (n_vars, k)
        let arr = if arr.shape()[1] == n_vars && arr.shape()[0] != n_vars {
            arr.t().as_standard_layout().into_owned()
        } else {
            arr
        };
        let shape = (arr.shape()[0], arr.shape()[1]);
        let varm_key = format!("X_{}", red_name.to_lowercase());
        map.insert(
            varm_key,
            DenseMatrix {
                shape,
                data: arr.into_raw_vec_and_offset().0,
            },
        );
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
    Ok(Box::new(H5SeuratReader::open(
        path,
        chunk_size,
        Some(assay),
        Some(layer),
    )?))
}

// ---------------------------------------------------------------------------
// DatasetReader impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetReader for H5SeuratReader {
    fn x_indptr(&self) -> &[u64] {
        match &self.x_backend {
            XBackend::DgCMatrix { indptr, .. } => indptr.as_slice(),
            XBackend::BpCells => &[],
        }
    }

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
        let path = self.path.clone();
        let assay = self.assay.clone();
        let grp_path = format!("assays/{}/{}", assay, meta.name);
        let n_rows = meta.shape.0;

        let is_bpcells = {
            let file = File::open(&path);
            match file {
                Ok(file) => matches!(
                    detect_sparse_group_kind(&file, &grp_path),
                    Some(SparseGroupKind::BpCells)
                ),
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
                    if row_start >= n_rows {
                        return None;
                    }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk = reader.read_chunk(row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        } else {
            Box::pin(stream::unfold(0usize, move |row_start| {
                let path = path.clone();
                let grp_path = grp_path.clone();
                async move {
                    if row_start >= n_rows {
                        return None;
                    }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk =
                        seurat_read_sparse_chunk(&path, &grp_path, meta, row_start, row_end);
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
        let path = self.path.clone();
        let grp_path = format!("graphs/{}", meta.name);
        let n_rows = meta.shape.0;

        let is_bpcells = {
            let file = File::open(&path);
            match file {
                Ok(file) => matches!(
                    detect_sparse_group_kind(&file, &grp_path),
                    Some(SparseGroupKind::BpCells)
                ),
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
                    if row_start >= n_rows {
                        return None;
                    }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk = reader.read_chunk(row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        } else {
            Box::pin(stream::unfold(0usize, move |row_start| {
                let path = path.clone();
                let grp_path = grp_path.clone();
                async move {
                    if row_start >= n_rows {
                        return None;
                    }
                    let row_end = (row_start + chunk_size).min(n_rows);
                    let chunk =
                        seurat_read_sparse_chunk(&path, &grp_path, meta, row_start, row_end);
                    Some((chunk, row_end))
                }
            }))
        }
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        match &self.x_backend {
            XBackend::DgCMatrix { indptr, dtype } => {
                let path = self.path.clone();
                let assay = self.assay.clone();
                let layer = self.layer.clone();
                let n_obs = self.n_obs;
                let n_vars = self.n_vars;
                let chunk_size = self.chunk_size;
                let indptr = indptr.clone();
                let dtype = *dtype;

                Box::pin(stream::unfold(0usize, move |cell_start| {
                    let path = path.clone();
                    let assay = assay.clone();
                    let layer = layer.clone();
                    let indptr = indptr.clone();
                    async move {
                        if cell_start >= n_obs {
                            return None;
                        }
                        let cell_end = (cell_start + chunk_size).min(n_obs);
                        let chunk = read_chunk_sync(
                            &path, &assay, &layer, &indptr, cell_start, cell_end, n_vars, dtype,
                        );
                        Some((chunk, cell_end))
                    }
                }))
            }
            XBackend::BpCells => {
                let path = self.path.clone();
                let assay = self.assay.clone();
                let layer = self.layer.clone();
                let n_obs = self.n_obs;
                let chunk_size = self.chunk_size;

                let bp_reader = {
                    let file = match File::open(&path) {
                        Ok(file) => file,
                        Err(e) => {
                            return Box::pin(stream::once(async move { Err(ScxError::from(e)) }));
                        }
                    };
                    let grp_path = match candidate_group_paths(&assay, &layer)
                        .into_iter()
                        .find(|p| file.group(p).is_ok())
                    {
                        Some(p) => p,
                        None => {
                            return Box::pin(stream::once(async move {
                                Err(ScxError::InvalidFormat(
                                    "missing assay/layer group for BPCells backend".into(),
                                ))
                            }));
                        }
                    };
                    match crate::h5bpcells::open_bpcells_h5(&file, &grp_path, chunk_size) {
                        Ok(reader) => reader,
                        Err(e) => return Box::pin(stream::once(async move { Err(e) })),
                    }
                };

                Box::pin(stream::unfold(0usize, move |cell_start| {
                    let reader = bp_reader.clone();
                    async move {
                        if cell_start >= n_obs {
                            return None;
                        }
                        let cell_end = (cell_start + chunk_size).min(n_obs);
                        let chunk = reader.read_chunk(cell_start, cell_end);
                        Some((chunk, cell_end))
                    }
                }))
            }
        }
    }
}
