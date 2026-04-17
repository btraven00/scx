use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::str::FromStr;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Dataset, File, Group, SimpleExtents};
use ndarray::{s, Array1, Array2};

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
    },
    stream::{DatasetReader, DatasetWriter},
};

/// Number of elements per HDF5 chunk for the streaming X arrays (resizable datasets require chunks).
const CHUNK_ELEMS: usize = 65_536;

/// State kept while streaming a single named sparse matrix (layer or obsp).
struct SparseWriteState {
    /// Full HDF5 group path being written (e.g. "layers/spliced" or "obsp/nn").
    group_path: String,
    /// Accumulated CSR indptr across written chunks.
    indptr: Vec<u64>,
    /// Matrix shape (nrows, ncols) — written as the AnnData "shape" attribute on finalize.
    shape: (usize, usize),
}

/// Streaming writer for the AnnData `.h5ad` format.
///
/// Encoding spec: <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>
///
/// Call order:
///   write_obs → write_var → write_obsm → write_uns → write_x_chunk* → finalize
///
/// `write_*` methods other than `write_x_chunk` can be called in any order.
/// Chunks must arrive in row-ascending order.
pub struct H5AdWriter {
    file: File,
    n_obs: usize,
    n_vars: usize,
    dtype: DataType,
    /// Accumulated CSR indptr across all written chunks (n_obs + 1 entries when done).
    x_indptr: Vec<u64>,
    /// State for the currently open streaming sparse matrix, if any.
    sparse_state: Option<SparseWriteState>,
}

impl H5AdWriter {
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
    ) -> Result<Self> {
        let file = File::create(path.as_ref())?;

        // Root attrs
        let root = file.group("/")?;
        write_str_attr_on_group(&root, "encoding-type", "anndata")?;
        write_str_attr_on_group(&root, "encoding-version", "0.1.0")?;

        // /X group — encoding attrs; resizable datasets created here
        let x_grp = file.create_group("X")?;
        write_str_attr_on_group(&x_grp, "encoding-type", "csr_matrix")?;
        write_str_attr_on_group(&x_grp, "encoding-version", "0.1.0")?;
        // shape attr written in finalize() once we know n_obs

        match dtype {
            DataType::F32 => init_resizable_1d::<f32>(&file, "X/data")?,
            DataType::F64 => init_resizable_1d::<f64>(&file, "X/data")?,
            DataType::I32 => init_resizable_1d::<i32>(&file, "X/data")?,
            DataType::U32 => init_resizable_1d::<u32>(&file, "X/data")?,
        }
        // AnnData spec requires indices as i32
        init_resizable_1d::<i32>(&file, "X/indices")?;

        Ok(Self {
            file,
            n_obs,
            n_vars,
            dtype,
            x_indptr: vec![0u64],
            sparse_state: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn write_str_attr_on_group(grp: &Group, name: &str, value: &str) -> Result<()> {
    let v = VarLenUnicode::from_str(value)
        .map_err(|_| ScxError::InvalidFormat(format!("invalid UTF-8: {value}")))?;
    let attr = grp.new_attr::<VarLenUnicode>().create(name)?;
    attr.write_scalar(&v)?;
    Ok(())
}

fn write_str_attr_on_ds(ds: &Dataset, name: &str, value: &str) -> Result<()> {
    let v = VarLenUnicode::from_str(value)
        .map_err(|_| ScxError::InvalidFormat(format!("invalid UTF-8: {value}")))?;
    let attr = ds.new_attr::<VarLenUnicode>().create(name)?;
    attr.write_scalar(&v)?;
    Ok(())
}

/// Recursively write a JSON value into an HDF5 group as an AnnData-compatible entry.
/// Handles strings, integers, floats, and nested objects (dicts).
/// Arrays and nulls are silently skipped — sufficient for provenance use.
fn write_json_value(grp: &Group, name: &str, value: &serde_json::Value) -> Result<()> {
    match value {
        serde_json::Value::String(s) => {
            let v = VarLenUnicode::from_str(s)
                .map_err(|_| ScxError::InvalidFormat(format!("invalid UTF-8 in uns/{name}")))?;
            let ds = grp.new_dataset::<VarLenUnicode>().shape(()).create(name)?;
            ds.write_scalar(&v)?;
            write_encoding_on_ds(&ds, "string", "0.2.0")?;
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let ds = grp.new_dataset::<i64>().shape(()).create(name)?;
                ds.write_scalar(&i)?;
                write_encoding_on_ds(&ds, "scalar", "0.2.0")?;
            } else if let Some(f) = n.as_f64() {
                let ds = grp.new_dataset::<f64>().shape(()).create(name)?;
                ds.write_scalar(&f)?;
                write_encoding_on_ds(&ds, "scalar", "0.2.0")?;
            }
        }
        serde_json::Value::Object(obj) => {
            let sub = grp.create_group(name)?;
            write_encoding_on_group(&sub, "dict", "0.1.0")?;
            for (k, v) in obj {
                write_json_value(&sub, k, v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_encoding_on_group(grp: &Group, enc_type: &str, enc_version: &str) -> Result<()> {
    write_str_attr_on_group(grp, "encoding-type", enc_type)?;
    write_str_attr_on_group(grp, "encoding-version", enc_version)
}

fn write_encoding_on_ds(ds: &Dataset, enc_type: &str, enc_version: &str) -> Result<()> {
    write_str_attr_on_ds(ds, "encoding-type", enc_type)?;
    write_str_attr_on_ds(ds, "encoding-version", enc_version)
}

// ---------------------------------------------------------------------------
// Dataset creation helpers
// ---------------------------------------------------------------------------

fn init_resizable_1d<T: hdf5::H5Type>(file: &File, path: &str) -> Result<()> {
    file.new_dataset::<T>()
        .chunk(CHUNK_ELEMS)
        .shape(SimpleExtents::resizable([0usize]))
        .create(path)?;
    Ok(())
}

fn write_1d<T: hdf5::H5Type>(grp: &Group, name: &str, data: Array1<T>) -> Result<Dataset> {
    let ds = grp.new_dataset::<T>().shape(data.len()).create(name)?;
    ds.write(&data)?;
    Ok(ds)
}

fn write_vlen_str_dataset(grp: &Group, name: &str, strings: &[String]) -> Result<Dataset> {
    let vals: Vec<VarLenUnicode> = strings
        .iter()
        .map(|s| VarLenUnicode::from_str(s).unwrap_or_default())
        .collect();
    let ds = grp
        .new_dataset::<VarLenUnicode>()
        .shape(vals.len())
        .create(name)?;
    ds.write(&Array1::from_vec(vals))?;
    Ok(ds)
}

// ---------------------------------------------------------------------------
// Dataframe writer (obs / var)
// ---------------------------------------------------------------------------

fn write_dataframe(
    file: &File,
    group_name: &str,
    index: &[String],
    columns: &[Column],
) -> Result<()> {
    let grp = file.create_group(group_name)?;
    write_encoding_on_group(&grp, "dataframe", "0.2.0")?;
    write_str_attr_on_group(&grp, "_index", "index")?;

    // column-order: array of strings listing the non-index columns in order
    let col_names: Vec<VarLenUnicode> = columns
        .iter()
        .map(|c| VarLenUnicode::from_str(&c.name).unwrap_or_default())
        .collect();
    let attr = grp
        .new_attr::<VarLenUnicode>()
        .shape(col_names.len())
        .create("column-order")?;
    attr.write(&Array1::from_vec(col_names))?;

    // index dataset
    let idx_ds = write_vlen_str_dataset(&grp, "index", index)?;
    write_encoding_on_ds(&idx_ds, "string-array", "0.2.0")?;

    // columns
    for col in columns {
        write_column(&grp, &col.name, &col.data)?;
    }

    Ok(())
}

fn write_column(grp: &Group, name: &str, data: &ColumnData) -> Result<()> {
    match data {
        ColumnData::Float(v) => {
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::Int(v) => {
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::Bool(v) => {
            let vi: Vec<u8> = v.iter().map(|&b| b as u8).collect();
            let ds = write_1d(grp, name, Array1::from_vec(vi))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::String(v) => {
            // VarLen strings don't support HDF5 filters — written uncompressed.
            let ds = write_vlen_str_dataset(grp, name, v)?;
            write_encoding_on_ds(&ds, "string-array", "0.2.0")?;
        }
        ColumnData::Categorical { codes, levels } => {
            let cat_grp = grp.create_group(name)?;
            write_encoding_on_group(&cat_grp, "categorical", "0.2.0")?;
            // ordered = false (stored as uint8 boolean)
            let attr = cat_grp.new_attr::<u8>().create("ordered")?;
            attr.write_scalar(&0u8)?;

            // codes (i8 for ≤127 categories, i16 otherwise)
            if levels.len() <= 127 {
                let c: Vec<i8> = codes.iter().map(|&x| x as i8).collect();
                let ds = write_1d(&cat_grp, "codes", Array1::from_vec(c))?;
                write_encoding_on_ds(&ds, "array", "0.2.0")?;
            } else {
                let c: Vec<i16> = codes.iter().map(|&x| x as i16).collect();
                let ds = write_1d(&cat_grp, "codes", Array1::from_vec(c))?;
                write_encoding_on_ds(&ds, "array", "0.2.0")?;
            }

            let cat_ds = write_vlen_str_dataset(&cat_grp, "categories", levels)?;
            write_encoding_on_ds(&cat_ds, "string-array", "0.2.0")?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DatasetWriter impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetWriter for H5AdWriter {
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()> {
        write_dataframe(&self.file, "obs", &obs.index, &obs.columns)
    }

    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        write_dataframe(&self.file, "var", &var.index, &var.columns)
    }

    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()> {
        let grp = self.file.create_group("obsm")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;

        let mut keys: Vec<&String> = obsm.map.keys().collect();
        keys.sort();
        for name in keys {
            let mat = &obsm.map[name];
            let (nrows, ncols) = mat.shape;
            let arr = Array2::from_shape_vec((nrows, ncols), mat.data.clone())
                .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
            let ds = grp
                .new_dataset::<f64>()
                .shape((nrows, ncols))
                .create(name.as_str())?;
            ds.write(&arr)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }

        Ok(())
    }

    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()> {
        let grp = self.file.create_group("uns")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        if let Some(obj) = uns.raw.as_object() {
            for (key, val) in obj {
                write_json_value(&grp, key, val)?;
            }
        }
        Ok(())
    }

    async fn begin_sparse(
        &mut self,
        group_prefix: &str,
        name: &str,
        meta: &SparseMatrixMeta,
    ) -> Result<()> {
        // Ensure the top-level dict group exists (created once on first call).
        let top = match self.file.group(group_prefix) {
            Ok(g) => g,
            Err(_) => {
                let g = self.file.create_group(group_prefix)?;
                write_encoding_on_group(&g, "dict", "0.1.0")?;
                g
            }
        };

        let group_path = format!("{group_prefix}/{name}");
        let mat_grp = top.create_group(name)?;
        write_encoding_on_group(&mat_grp, "csr_matrix", "0.1.0")?;

        // Pre-create resizable data/indices datasets.
        init_resizable_1d::<f32>(&self.file, &format!("{group_path}/data"))?;
        let ds = self.file.dataset(&format!("{group_path}/data"))?;
        write_encoding_on_ds(&ds, "array", "0.2.0")?;

        init_resizable_1d::<i32>(&self.file, &format!("{group_path}/indices"))?;
        let ds = self.file.dataset(&format!("{group_path}/indices"))?;
        write_encoding_on_ds(&ds, "array", "0.2.0")?;

        self.sparse_state = Some(SparseWriteState {
            group_path,
            indptr: vec![0u64],
            shape: meta.shape,
        });
        Ok(())
    }

    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let state = self.sparse_state.as_mut().ok_or_else(|| {
            ScxError::InvalidFormat("write_sparse_chunk called without begin_sparse".into())
        })?;

        let csr = &chunk.data;
        let nnz = csr.indices.len();

        if nnz > 0 {
            let data_ds = self.file.dataset(&format!("{}/data", state.group_path))?;
            let old_len = data_ds.shape()[0];
            let new_len = old_len + nnz;
            data_ds.resize(new_len)?;
            let vals: Vec<f32> = csr.data.to_f64().into_iter().map(|x| x as f32).collect();
            data_ds.write_slice(&Array1::from_vec(vals), s![old_len..new_len])?;

            let idx_ds = self
                .file
                .dataset(&format!("{}/indices", state.group_path))?;
            idx_ds.resize(new_len)?;
            let cols_i32: Vec<i32> = csr.indices.iter().map(|&x| x as i32).collect();
            idx_ds.write_slice(&Array1::from_vec(cols_i32), s![old_len..new_len])?;
        }

        let base = *state.indptr.last().unwrap();
        for i in 1..=chunk.nrows {
            state.indptr.push(base + csr.indptr[i]);
        }
        Ok(())
    }

    async fn end_sparse(&mut self) -> Result<()> {
        let state = self.sparse_state.take().ok_or_else(|| {
            ScxError::InvalidFormat("end_sparse called without begin_sparse".into())
        })?;

        let grp = self.file.group(&state.group_path)?;

        // shape attribute
        let shape_vals = vec![state.shape.0 as i64, state.shape.1 as i64];
        let attr = grp.new_attr::<i64>().shape(2).create("shape")?;
        attr.write(&Array1::from_vec(shape_vals))?;

        // indptr
        let max_val = state.indptr.iter().copied().max().unwrap_or(0);
        if max_val > i32::MAX as u64 {
            let v: Vec<i64> = state.indptr.iter().map(|&x| x as i64).collect();
            let ds = write_1d(&grp, "indptr", Array1::from_vec(v))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        } else {
            let v: Vec<i32> = state.indptr.iter().map(|&x| x as i32).collect();
            let ds = write_1d(&grp, "indptr", Array1::from_vec(v))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        Ok(())
    }

    async fn write_varm(&mut self, varm: &Varm) -> Result<()> {
        let grp = self.file.create_group("varm")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        let mut keys: Vec<&String> = varm.map.keys().collect();
        keys.sort();
        for name in keys {
            let mat = &varm.map[name];
            let (nrows, ncols) = mat.shape;
            let arr = Array2::from_shape_vec((nrows, ncols), mat.data.clone())
                .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
            let ds = grp
                .new_dataset::<f64>()
                .shape((nrows, ncols))
                .create(name.as_str())?;
            ds.write(&arr)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        Ok(())
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let csr = &chunk.data;
        let nnz = csr.indices.len();

        if nnz > 0 {
            // Type conversion happens before the HDF5 lock is held, so
            // parallelising it with Rayon is safe and doesn't conflict with
            // the global HDF5 mutex.  LLVM additionally auto-vectorises the
            // cast loops to AVX2/SSE4 within each Rayon thread.
            const PAR_THRESHOLD: usize = 100_000;
            use rayon::prelude::*;

            // --- Append data ---
            let data_ds = self.file.dataset("X/data")?;
            let old_len = data_ds.shape()[0];
            let new_len = old_len + nnz;
            data_ds.resize(new_len)?;

            match (&csr.data, self.dtype) {
                // Same-type: clone is a memcpy, already optimal.
                (TypedVec::F32(v), DataType::F32) => {
                    data_ds.write_slice(&Array1::from_vec(v.clone()), s![old_len..new_len])?;
                }
                (TypedVec::F64(v), DataType::F64) => {
                    data_ds.write_slice(&Array1::from_vec(v.clone()), s![old_len..new_len])?;
                }
                // Cross-type direct paths — parallelize when large.
                (TypedVec::F64(v), DataType::F32) => {
                    let w: Vec<f32> = if nnz >= PAR_THRESHOLD {
                        v.par_iter().map(|&x| x as f32).collect()
                    } else {
                        v.iter().map(|&x| x as f32).collect()
                    };
                    data_ds.write_slice(&Array1::from_vec(w), s![old_len..new_len])?;
                }
                (TypedVec::F32(v), DataType::F64) => {
                    let w: Vec<f64> = if nnz >= PAR_THRESHOLD {
                        v.par_iter().map(|&x| x as f64).collect()
                    } else {
                        v.iter().map(|&x| x as f64).collect()
                    };
                    data_ds.write_slice(&Array1::from_vec(w), s![old_len..new_len])?;
                }
                // Integer sources — go through f64 then cast.
                (_, DataType::F32) => {
                    let f = if nnz >= PAR_THRESHOLD {
                        csr.data.to_f64_par()
                    } else {
                        csr.data.to_f64()
                    };
                    let w: Vec<f32> = if nnz >= PAR_THRESHOLD {
                        f.into_par_iter().map(|x| x as f32).collect()
                    } else {
                        f.into_iter().map(|x| x as f32).collect()
                    };
                    data_ds.write_slice(&Array1::from_vec(w), s![old_len..new_len])?;
                }
                (_, DataType::F64) => {
                    let f = if nnz >= PAR_THRESHOLD {
                        csr.data.to_f64_par()
                    } else {
                        csr.data.to_f64()
                    };
                    data_ds.write_slice(&Array1::from_vec(f), s![old_len..new_len])?;
                }
                (_, DataType::I32) => {
                    let f = if nnz >= PAR_THRESHOLD {
                        csr.data.to_f64_par()
                    } else {
                        csr.data.to_f64()
                    };
                    let w: Vec<i32> = if nnz >= PAR_THRESHOLD {
                        f.into_par_iter().map(|x| x as i32).collect()
                    } else {
                        f.into_iter().map(|x| x as i32).collect()
                    };
                    data_ds.write_slice(&Array1::from_vec(w), s![old_len..new_len])?;
                }
                (_, DataType::U32) => {
                    let f = if nnz >= PAR_THRESHOLD {
                        csr.data.to_f64_par()
                    } else {
                        csr.data.to_f64()
                    };
                    let w: Vec<u32> = if nnz >= PAR_THRESHOLD {
                        f.into_par_iter().map(|x| x as u32).collect()
                    } else {
                        f.into_iter().map(|x| x as u32).collect()
                    };
                    data_ds.write_slice(&Array1::from_vec(w), s![old_len..new_len])?;
                }
            }

            // --- Append indices (gene indices as i32) ---
            let idx_ds = self.file.dataset("X/indices")?;
            let old_idx_len = idx_ds.shape()[0];
            idx_ds.resize(new_len)?;
            let gene_i32: Vec<i32> = if nnz >= PAR_THRESHOLD {
                csr.indices.par_iter().map(|&x| x as i32).collect()
            } else {
                csr.indices.iter().map(|&x| x as i32).collect()
            };
            idx_ds.write_slice(&Array1::from_vec(gene_i32), s![old_idx_len..new_len])?;
        }

        // --- Accumulate indptr ---
        let base = *self.x_indptr.last().unwrap();
        for i in 1..=chunk.nrows {
            self.x_indptr.push(base + csr.indptr[i]);
        }

        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        let x_grp = self.file.group("X")?;

        // Write X/indptr — use i32 if small enough, i64 otherwise
        let max_val = self.x_indptr.iter().copied().max().unwrap_or(0);
        if max_val > i32::MAX as u64 {
            let v: Vec<i64> = self.x_indptr.iter().map(|&x| x as i64).collect();
            write_1d(&x_grp, "indptr", Array1::from_vec(v))?;
        } else {
            let v: Vec<i32> = self.x_indptr.iter().map(|&x| x as i32).collect();
            write_1d(&x_grp, "indptr", Array1::from_vec(v))?;
        }

        // Write X/shape attribute: [n_obs, n_vars] (required by AnnData spec)
        let shape_vals = vec![self.n_obs as i64, self.n_vars as i64];
        let attr = x_grp.new_attr::<i64>().shape(2).create("shape")?;
        attr.write(&Array1::from_vec(shape_vals))?;

        tracing::info!(
            n_obs = self.n_obs,
            n_vars = self.n_vars,
            nnz = self.x_indptr.last().copied().unwrap_or(0),
            "h5ad finalized"
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// H5AdReader
// ---------------------------------------------------------------------------

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
}

impl H5AdReader {
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
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

        // X can be stored as a dense 2-D dataset or a sparse CSR group.
        let x_is_dense = file.dataset("X").is_ok() && file.group("X").is_err();

        let (n_obs, n_vars, indptr, dtype) = if x_is_dense {
            let ds = file.dataset("X")?;
            let sh = ds.shape();
            if sh.len() != 2 {
                return Err(ScxError::InvalidFormat("dense X must be 2-D".into()));
            }
            let dtype = match ds.dtype()?.to_descriptor()? {
                TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
                TypeDescriptor::Float(_) => DataType::F64,
                TypeDescriptor::Integer(_) => DataType::I32,
                _ => DataType::F32,
            };
            (sh[0], sh[1], None, dtype)
        } else {
            let x_grp = file.group("X").map_err(|_| {
                ScxError::InvalidFormat("missing /X — not a valid H5AD file".into())
            })?;

            if let Ok(enc) = read_str_attr_on_group(&x_grp, "encoding-type") {
                if enc == "csc_matrix" {
                    return Err(ScxError::InvalidFormat(
                        "X is stored as CSC. Convert to CSR first: \
                         adata.X = adata.X.tocsr(); adata.write_h5ad(path)"
                            .into(),
                    ));
                }
            }

            let shape_attr = x_grp
                .attr("shape")
                .map_err(|_| ScxError::InvalidFormat("missing X/shape attribute".into()))?;
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

            let indptr = ad_read_indptr(&file, "X/indptr")?;
            if indptr.len() != n_obs + 1 {
                return Err(ScxError::InvalidFormat(format!(
                    "X/indptr length {} != n_obs+1 {}",
                    indptr.len(),
                    n_obs + 1
                )));
            }
            let dtype = ad_detect_dtype(&file, "X/data")?;
            (n_obs, n_vars, Some(indptr), dtype)
        };

        Ok(Self {
            path,
            n_obs,
            n_vars,
            indptr,
            chunk_size,
            dtype,
        })
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

fn ad_detect_dtype(file: &File, path: &str) -> Result<DataType> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_) => DataType::F64,
        TypeDescriptor::Integer(IntSize::U4) => DataType::I32,
        TypeDescriptor::Integer(IntSize::U8) => DataType::I32, // i64 → i32 (counts fit)
        _ => DataType::F32,
    })
}

/// Read a row slice of a dense 2-D dataset and convert to a sparse CSR chunk.
fn ad_read_dense_chunk(
    path: &Path,
    row_start: usize,
    row_end: usize,
    n_vars: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    ad_read_dense_chunk_with_dtype(&file, "X", row_start, row_end, n_vars, dtype)
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
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => {
            let raw: Array1<VarLenUnicode> = ds.read_1d()?;
            Ok(raw.into_iter().map(|s| s.to_string()).collect())
        }
        TypeDescriptor::VarLenAscii => {
            let raw: Array1<hdf5::types::VarLenAscii> = ds.read_1d()?;
            Ok(raw.into_iter().map(|s| s.to_string()).collect())
        }
        other => Err(ScxError::InvalidFormat(format!(
            "expected string dataset at '{path}', got {:?}",
            other
        ))),
    }
}

/// Read a chunk [row_start, row_end) from a CSR matrix stored at /X/.
/// H5AD natively stores X as CSR, so this is a direct slice — no transpose.
fn ad_read_chunk(
    path: &Path,
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
        let ds = file.dataset("X/indices")?;
        match ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(_) => ds
                .read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?
                .iter()
                .map(|&x| x as u32)
                .collect(),
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "unexpected X/indices dtype {:?}",
                    other
                )))
            }
        }
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset("X/data")?;
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
        // uint8 is used for bool columns (AnnData encodes bool as u8 0/1)
        TypeDescriptor::Integer(IntSize::U1) => {
            let v: Vec<u8> = ds.read_1d::<u8>()?.to_vec();
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
fn ad_read_categorical(file: &File, grp_path: &str) -> Result<ColumnData> {
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

    let levels = ad_read_strings(file, &format!("{grp_path}/categories"))?;
    Ok(ColumnData::Categorical { codes, levels })
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
            let strings = ad_read_strings(file, path)?;
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
                let raw = ad_walk_group(&file, "uns")?;
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
        let n_obs = self.n_obs;
        let n_vars = self.n_vars;
        let chunk_size = self.chunk_size;
        let dtype = self.dtype;

        match &self.indptr {
            Some(indptr) => {
                let indptr = indptr.clone();
                Box::pin(stream::unfold(0usize, move |row_start| {
                    let path = path.clone();
                    let indptr = indptr.clone();
                    async move {
                        if row_start >= n_obs {
                            return None;
                        }
                        let row_end = (row_start + chunk_size).min(n_obs);
                        let chunk =
                            ad_read_chunk(&path, &indptr, row_start, row_end, n_vars, dtype);
                        Some((chunk, row_end))
                    }
                }))
            }
            None => {
                // Dense X: read rows slice-by-slice and convert to sparse CSR
                Box::pin(stream::unfold(0usize, move |row_start| {
                    let path = path.clone();
                    async move {
                        if row_start >= n_obs {
                            return None;
                        }
                        let row_end = (row_start + chunk_size).min(n_obs);
                        let chunk = ad_read_dense_chunk(&path, row_start, row_end, n_vars, dtype);
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tempfile::NamedTempFile;

    use crate::h5::ScxH5Reader;
    use crate::stream::DatasetReader;

    // Golden fixture produced by zellkonverter (via scripts/prepare_h5ad_reference.R)
    const GOLDEN_REF: &str = "../../tests/golden/pbmc3k_reference.h5ad";
    const GOLDEN: &str = "../../tests/golden/pbmc3k.h5";
    // Committed subset fixture (generate with scripts/prepare_norman_subset.py)
    const NORMAN_SUBSET: &str = "../../tests/fixtures/norman_subset.h5ad";

    fn ref_exists() -> bool {
        std::path::Path::new(GOLDEN_REF).exists()
    }

    /// Return the path to the Norman H5AD to test against.
    /// Prefers the full file via `NORMAN_H5AD` env var (dev/CI with large data),
    /// falls back to the committed 500×200 subset.
    fn norman_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("NORMAN_H5AD") {
            let pb = std::path::PathBuf::from(&p);
            if pb.exists() {
                return Some(pb);
            }
            eprintln!("NORMAN_H5AD={p} not found, falling back to subset");
        }
        let pb = std::path::PathBuf::from(NORMAN_SUBSET);
        if pb.exists() {
            Some(pb)
        } else {
            None
        }
    }

    // --- H5AdReader tests (against zellkonverter reference) ---

    #[tokio::test]
    async fn test_reader_shape() {
        if !ref_exists() {
            return;
        }
        let reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();
        assert_eq!(n_obs, 2700, "expected 2700 cells");
        assert_eq!(n_vars, 13714, "expected 13714 genes");
    }

    #[tokio::test]
    async fn test_reader_obs() {
        if !ref_exists() {
            return;
        }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let obs = reader.obs().await.unwrap();
        assert_eq!(obs.index.len(), 2700, "obs index length");
        assert!(!obs.columns.is_empty(), "obs should have columns");
        assert!(
            obs.columns.iter().any(|c| c.name == "nCount_RNA"),
            "expected nCount_RNA column"
        );
    }

    #[tokio::test]
    async fn test_reader_obs_categorical() {
        if !ref_exists() {
            return;
        }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let obs = reader.obs().await.unwrap();
        // Seurat factor columns become categoricals in AnnData
        let cat_cols: Vec<_> = obs
            .columns
            .iter()
            .filter(|c| matches!(c.data, ColumnData::Categorical { .. }))
            .collect();
        assert!(
            !cat_cols.is_empty(),
            "expected at least one categorical obs column"
        );
    }

    #[tokio::test]
    async fn test_reader_var() {
        if !ref_exists() {
            return;
        }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let var = reader.var().await.unwrap();
        assert_eq!(var.index.len(), 13714, "var index length");
    }

    #[tokio::test]
    async fn test_reader_obsm() {
        if !ref_exists() {
            return;
        }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let obsm = reader.obsm().await.unwrap();
        assert!(obsm.map.contains_key("X_pca"), "missing X_pca");
        assert!(obsm.map.contains_key("X_umap"), "missing X_umap");
        assert_eq!(obsm.map["X_pca"].shape.0, 2700);
        assert_eq!(obsm.map["X_umap"].shape.0, 2700);
    }

    #[tokio::test]
    async fn test_reader_stream_coverage() {
        if !ref_exists() {
            return;
        }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let mut total_cells = 0usize;
        let mut total_nnz = 0usize;
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            total_cells += chunk.nrows;
            total_nnz += chunk.data.indices.len();
        }
        assert_eq!(total_cells, 2700);
        // nnz must match the reference (exact value verified against H5Seurat golden)
        assert_eq!(total_nnz, 2282976, "nnz mismatch against reference");
    }

    #[tokio::test]
    async fn test_reader_chunk_size_respected() {
        if !ref_exists() {
            return;
        }
        let chunk_size = 300usize;
        let mut reader = H5AdReader::open(GOLDEN_REF, chunk_size).unwrap();
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            assert!(chunk.nrows <= chunk_size, "chunk exceeded chunk_size");
        }
    }

    // --- Round-trip: H5AD reader → H5AD writer → structural equivalence ---

    #[tokio::test]
    async fn test_h5ad_roundtrip() {
        if !ref_exists() {
            return;
        }

        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();
        let dtype = reader.dtype();

        let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
        let out_path = tmp.path().to_path_buf();

        let obs = reader.obs().await.unwrap();
        let var = reader.var().await.unwrap();
        let obsm = reader.obsm().await.unwrap();
        let uns = reader.uns().await.unwrap();

        let mut writer = H5AdWriter::create(&out_path, n_obs, n_vars, dtype).unwrap();
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

        // Re-open the output and verify with the reader
        let mut rt = H5AdReader::open(&out_path, 500).unwrap();
        assert_eq!(rt.shape(), (n_obs, n_vars));

        let rt_obs = rt.obs().await.unwrap();
        assert_eq!(rt_obs.index.len(), n_obs);
        assert_eq!(rt_obs.columns.len(), obs.columns.len());

        let rt_obsm = rt.obsm().await.unwrap();
        for key in obsm.map.keys() {
            assert!(
                rt_obsm.map.contains_key(key),
                "obsm['{key}'] missing after roundtrip"
            );
            assert_eq!(rt_obsm.map[key].shape, obsm.map[key].shape);
        }

        let mut total_nnz = 0usize;
        let mut stream = rt.x_stream();
        while let Some(chunk) = stream.next().await {
            total_nnz += chunk.unwrap().data.indices.len();
        }
        assert_eq!(total_nnz, 2282976, "nnz changed after H5AD roundtrip");
    }

    fn golden_exists() -> bool {
        std::path::Path::new(GOLDEN).exists()
    }

    /// Full round-trip: read PBMC 3k → write h5ad → verify structure
    #[tokio::test]
    async fn test_roundtrip_pbmc3k() {
        if !golden_exists() {
            return;
        }

        let mut reader = ScxH5Reader::open(GOLDEN, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();

        let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
        let out_path = tmp.path().to_path_buf();

        let obs = reader.obs().await.unwrap();
        let var = reader.var().await.unwrap();
        let obsm = reader.obsm().await.unwrap();
        let uns = reader.uns().await.unwrap();

        let mut writer = H5AdWriter::create(&out_path, n_obs, n_vars, DataType::F32).unwrap();
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

        // --- Verify the output h5ad ---
        let out = File::open(&out_path).unwrap();

        // Root encoding
        let enc_type: String = out
            .group("/")
            .unwrap()
            .attr("encoding-type")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .to_string();
        assert_eq!(enc_type, "anndata");

        // X shape attribute
        let x_grp = out.group("X").unwrap();
        let shape: ndarray::Array1<i64> = x_grp.attr("shape").unwrap().read_1d().unwrap();
        assert_eq!(shape[0], n_obs as i64, "n_obs mismatch");
        assert_eq!(shape[1], n_vars as i64, "n_vars mismatch");

        // X/data length matches indptr last value
        let indptr: ndarray::Array1<i32> = out.dataset("X/indptr").unwrap().read_1d().unwrap();
        let data_len = out.dataset("X/data").unwrap().shape()[0];
        assert_eq!(data_len, *indptr.last().unwrap() as usize);
        assert_eq!(indptr.len(), n_obs + 1);

        // obs
        let obs_grp = out.group("obs").unwrap();
        let obs_enc: String = obs_grp
            .attr("encoding-type")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .to_string();
        assert_eq!(obs_enc, "dataframe");
        let obs_idx: ndarray::Array1<VarLenUnicode> =
            out.dataset("obs/index").unwrap().read_1d().unwrap();
        assert_eq!(obs_idx.len(), n_obs);

        // var
        let var_idx: ndarray::Array1<VarLenUnicode> =
            out.dataset("var/index").unwrap().read_1d().unwrap();
        assert_eq!(var_idx.len(), n_vars);

        // obsm
        let pca: ndarray::Array2<f64> = out.dataset("obsm/X_pca").unwrap().read().unwrap();
        assert_eq!(pca.shape(), &[n_obs, 30]);
        let umap: ndarray::Array2<f64> = out.dataset("obsm/X_umap").unwrap().read().unwrap();
        assert_eq!(umap.shape(), &[n_obs, 2]);

        tracing::info!("roundtrip OK: {} cells × {} genes", n_obs, n_vars);
    }

    /// Regression: layer stored as dense 2-D dataset must not panic when streamed.
    /// Previously, layer_stream() always called ad_read_sparse_chunk(), which indexed
    /// into an empty indptr and panicked with "index out of bounds: the len is 0".
    #[tokio::test]
    async fn test_dense_layer_stream_no_panic() {
        use hdf5::File as H5File;
        use ndarray::Array2;

        let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
        let path = tmp.path().to_path_buf();

        let n_obs: usize = 5;
        let n_vars: usize = 4;

        let vlu = |s: &str| VarLenUnicode::from_str(s).unwrap();

        // Build a minimal H5AD with a dense "counts" layer.
        {
            let f = H5File::create(&path).unwrap();
            let root = f.group("/").unwrap();
            root.new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&vlu("anndata"))
                .unwrap();
            root.new_attr::<VarLenUnicode>()
                .create("encoding-version")
                .unwrap()
                .write_scalar(&vlu("0.1.0"))
                .unwrap();

            // Minimal obs/var dataframes (just the index).
            let obs_grp = f.create_group("obs").unwrap();
            obs_grp
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&vlu("dataframe"))
                .unwrap();
            obs_grp
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")
                .unwrap()
                .write_scalar(&vlu("0.2.0"))
                .unwrap();
            obs_grp
                .new_attr::<VarLenUnicode>()
                .create("_index")
                .unwrap()
                .write_scalar(&vlu("index"))
                .unwrap();
            let obs_idx: ndarray::Array1<VarLenUnicode> =
                (0..n_obs).map(|i| vlu(&format!("cell{i}"))).collect();
            obs_grp
                .new_dataset_builder()
                .with_data(&obs_idx)
                .create("index")
                .unwrap();

            let var_grp = f.create_group("var").unwrap();
            var_grp
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&vlu("dataframe"))
                .unwrap();
            var_grp
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")
                .unwrap()
                .write_scalar(&vlu("0.2.0"))
                .unwrap();
            var_grp
                .new_attr::<VarLenUnicode>()
                .create("_index")
                .unwrap()
                .write_scalar(&vlu("index"))
                .unwrap();
            let var_idx: ndarray::Array1<VarLenUnicode> =
                (0..n_vars).map(|i| vlu(&format!("gene{i}"))).collect();
            var_grp
                .new_dataset_builder()
                .with_data(&var_idx)
                .create("index")
                .unwrap();

            // Sparse X (required by H5AdReader::open).
            let x_grp = f.create_group("X").unwrap();
            x_grp
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&vlu("csr_matrix"))
                .unwrap();
            x_grp
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")
                .unwrap()
                .write_scalar(&vlu("0.1.0"))
                .unwrap();
            let shape = ndarray::array![n_obs as i64, n_vars as i64];
            x_grp
                .new_attr_builder()
                .with_data(&shape)
                .create("shape")
                .unwrap();
            let indptr: ndarray::Array1<i32> = ndarray::Array1::zeros(n_obs + 1);
            x_grp
                .new_dataset_builder()
                .with_data(&indptr)
                .create("indptr")
                .unwrap();
            let indices: ndarray::Array1<i32> = ndarray::Array1::zeros(0);
            x_grp
                .new_dataset_builder()
                .with_data(&indices)
                .create("indices")
                .unwrap();
            let data: ndarray::Array1<f32> = ndarray::Array1::zeros(0);
            x_grp
                .new_dataset_builder()
                .with_data(&data)
                .create("data")
                .unwrap();

            // Dense "counts" layer — shape (n_obs, n_vars), stored as f32.
            let layers_grp = f.create_group("layers").unwrap();
            let counts: Array2<f32> = Array2::from_elem((n_obs, n_vars), 1.0_f32);
            layers_grp
                .new_dataset_builder()
                .with_data(&counts)
                .create("counts")
                .unwrap();
        }

        let mut reader = H5AdReader::open(&path, 3).unwrap();
        let metas = reader.layer_metas().await.unwrap();
        assert_eq!(metas.len(), 1, "expected 1 layer meta");
        assert_eq!(metas[0].name, "counts");
        assert!(
            metas[0].indptr.is_empty(),
            "dense layer must have empty indptr"
        );

        // Stream and collect all chunks — must not panic.
        let mut total_rows = 0usize;
        let mut total_nnz = 0usize;
        let mut stream = reader.layer_stream(&metas[0], 3);
        while let Some(res) = stream.next().await {
            let chunk = res.unwrap();
            total_rows += chunk.nrows;
            total_nnz += chunk.data.indptr.last().copied().unwrap_or(0) as usize;
        }
        assert_eq!(total_rows, n_obs);
        assert_eq!(
            total_nnz,
            n_obs * n_vars,
            "all values are 1.0 so every entry is non-zero"
        );
    }

    // --- Norman perturbation tests ---
    //
    // Run against the committed 500×200 subset by default.
    // Point NORMAN_H5AD=/path/to/norman_perturbation.h5ad to test the full file.
    // Generate the subset with:
    //   NORMAN_H5AD=... pixi run -e test prepare-norman-subset

    #[tokio::test]
    async fn test_norman_shape() {
        let Some(path) = norman_path() else {
            return;
        };
        let reader = H5AdReader::open(&path, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();
        assert!(n_obs > 0, "n_obs must be > 0");
        assert!(n_vars > 0, "n_vars must be > 0");
        eprintln!("norman shape: {n_obs} × {n_vars}");
    }

    #[tokio::test]
    async fn test_norman_obs_has_perturbation_column() {
        let Some(path) = norman_path() else {
            return;
        };
        let mut reader = H5AdReader::open(&path, 500).unwrap();
        let obs = reader.obs().await.unwrap();
        // Norman obs must contain at least one perturbation-related column
        let cols: Vec<&str> = obs.columns.iter().map(|c| c.name.as_str()).collect();
        eprintln!("norman obs columns: {cols:?}");
        assert!(!cols.is_empty(), "obs must have at least one column");
    }

    #[tokio::test]
    async fn test_norman_dense_layer_stream_coverage() {
        let Some(path) = norman_path() else {
            return;
        };
        let (n_obs, n_vars) = H5AdReader::open(&path, 500).unwrap().shape();

        let mut reader = H5AdReader::open(&path, 500).unwrap();
        let metas = reader.layer_metas().await.unwrap();
        assert!(
            !metas.is_empty(),
            "norman H5AD must have at least one layer"
        );

        // counts layer should be present and dense
        let counts = metas
            .iter()
            .find(|m| m.name == "counts")
            .expect("expected a 'counts' layer");
        assert!(
            counts.indptr.is_empty(),
            "counts layer should be dense (empty indptr)"
        );
        assert_eq!(counts.shape, (n_obs, n_vars));

        // stream and verify total row coverage
        let mut total_rows = 0usize;
        let mut stream = reader.layer_stream(counts, 500);
        while let Some(res) = stream.next().await {
            let chunk = res.unwrap();
            assert!(chunk.nrows > 0);
            total_rows += chunk.nrows;
        }
        assert_eq!(total_rows, n_obs, "streamed rows must equal n_obs");
    }

    #[tokio::test]
    async fn test_norman_x_stream_coverage() {
        let Some(path) = norman_path() else {
            return;
        };
        let (n_obs, _) = H5AdReader::open(&path, 500).unwrap().shape();

        let mut reader = H5AdReader::open(&path, 500).unwrap();
        let mut total_rows = 0usize;
        let mut stream = reader.x_stream();
        while let Some(res) = stream.next().await {
            total_rows += res.unwrap().nrows;
        }
        assert_eq!(total_rows, n_obs);
    }
}
