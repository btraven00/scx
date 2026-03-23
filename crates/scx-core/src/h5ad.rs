use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::str::FromStr;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::{Dataset, File, Group, SimpleExtents};
use hdf5::types::{IntSize, TypeDescriptor, FloatSize, VarLenUnicode};
use ndarray::{s, Array1, Array2};

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{Column, ColumnData, DenseMatrix, Embeddings, Layers, MatrixChunk, Obsp, ObsTable,
         SparseMatrixCSR, UnsTable, VarTable, Varm, Varp},
    stream::{DatasetReader, DatasetWriter},
};

/// Number of elements per HDF5 chunk for resizable datasets.
const CHUNK_ELEMS: usize = 65_536;

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
    // In hdf5 0.8: resizability lives in SimpleExtents, not the builder.
    file.new_dataset::<T>()
        .chunk(CHUNK_ELEMS)
        .shape(SimpleExtents::resizable([0usize]))
        .create(path)?;
    Ok(())
}

fn write_vlen_str_dataset(grp: &Group, name: &str, strings: &[String]) -> Result<Dataset> {
    let vals: Vec<VarLenUnicode> = strings
        .iter()
        .map(|s| VarLenUnicode::from_str(s).unwrap_or_default())
        .collect();
    let ds = grp.new_dataset::<VarLenUnicode>().shape(vals.len()).create(name)?;
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
            let ds = grp.new_dataset::<f64>().shape(v.len()).create(name)?;
            ds.write(&Array1::from_vec(v.clone()))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::Int(v) => {
            let ds = grp.new_dataset::<i32>().shape(v.len()).create(name)?;
            ds.write(&Array1::from_vec(v.clone()))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::Bool(v) => {
            let vi: Vec<u8> = v.iter().map(|&b| b as u8).collect();
            let ds = grp.new_dataset::<u8>().shape(vi.len()).create(name)?;
            ds.write(&Array1::from_vec(vi))?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::String(v) => {
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
                let ds = cat_grp.new_dataset::<i8>().shape(c.len()).create("codes")?;
                ds.write(&Array1::from_vec(c))?;
                write_encoding_on_ds(&ds, "array", "0.2.0")?;
            } else {
                let c: Vec<i16> = codes.iter().map(|&x| x as i16).collect();
                let ds = cat_grp.new_dataset::<i16>().shape(c.len()).create("codes")?;
                ds.write(&Array1::from_vec(c))?;
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

        for (name, mat) in &obsm.map {
            let (nrows, ncols) = mat.shape;
            let arr = Array2::from_shape_vec((nrows, ncols), mat.data.clone())
                .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
            let ds = grp.new_dataset::<f64>().shape((nrows, ncols)).create(name.as_str())?;
            ds.write(&arr)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }

        Ok(())
    }

    async fn write_uns(&mut self, _uns: &UnsTable) -> Result<()> {
        let grp = self.file.create_group("uns")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        Ok(())
    }

    async fn write_layers(&mut self, layers: &Layers) -> Result<()> {
        let grp = self.file.create_group("layers")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        for (name, mat) in &layers.map {
            ad_write_csr_group(&self.file, &format!("layers/{name}"), mat)?;
        }
        Ok(())
    }

    async fn write_obsp(&mut self, obsp: &Obsp) -> Result<()> {
        let grp = self.file.create_group("obsp")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        for (name, mat) in &obsp.map {
            ad_write_csr_group(&self.file, &format!("obsp/{name}"), mat)?;
        }
        Ok(())
    }

    async fn write_varp(&mut self, varp: &Varp) -> Result<()> {
        let grp = self.file.create_group("varp")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        for (name, mat) in &varp.map {
            ad_write_csr_group(&self.file, &format!("varp/{name}"), mat)?;
        }
        Ok(())
    }

    async fn write_varm(&mut self, varm: &Varm) -> Result<()> {
        let grp = self.file.create_group("varm")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        for (name, mat) in &varm.map {
            let (nrows, ncols) = mat.shape;
            let arr = Array2::from_shape_vec((nrows, ncols), mat.data.clone())
                .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
            let ds = grp.new_dataset::<f64>().shape((nrows, ncols)).create(name.as_str())?;
            ds.write(&arr)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        Ok(())
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let csr = &chunk.data;
        let nnz = csr.indices.len();

        if nnz > 0 {
            // --- Append data ---
            let data_ds = self.file.dataset("X/data")?;
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

            // --- Append indices (gene indices as i32) ---
            let idx_ds = self.file.dataset("X/indices")?;
            let old_idx_len = idx_ds.shape()[0];
            idx_ds.resize(new_len)?;
            let gene_i32: Vec<i32> = csr.indices.iter().map(|&x| x as i32).collect();
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
        // Write X/indptr — use i32 if small enough, i64 otherwise
        let max_val = self.x_indptr.iter().copied().max().unwrap_or(0);
        if max_val > i32::MAX as u64 {
            let v: Vec<i64> = self.x_indptr.iter().map(|&x| x as i64).collect();
            let ds = self.file.new_dataset::<i64>().shape(v.len()).create("X/indptr")?;
            ds.write(&Array1::from_vec(v))?;
        } else {
            let v: Vec<i32> = self.x_indptr.iter().map(|&x| x as i32).collect();
            let ds = self.file.new_dataset::<i32>().shape(v.len()).create("X/indptr")?;
            ds.write(&Array1::from_vec(v))?;
        }

        // Write X/shape attribute: [n_obs, n_vars] (required by AnnData spec)
        let x_grp = self.file.group("X")?;
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
/// Only CSR-encoded X is supported. Files with CSC X (e.g. written by older
/// scanpy versions) must be converted first:
///   `adata.X = adata.X.tocsr(); adata.write_h5ad(path)`
pub struct H5AdReader {
    path: PathBuf,
    n_obs: usize,
    n_vars: usize,
    /// CSR row pointer array (n_obs + 1 entries). Small enough to hold in RAM.
    indptr: Vec<u64>,
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

        let x_grp = file.group("X").map_err(|_| {
            ScxError::InvalidFormat("missing /X group — not a valid H5AD file".into())
        })?;

        if let Ok(enc) = read_str_attr_on_group(&x_grp, "encoding-type") {
            if enc == "csc_matrix" {
                return Err(ScxError::InvalidFormat(
                    "X is stored as CSC. Convert to CSR first: \
                     adata.X = adata.X.tocsr(); adata.write_h5ad(path)".into(),
                ));
            }
        }

        // Shape attribute: [n_obs, n_vars], written as i64 by our writer.
        // Older AnnData files may use i32.
        let shape_attr = x_grp.attr("shape").map_err(|_| {
            ScxError::InvalidFormat("missing X/shape attribute".into())
        })?;
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
                "X/indptr length {} != n_obs+1 {}", indptr.len(), n_obs + 1
            )));
        }

        let dtype = ad_detect_dtype(&file, "X/data")?;

        Ok(Self { path, n_obs, n_vars, indptr, chunk_size, dtype })
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
        TypeDescriptor::Integer(_) => {
            ds.read_1d::<i32>()?.iter().map(|&x| x as u64).collect()
        }
        TypeDescriptor::Float(_) => {
            ds.read_1d::<f64>()?.iter().map(|&x| x as u64).collect()
        }
        other => return Err(ScxError::InvalidFormat(format!(
            "unexpected indptr dtype {:?} at {path}", other
        ))),
    })
}

fn ad_detect_dtype(file: &File, path: &str) -> Result<DataType> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_)             => DataType::F64,
        TypeDescriptor::Integer(IntSize::U4) => DataType::I32,
        TypeDescriptor::Integer(IntSize::U8) => DataType::I32, // i64 → i32 (counts fit)
        _                                    => DataType::F32,
    })
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
            "expected string dataset at '{path}', got {:?}", other
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
    let nnz_end   = indptr[row_end]   as usize;
    let nnz = nnz_end - nnz_start;

    let indices: Vec<u32> = if nnz > 0 {
        let ds = file.dataset("X/indices")?;
        match ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(_) => {
                ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?
                    .iter().map(|&x| x as u32).collect()
            }
            other => return Err(ScxError::InvalidFormat(format!(
                "unexpected X/indices dtype {:?}", other
            ))),
        }
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset("X/data")?;
        match dtype {
            DataType::F32 => TypedVec::F32(
                ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::F64 => TypedVec::F64(
                ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::I32 => TypedVec::I32(
                ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::U32 => TypedVec::U32(
                ds.read_slice_1d::<u32, _>(s![nnz_start..nnz_end])?.to_vec()),
        }
    } else {
        TypedVec::F32(Vec::new())
    };

    // Normalise indptr to start from 0 for this chunk
    let chunk_indptr: Vec<u64> = indptr[row_start..=row_end]
        .iter().map(|&p| p - indptr[row_start]).collect();

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
        let is_group = file.group(&col_path).is_ok()
            && file.dataset(&col_path).is_err();

        let col_data = if is_group {
            match ad_read_categorical(file, &col_path) {
                Ok(cd) => cd,
                Err(e) => { tracing::warn!("skipping categorical '{col_name}': {e}"); continue; }
            }
        } else {
            match ad_read_column(file, &col_path) {
                Ok(cd) => cd,
                Err(e) => { tracing::warn!("skipping column '{col_name}': {e}"); continue; }
            }
        };
        columns.push(Column { name: col_name, data: col_data });
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
        TypeDescriptor::Float(_) => {
            Ok(ColumnData::Float(ds.read_1d::<f64>()?.to_vec()))
        }
        // uint8 is used for bool columns (AnnData encodes bool as u8 0/1)
        TypeDescriptor::Integer(IntSize::U1) => {
            let v: Vec<u8> = ds.read_1d::<u8>()?.to_vec();
            Ok(ColumnData::Bool(v.into_iter().map(|x| x != 0).collect()))
        }
        TypeDescriptor::Integer(_) => {
            Ok(ColumnData::Int(ds.read_1d::<i32>()?.to_vec()))
        }
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            Ok(ColumnData::String(ad_read_strings(file, path)?))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported column dtype {:?} at '{path}'", other
        ))),
    }
}

/// Read a categorical group: codes (i8 or i16) + categories (string-array).
fn ad_read_categorical(file: &File, grp_path: &str) -> Result<ColumnData> {
    let codes_path = format!("{grp_path}/codes");
    let codes_ds = file.dataset(&codes_path)?;
    let codes: Vec<u32> = match codes_ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(IntSize::U1) => {
            codes_ds.read_1d::<i8>()?.iter().map(|&x| x as u32).collect()
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            codes_ds.read_1d::<i16>()?.iter().map(|&x| x as u32).collect()
        }
        TypeDescriptor::Integer(_) => {
            codes_ds.read_1d::<i32>()?.iter().map(|&x| x as u32).collect()
        }
        other => return Err(ScxError::InvalidFormat(format!(
            "unexpected categorical codes dtype {:?}", other
        ))),
    };

    let levels = ad_read_strings(file, &format!("{grp_path}/categories"))?;
    Ok(ColumnData::Categorical { codes, levels })
}

/// Read the obsm group as named dense matrices.
fn ad_read_obsm(path: &Path, n_obs: usize) -> Result<Embeddings> {
    let file = File::open(path)?;
    let grp = match file.group("obsm") {
        Ok(g)  => g,
        Err(_) => return Ok(Embeddings::default()),
    };
    let mut map = HashMap::new();
    for name in grp.member_names().unwrap_or_default() {
        let ds_path = format!("obsm/{name}");
        let ds = match file.dataset(&ds_path) {
            Ok(d)  => d,
            Err(_) => continue,
        };
        let arr: Array2<f64> = match ds.read::<f64, ndarray::Ix2>() {
            Ok(a)  => a,
            Err(e) => { tracing::warn!("skipping obsm['{name}']: {e}"); continue; }
        };
        // Guard against transposed storage (some writers store (k, n_obs))
        let arr = if arr.shape()[0] != n_obs && arr.shape()[1] == n_obs {
            arr.t().to_owned()
        } else {
            arr
        };
        let shape = (arr.shape()[0], arr.shape()[1]);
        map.insert(name, DenseMatrix { shape, data: arr.into_raw_vec() });
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
        let is_group = file.group(&child_path).is_ok()
            && file.dataset(&child_path).is_err();
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
                    strings.into_iter().next().unwrap_or_default()
                ))
            } else {
                Ok(serde_json::json!(strings))
            }
        }
        _ => Ok(serde_json::Value::Null),
    }
}

/// Read a complete CSR sparse matrix from an H5AD-encoded group (e.g. "layers/X").
fn ad_read_full_csr(file: &File, group_path: &str) -> Result<SparseMatrixCSR> {
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

    let indices_ds = file.dataset(&format!("{group_path}/indices"))?;
    let indices: Vec<u32> = match indices_ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) => {
            indices_ds.read_1d::<i32>()?.iter().map(|&x| x as u32).collect()
        }
        other => return Err(ScxError::InvalidFormat(format!(
            "unexpected indices dtype {:?} at {group_path}/indices", other
        ))),
    };

    let data_ds = file.dataset(&format!("{group_path}/data"))?;
    let data = match data_ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => TypedVec::F32(data_ds.read_1d::<f32>()?.to_vec()),
        TypeDescriptor::Float(_)             => TypedVec::F64(data_ds.read_1d::<f64>()?.to_vec()),
        TypeDescriptor::Integer(_)           => TypedVec::I32(data_ds.read_1d::<i32>()?.to_vec()),
        _                                    => TypedVec::F32(data_ds.read_1d::<f32>()?.to_vec()),
    };

    Ok(SparseMatrixCSR { shape: (nrows, ncols), indptr, indices, data })
}

/// Write a complete CSR sparse matrix as an H5AD-encoded group (data stored as f32).
fn ad_write_csr_group(file: &File, group_path: &str, mat: &SparseMatrixCSR) -> Result<()> {
    let grp = file.create_group(group_path)?;
    write_encoding_on_group(&grp, "csr_matrix", "0.1.0")?;

    let shape_vals = vec![mat.shape.0 as i64, mat.shape.1 as i64];
    let attr = grp.new_attr::<i64>().shape(2).create("shape")?;
    attr.write(&Array1::from_vec(shape_vals))?;

    let data_f32: Vec<f32> = mat.data.to_f64().into_iter().map(|x| x as f32).collect();
    let ds = grp.new_dataset::<f32>().shape(data_f32.len()).create("data")?;
    ds.write(&Array1::from_vec(data_f32))?;
    write_encoding_on_ds(&ds, "array", "0.2.0")?;

    let indices_i32: Vec<i32> = mat.indices.iter().map(|&x| x as i32).collect();
    let ds = grp.new_dataset::<i32>().shape(indices_i32.len()).create("indices")?;
    ds.write(&Array1::from_vec(indices_i32))?;
    write_encoding_on_ds(&ds, "array", "0.2.0")?;

    let max_val = mat.indptr.iter().copied().max().unwrap_or(0);
    if max_val > i32::MAX as u64 {
        let v: Vec<i64> = mat.indptr.iter().map(|&x| x as i64).collect();
        let ds = grp.new_dataset::<i64>().shape(v.len()).create("indptr")?;
        ds.write(&Array1::from_vec(v))?;
    } else {
        let v: Vec<i32> = mat.indptr.iter().map(|&x| x as i32).collect();
        let ds = grp.new_dataset::<i32>().shape(v.len()).create("indptr")?;
        ds.write(&Array1::from_vec(v))?;
    }

    Ok(())
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
            Ok(_)  => {
                let raw = ad_walk_group(&file, "uns")?;
                Ok(UnsTable { raw })
            }
        }
    }

    async fn layers(&mut self) -> Result<Layers> {
        let file = File::open(&self.path)?;
        let grp = match file.group("layers") {
            Err(_) => return Ok(Layers::default()),
            Ok(g)  => g,
        };
        let mut map = HashMap::new();
        for name in grp.member_names().unwrap_or_default() {
            match ad_read_full_csr(&file, &format!("layers/{name}")) {
                Ok(m)  => { map.insert(name, m); }
                Err(e) => tracing::warn!("skipping layers['{name}']: {e}"),
            }
        }
        Ok(Layers { map })
    }

    async fn obsp(&mut self) -> Result<Obsp> {
        let file = File::open(&self.path)?;
        let grp = match file.group("obsp") {
            Err(_) => return Ok(Obsp::default()),
            Ok(g)  => g,
        };
        let mut map = HashMap::new();
        for name in grp.member_names().unwrap_or_default() {
            match ad_read_full_csr(&file, &format!("obsp/{name}")) {
                Ok(m)  => { map.insert(name, m); }
                Err(e) => tracing::warn!("skipping obsp['{name}']: {e}"),
            }
        }
        Ok(Obsp { map })
    }

    async fn varp(&mut self) -> Result<Varp> {
        let file = File::open(&self.path)?;
        let grp = match file.group("varp") {
            Err(_) => return Ok(Varp::default()),
            Ok(g)  => g,
        };
        let mut map = HashMap::new();
        for name in grp.member_names().unwrap_or_default() {
            match ad_read_full_csr(&file, &format!("varp/{name}")) {
                Ok(m)  => { map.insert(name, m); }
                Err(e) => tracing::warn!("skipping varp['{name}']: {e}"),
            }
        }
        Ok(Varp { map })
    }

    async fn varm(&mut self) -> Result<Varm> {
        let file = File::open(&self.path)?;
        let grp = match file.group("varm") {
            Err(_) => return Ok(Varm::default()),
            Ok(g)  => g,
        };
        let mut map = HashMap::new();
        for name in grp.member_names().unwrap_or_default() {
            let ds = match file.dataset(&format!("varm/{name}")) {
                Ok(d)  => d,
                Err(_) => continue,
            };
            match ds.read::<f64, ndarray::Ix2>() {
                Ok(arr) => {
                    let shape = (arr.shape()[0], arr.shape()[1]);
                    map.insert(name, DenseMatrix { shape, data: arr.into_raw_vec() });
                }
                Err(e) => tracing::warn!("skipping varm['{name}']: {e}"),
            }
        }
        Ok(Varm { map })
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let path       = self.path.clone();
        let indptr     = self.indptr.clone();
        let n_obs      = self.n_obs;
        let n_vars     = self.n_vars;
        let chunk_size = self.chunk_size;
        let dtype      = self.dtype;

        Box::pin(stream::unfold(0usize, move |row_start| {
            let path   = path.clone();
            let indptr = indptr.clone();
            async move {
                if row_start >= n_obs { return None; }
                let row_end = (row_start + chunk_size).min(n_obs);
                let chunk = ad_read_chunk(&path, &indptr, row_start, row_end, n_vars, dtype);
                Some((chunk, row_end))
            }
        }))
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

    fn ref_exists() -> bool { std::path::Path::new(GOLDEN_REF).exists() }

    // --- H5AdReader tests (against zellkonverter reference) ---

    #[tokio::test]
    async fn test_reader_shape() {
        if !ref_exists() { return; }
        let reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();
        assert_eq!(n_obs,  2700,  "expected 2700 cells");
        assert_eq!(n_vars, 13714, "expected 13714 genes");
    }

    #[tokio::test]
    async fn test_reader_obs() {
        if !ref_exists() { return; }
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
        if !ref_exists() { return; }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let obs = reader.obs().await.unwrap();
        // Seurat factor columns become categoricals in AnnData
        let cat_cols: Vec<_> = obs.columns.iter()
            .filter(|c| matches!(c.data, ColumnData::Categorical { .. }))
            .collect();
        assert!(!cat_cols.is_empty(), "expected at least one categorical obs column");
    }

    #[tokio::test]
    async fn test_reader_var() {
        if !ref_exists() { return; }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let var = reader.var().await.unwrap();
        assert_eq!(var.index.len(), 13714, "var index length");
    }

    #[tokio::test]
    async fn test_reader_obsm() {
        if !ref_exists() { return; }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let obsm = reader.obsm().await.unwrap();
        assert!(obsm.map.contains_key("X_pca"),  "missing X_pca");
        assert!(obsm.map.contains_key("X_umap"), "missing X_umap");
        assert_eq!(obsm.map["X_pca"].shape.0,  2700);
        assert_eq!(obsm.map["X_umap"].shape.0, 2700);
    }

    #[tokio::test]
    async fn test_reader_stream_coverage() {
        if !ref_exists() { return; }
        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let mut total_cells = 0usize;
        let mut total_nnz   = 0usize;
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            total_cells += chunk.nrows;
            total_nnz   += chunk.data.indices.len();
        }
        assert_eq!(total_cells, 2700);
        // nnz must match the reference (exact value verified against H5Seurat golden)
        assert_eq!(total_nnz, 2282976, "nnz mismatch against reference");
    }

    #[tokio::test]
    async fn test_reader_chunk_size_respected() {
        if !ref_exists() { return; }
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
        if !ref_exists() { return; }

        let mut reader = H5AdReader::open(GOLDEN_REF, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();
        let dtype = reader.dtype();

        let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
        let out_path = tmp.path().to_path_buf();

        let obs  = reader.obs().await.unwrap();
        let var  = reader.var().await.unwrap();
        let obsm = reader.obsm().await.unwrap();
        let uns  = reader.uns().await.unwrap();

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
            assert!(rt_obsm.map.contains_key(key), "obsm['{key}'] missing after roundtrip");
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
        if !golden_exists() { return; }

        let mut reader = ScxH5Reader::open(GOLDEN, 500).unwrap();
        let (n_obs, n_vars) = reader.shape();

        let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();
        let out_path = tmp.path().to_path_buf();

        let obs  = reader.obs().await.unwrap();
        let var  = reader.var().await.unwrap();
        let obsm = reader.obsm().await.unwrap();
        let uns  = reader.uns().await.unwrap();

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
        let enc_type: String = out.group("/").unwrap().attr("encoding-type").unwrap()
            .read_scalar::<VarLenUnicode>().unwrap().to_string();
        assert_eq!(enc_type, "anndata");

        // X shape attribute
        let x_grp = out.group("X").unwrap();
        let shape: ndarray::Array1<i64> = x_grp.attr("shape").unwrap().read_1d().unwrap();
        assert_eq!(shape[0], n_obs as i64,  "n_obs mismatch");
        assert_eq!(shape[1], n_vars as i64, "n_vars mismatch");

        // X/data length matches indptr last value
        let indptr: ndarray::Array1<i32> = out.dataset("X/indptr").unwrap().read_1d().unwrap();
        let data_len = out.dataset("X/data").unwrap().shape()[0];
        assert_eq!(data_len, *indptr.last().unwrap() as usize);
        assert_eq!(indptr.len(), n_obs + 1);

        // obs
        let obs_grp = out.group("obs").unwrap();
        let obs_enc: String = obs_grp.attr("encoding-type").unwrap()
            .read_scalar::<VarLenUnicode>().unwrap().to_string();
        assert_eq!(obs_enc, "dataframe");
        let obs_idx: ndarray::Array1<VarLenUnicode> = out.dataset("obs/index").unwrap().read_1d().unwrap();
        assert_eq!(obs_idx.len(), n_obs);

        // var
        let var_idx: ndarray::Array1<VarLenUnicode> = out.dataset("var/index").unwrap().read_1d().unwrap();
        assert_eq!(var_idx.len(), n_vars);

        // obsm
        let pca: ndarray::Array2<f64> = out.dataset("obsm/X_pca").unwrap().read().unwrap();
        assert_eq!(pca.shape(), &[n_obs, 30]);
        let umap: ndarray::Array2<f64> = out.dataset("obsm/X_umap").unwrap().read().unwrap();
        assert_eq!(umap.shape(), &[n_obs, 2]);

        tracing::info!("roundtrip OK: {} cells × {} genes", n_obs, n_vars);
    }
}
