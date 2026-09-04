use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use hdf5::types::{IntSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Dataset, File, Group, SimpleExtents};
use ndarray::{s, Array1, Array2};

use super::reader::ad_detect_dtype;
use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta,
        UnsTable, VarTable, Varm,
    },
    stream::DatasetWriter,
};

/// Number of elements per HDF5 chunk for the streaming X arrays (resizable datasets require chunks).
const CHUNK_ELEMS: usize = 65_536;

/// Controls whether /X write paths are active.
/// Append-mode writers must not call write_x_chunk or finalize — those paths
/// assume a freshly-created file with an empty resizable /X dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterMode {
    Create,
    Append,
}

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
    /// gzip (deflate) level applied to numeric datasets. `None` = uncompressed.
    compression: Option<u8>,
    /// Accumulated CSR indptr across all written chunks (n_obs + 1 entries when done).
    x_indptr: Vec<u64>,
    /// State for the currently open streaming sparse matrix, if any.
    sparse_state: Option<SparseWriteState>,
    mode: WriterMode,
}

impl H5AdWriter {
    /// Create an uncompressed writer (gzip off).
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
    ) -> Result<Self> {
        Self::create_compressed(path, n_obs, n_vars, dtype, None)
    }

    /// Create a writer, optionally gzip-compressing numeric datasets.
    ///
    /// `compression` is a deflate level in `0..=9`; `None` writes uncompressed.
    /// Variable-length string datasets (index, string columns, categories) are
    /// always written uncompressed — HDF5 filters don't apply to the global heap
    /// where vlen data lives.
    pub fn create_compressed<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
        compression: Option<u8>,
    ) -> Result<Self> {
        if let Some(level) = compression {
            if level > 9 {
                return Err(ScxError::InvalidFormat(format!(
                    "gzip compression level must be 0..=9, got {level}"
                )));
            }
        }

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
            DataType::F32 => init_resizable_1d::<f32>(&file, "X/data", compression)?,
            DataType::F64 => init_resizable_1d::<f64>(&file, "X/data", compression)?,
            DataType::I32 => init_resizable_1d::<i32>(&file, "X/data", compression)?,
            DataType::U32 => init_resizable_1d::<u32>(&file, "X/data", compression)?,
        }
        // AnnData spec requires indices as i32
        init_resizable_1d::<i32>(&file, "X/indices", compression)?;

        Ok(Self {
            file,
            n_obs,
            n_vars,
            dtype,
            compression,
            x_indptr: vec![0u64],
            sparse_state: None,
            mode: WriterMode::Create,
        })
    }

    /// Open an existing h5ad file for append (R/W mode).
    ///
    /// Recovers n_obs/n_vars from /X shape and dtype from /X/data — same logic
    /// as H5AdReader::open. write_x_chunk and finalize must NOT be called on
    /// append-mode writers; they assume an empty resizable /X dataset.
    pub fn open_for_append<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open_rw(path.as_ref())?;

        let x_grp = file
            .group("X")
            .map_err(|_| ScxError::InvalidFormat("missing /X — not a valid H5AD file".into()))?;
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
        let dtype = ad_detect_dtype(&file, "X/data")?;

        Ok(Self {
            file,
            n_obs,
            n_vars,
            dtype,
            // Append-mode never creates the X arrays and we don't compress
            // merge-appended slots; keep their layout matching the base file.
            compression: None,
            x_indptr: Vec::new(),
            sparse_state: None,
            mode: WriterMode::Append,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Returns true if the HDF5 group at `path` exists in the file.
    pub fn group_exists(&self, path: &str) -> bool {
        self.file.group(path).is_ok()
    }

    /// Deletes the named child link from `parent_path`.
    ///
    /// Used by conflict=overwrite to remove a slot before rewriting it.
    pub fn unlink_child(&self, parent_path: &str, name: &str) -> Result<()> {
        let parent = self.file.group(parent_path)?;
        parent.unlink(name)?;
        Ok(())
    }

    /// Returns true if a dataset or group named `name` exists inside `parent_path`.
    pub fn child_exists(&self, parent_path: &str, name: &str) -> bool {
        if let Ok(grp) = self.file.group(parent_path) {
            grp.group(name).is_ok() || grp.dataset(name).is_ok()
        } else {
            false
        }
    }

    /// Add a single column to `/obs`, updating the `column-order` attribute.
    ///
    /// Safe to call on both Create and Append mode writers.
    pub fn add_obs_column(&self, name: &str, data: &ColumnData) -> Result<()> {
        add_dataframe_column(&self.file, "obs", name, data, self.compression)
    }

    /// Add a single column to `/var`, updating the `column-order` attribute.
    pub fn add_var_column(&self, name: &str, data: &ColumnData) -> Result<()> {
        add_dataframe_column(&self.file, "var", name, data, self.compression)
    }

    /// Add a single entry to `/obsm` (creates the group if absent).
    pub fn add_obsm_entry(&self, name: &str, mat: &DenseMatrix) -> Result<()> {
        add_dense_dict_entry(&self.file, "obsm", name, mat, self.compression)
    }

    /// Add a single entry to `/varm` (creates the group if absent).
    pub fn add_varm_entry(&self, name: &str, mat: &DenseMatrix) -> Result<()> {
        add_dense_dict_entry(&self.file, "varm", name, mat, self.compression)
    }

    /// Add (or replace) a single top-level `/uns` entry from a JSON value.
    ///
    /// Mirrors the conversion path's uns encoding: scalars and nested dicts are
    /// written as native AnnData entries; arrays and nulls are skipped (same
    /// limitation as [`write_json_value`]). Creates `/uns` if absent and
    /// replaces any existing entry of the same name.
    pub fn add_uns_entry(&self, name: &str, value: &serde_json::Value) -> Result<()> {
        let uns_grp = match self.file.group("uns") {
            Ok(g) => g,
            Err(_) => {
                let g = self.file.create_group("uns")?;
                write_encoding_on_group(&g, "dict", "0.1.0")?;
                g
            }
        };
        if uns_grp.group(name).is_ok() || uns_grp.dataset(name).is_ok() {
            uns_grp.unlink(name)?;
        }
        write_json_value(&uns_grp, name, value)
    }

    /// Write or replace `uns["scx_provenance"]` with `prov`.
    ///
    /// The value is serialised as a single JSON string scalar rather than a
    /// nested HDF5 group tree, because slot keys contain "/" which HDF5
    /// interprets as a path separator — causing silent data corruption when
    /// stored as nested groups. The reader un-parses the string back to JSON.
    ///
    /// Creates `/uns` if it does not exist. Idempotent: deletes any existing
    /// `scx_provenance` entry (string or group) before writing the new one.
    pub fn upsert_uns_provenance(&self, prov: &serde_json::Value) -> Result<()> {
        let uns_grp = match self.file.group("uns") {
            Ok(g) => g,
            Err(_) => {
                let g = self.file.create_group("uns")?;
                write_encoding_on_group(&g, "dict", "0.1.0")?;
                g
            }
        };
        // Remove any pre-existing entry (may be a group or a dataset).
        if uns_grp.group("scx_provenance").is_ok() || uns_grp.dataset("scx_provenance").is_ok() {
            uns_grp.unlink("scx_provenance")?;
        }
        let json_str = serde_json::to_string(prov)
            .map_err(|e| ScxError::InvalidFormat(format!("provenance serialize error: {e}")))?;
        let v = VarLenUnicode::from_str(&json_str)
            .map_err(|_| ScxError::InvalidFormat("provenance contains invalid UTF-8".into()))?;
        let ds = uns_grp
            .new_dataset::<VarLenUnicode>()
            .shape(())
            .create("scx_provenance")?;
        ds.write_scalar(&v)?;
        write_encoding_on_ds(&ds, "string", "0.2.0")?;
        Ok(())
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
                write_encoding_on_ds(&ds, "numeric-scalar", "0.2.0")?;
            } else if let Some(f) = n.as_f64() {
                let ds = grp.new_dataset::<f64>().shape(()).create(name)?;
                ds.write_scalar(&f)?;
                write_encoding_on_ds(&ds, "numeric-scalar", "0.2.0")?;
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

fn init_resizable_1d<T: hdf5::H5Type>(
    file: &File,
    path: &str,
    compression: Option<u8>,
) -> Result<()> {
    // Resizable datasets are always chunked, so deflate applies directly.
    let mut builder = file.new_dataset::<T>().chunk(CHUNK_ELEMS);
    if let Some(level) = compression {
        builder = builder.deflate(level);
    }
    builder
        .shape(SimpleExtents::resizable([0usize]))
        .create(path)?;
    Ok(())
}

fn write_1d<T: hdf5::H5Type>(
    grp: &Group,
    name: &str,
    data: Array1<T>,
    compression: Option<u8>,
) -> Result<Dataset> {
    let len = data.len();
    let mut builder = grp.new_dataset::<T>();
    // Compression requires chunked storage; an empty dataset can't be chunked
    // and wouldn't benefit anyway.
    if let Some(level) = compression {
        if len > 0 {
            builder = builder.chunk(len.min(CHUNK_ELEMS)).deflate(level);
        }
    }
    let ds = builder.shape(len).create(name)?;
    ds.write(&data)?;
    Ok(ds)
}

/// Write a dense 2-D `f64` matrix, optionally gzip-compressed (chunked by rows).
fn write_2d_f64(
    grp: &Group,
    name: &str,
    arr: &Array2<f64>,
    compression: Option<u8>,
) -> Result<Dataset> {
    let (nrows, ncols) = arr.dim();
    let mut builder = grp.new_dataset::<f64>();
    if let Some(level) = compression {
        if nrows > 0 && ncols > 0 {
            let rows_per_chunk = (CHUNK_ELEMS / ncols.max(1)).clamp(1, nrows);
            builder = builder.chunk((rows_per_chunk, ncols)).deflate(level);
        }
    }
    let ds = builder.shape((nrows, ncols)).create(name)?;
    ds.write(arr)?;
    Ok(ds)
}

pub(super) fn write_vlen_str_dataset(
    grp: &Group,
    name: &str,
    strings: &[String],
) -> Result<Dataset> {
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
    compression: Option<u8>,
) -> Result<()> {
    let grp = file.create_group(group_name)?;
    write_encoding_on_group(&grp, "dataframe", "0.2.0")?;
    // anndata's canonical name for an unnamed index. scx has no named-index
    // concept, so always emit "_index": anndata resolves the index via this
    // attr (so it round-trips), and hdf5r/rhdf5-based R readers that hardcode
    // the literal `obs/_index` / `var/_index` path (e.g. omnibenchmark modules)
    // only find it under this name. Writing "index" satisfies only the former.
    write_str_attr_on_group(&grp, "_index", "_index")?;

    // A source column literally named "_index" collides with the reserved index
    // dataset we just declared (anndata stores the frame index at
    // `<obs|var>/_index`). Writing it as a column would create `_index` twice —
    // HDF5 fails the second create with a cryptic "name already exists". It's
    // invariably a stale round-trip artifact (an AnnData index that became a
    // Seurat meta.data column), so drop it rather than emit an invalid file.
    let columns: Vec<&Column> = columns
        .iter()
        .filter(|c| {
            if c.name == "_index" {
                tracing::warn!(
                    group = group_name,
                    "dropping column '_index': collides with the reserved frame index"
                );
                false
            } else {
                true
            }
        })
        .collect();

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
    let idx_ds = write_vlen_str_dataset(&grp, "_index", index)?;
    write_encoding_on_ds(&idx_ds, "string-array", "0.2.0")?;

    // columns
    for col in columns {
        write_column(&grp, &col.name, &col.data, compression)?;
    }

    Ok(())
}

fn write_column(grp: &Group, name: &str, data: &ColumnData, compression: Option<u8>) -> Result<()> {
    match data {
        ColumnData::Float(v) => {
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()), compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::Int(v) => {
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()), compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        ColumnData::Bool(v) => {
            // Write as Rust `bool`, which hdf5 maps to the H5T_ENUM {FALSE=0,
            // TRUE=1} that h5py/AnnData use for booleans. Writing a plain u8
            // here instead produces an H5T_INTEGER that readers key off dtype
            // — rhdf5 hands it back as `raw` rather than logical, which breaks
            // consumers downstream of a round-trip.
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()), compression)?;
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
                let ds = write_1d(&cat_grp, "codes", Array1::from_vec(c), compression)?;
                write_encoding_on_ds(&ds, "array", "0.2.0")?;
            } else {
                let c: Vec<i16> = codes.iter().map(|&x| x as i16).collect();
                let ds = write_1d(&cat_grp, "codes", Array1::from_vec(c), compression)?;
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
        write_dataframe(
            &self.file,
            "obs",
            &obs.index,
            &obs.columns,
            self.compression,
        )
    }

    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        write_dataframe(
            &self.file,
            "var",
            &var.index,
            &var.columns,
            self.compression,
        )
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
            let ds = write_2d_f64(&grp, name.as_str(), &arr, self.compression)?;
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
        init_resizable_1d::<f32>(&self.file, &format!("{group_path}/data"), self.compression)?;
        let ds = self.file.dataset(&format!("{group_path}/data"))?;
        write_encoding_on_ds(&ds, "array", "0.2.0")?;

        init_resizable_1d::<i32>(
            &self.file,
            &format!("{group_path}/indices"),
            self.compression,
        )?;
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
            let ds = write_1d(&grp, "indptr", Array1::from_vec(v), self.compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        } else {
            let v: Vec<i32> = state.indptr.iter().map(|&x| x as i32).collect();
            let ds = write_1d(&grp, "indptr", Array1::from_vec(v), self.compression)?;
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
            let ds = write_2d_f64(&grp, name.as_str(), &arr, self.compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        Ok(())
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        if self.mode == WriterMode::Append {
            return Err(ScxError::InvalidFormat(
                "write_x_chunk must not be called on an append-mode H5AdWriter".into(),
            ));
        }
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
        if self.mode == WriterMode::Append {
            return Err(ScxError::InvalidFormat(
                "finalize must not be called on an append-mode H5AdWriter".into(),
            ));
        }
        let x_grp = self.file.group("X")?;

        // Write X/indptr — use i32 if small enough, i64 otherwise
        let max_val = self.x_indptr.iter().copied().max().unwrap_or(0);
        if max_val > i32::MAX as u64 {
            let v: Vec<i64> = self.x_indptr.iter().map(|&x| x as i64).collect();
            write_1d(&x_grp, "indptr", Array1::from_vec(v), self.compression)?;
        } else {
            let v: Vec<i32> = self.x_indptr.iter().map(|&x| x as i32).collect();
            write_1d(&x_grp, "indptr", Array1::from_vec(v), self.compression)?;
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
// Append-mode helper primitives
// ---------------------------------------------------------------------------

/// Append a single column to an existing dataframe group (`/obs` or `/var`),
/// keeping the `column-order` attribute in sync.
fn add_dataframe_column(
    file: &File,
    group_name: &str,
    col_name: &str,
    data: &ColumnData,
    compression: Option<u8>,
) -> Result<()> {
    let grp = file.group(group_name)?;

    // Read existing column-order (tolerate missing attr for older files).
    let (existing, attr_existed) = match grp.attr("column-order") {
        Ok(attr) => {
            let raw: ndarray::Array1<VarLenUnicode> = attr.read_1d().unwrap_or_default();
            let names: Vec<String> = raw.into_iter().map(|s| s.to_string()).collect();
            (names, true)
        }
        Err(_) => (Vec::new(), false),
    };

    // Only update column-order if this is a new column (not an overwrite).
    if !existing.contains(&col_name.to_string()) {
        let mut new_order = existing;
        new_order.push(col_name.to_string());
        if attr_existed {
            grp.delete_attr("column-order")?;
        }
        let vals: Vec<VarLenUnicode> = new_order
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap_or_default())
            .collect();
        let attr = grp
            .new_attr::<VarLenUnicode>()
            .shape(vals.len())
            .create("column-order")?;
        attr.write(&ndarray::Array1::from_vec(vals))?;
    }

    write_column(&grp, col_name, data, compression)
}

/// Add a dense 2-D matrix as a named entry inside `/obsm` or `/varm`.
/// Creates the parent dict group with encoding attrs if it does not exist.
fn add_dense_dict_entry(
    file: &File,
    group_name: &str,
    entry_name: &str,
    mat: &DenseMatrix,
    compression: Option<u8>,
) -> Result<()> {
    let grp = match file.group(group_name) {
        Ok(g) => g,
        Err(_) => {
            let g = file.create_group(group_name)?;
            write_encoding_on_group(&g, "dict", "0.1.0")?;
            g
        }
    };
    let (nrows, ncols) = mat.shape;
    let arr = ndarray::Array2::from_shape_vec((nrows, ncols), mat.data.clone())
        .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
    let ds = write_2d_f64(&grp, entry_name, &arr, compression)?;
    write_encoding_on_ds(&ds, "array", "0.2.0")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// H5AdReader
// ---------------------------------------------------------------------------
