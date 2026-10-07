use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use hdf5::types::VarLenUnicode;
use hdf5::{Dataset, File, Group, SimpleExtents};
use ndarray::{s, Array1, Array2};

use super::reader::read_shape;
use crate::h5_str::{read_str_array_attr, write_str_array_attr, write_str_attr, write_strings};
use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
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
    /// The stored value type: the first chunk's, so `data` is created then.
    dtype: Option<DataType>,
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
        write_str_attr(&root, "encoding-type", "anndata")?;
        write_str_attr(&root, "encoding-version", "0.1.0")?;

        // /X group — encoding attrs; resizable datasets created here
        let x_grp = file.create_group("X")?;
        write_str_attr(&x_grp, "encoding-type", "csr_matrix")?;
        write_str_attr(&x_grp, "encoding-version", "0.1.0")?;
        // shape attr written in finalize() once we know n_obs

        init_values(&x_grp, dtype, compression)?;
        // AnnData spec requires indices as i32
        init_resizable_1d::<i32>(&x_grp, "indices", compression)?;

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
        let (n_obs, n_vars) = read_shape(&x_grp)?;
        let dtype = crate::h5::value_dtype(&file.dataset("X/data")?)?;

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
    pub fn add_obs_column(&self, col: &Column) -> Result<()> {
        add_dataframe_column(&self.file, "obs", col, self.compression)
    }

    /// Add a single column to `/var`, updating the `column-order` attribute.
    pub fn add_var_column(&self, col: &Column) -> Result<()> {
        add_dataframe_column(&self.file, "var", col, self.compression)
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
    /// limitations as [`crate::h5_json::write_json`]). Creates `/uns` if absent and
    /// replaces any existing entry of the same name.
    pub fn add_uns_entry(&self, name: &str, value: &serde_json::Value) -> Result<()> {
        let uns_grp = dict_group(&self.file, "uns")?;
        if uns_grp.group(name).is_ok() || uns_grp.dataset(name).is_ok() {
            uns_grp.unlink(name)?;
        }
        crate::h5_json::write_json(&uns_grp, name, value, crate::h5_json::Encoding::AnnData)
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
        let uns_grp = dict_group(&self.file, "uns")?;
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

fn write_encoding_on_group(grp: &Group, enc_type: &str, enc_version: &str) -> Result<()> {
    write_str_attr(grp, "encoding-type", enc_type)?;
    write_str_attr(grp, "encoding-version", enc_version)
}

fn write_encoding_on_ds(ds: &Dataset, enc_type: &str, enc_version: &str) -> Result<()> {
    write_str_attr(ds, "encoding-type", enc_type)?;
    write_str_attr(ds, "encoding-version", enc_version)
}

// ---------------------------------------------------------------------------
// Dataset creation helpers
// ---------------------------------------------------------------------------

fn init_resizable_1d<T: hdf5::H5Type>(
    grp: &Group,
    name: &str,
    compression: Option<u8>,
) -> Result<Dataset> {
    // Resizable datasets are always chunked, so deflate applies directly.
    let mut builder = grp.new_dataset::<T>().chunk(CHUNK_ELEMS);
    if let Some(level) = compression {
        builder = builder.deflate(level);
    }
    Ok(builder
        .shape(SimpleExtents::resizable([0usize]))
        .create(name)?)
}

/// An empty, growable `data` dataset of `dtype` in the matrix group `grp`.
fn init_values(grp: &Group, dtype: DataType, compression: Option<u8>) -> Result<Dataset> {
    match dtype {
        DataType::F32 => init_resizable_1d::<f32>(grp, "data", compression),
        DataType::F64 => init_resizable_1d::<f64>(grp, "data", compression),
        DataType::I32 => init_resizable_1d::<i32>(grp, "data", compression),
        DataType::U32 => init_resizable_1d::<u32>(grp, "data", compression),
    }
}

/// The dict group `name` (uns, obsm, layers, ...), created if absent.
fn dict_group(file: &File, name: &str) -> Result<Group> {
    if let Ok(g) = file.group(name) {
        return Ok(g);
    }
    let g = file.create_group(name)?;
    write_encoding_on_group(&g, "dict", "0.1.0")?;
    Ok(g)
}

/// Chunks at least this large are converted in parallel. The conversion runs
/// before HDF5 is called, so it doesn't contend for HDF5's global lock.
const PAR_THRESHOLD: usize = 100_000;

/// `v` converted to `T` through `f`.
fn cast_slice<S: Copy + Into<f64> + Sync, T: Send>(
    v: &[S],
    f: impl Fn(f64) -> T + Sync + Send,
) -> Vec<T> {
    use rayon::prelude::*;
    if v.len() >= PAR_THRESHOLD {
        v.par_iter().map(|&x| f(x.into())).collect()
    } else {
        v.iter().map(|&x| f(x.into())).collect()
    }
}

/// `v` converted to `T` through `f`.
fn cast<T: Send>(v: &TypedVec, f: impl Fn(f64) -> T + Sync + Send) -> Vec<T> {
    match v {
        TypedVec::F32(v) => cast_slice(v, f),
        TypedVec::F64(v) => cast_slice(v, f),
        TypedVec::I32(v) => cast_slice(v, f),
        TypedVec::U32(v) => cast_slice(v, f),
    }
}

/// Append a chunk's values and column indices to the growable `data` (stored
/// as `dtype`) and `indices` datasets of the matrix group `grp`.
fn append_csr(grp: &Group, csr: &SparseMatrixCSR, dtype: DataType) -> Result<()> {
    let nnz = csr.indices.len();
    if nnz == 0 {
        return Ok(());
    }
    let data_ds = grp.dataset("data")?;
    let range = data_ds.shape()[0]..data_ds.shape()[0] + nnz;
    data_ds.resize(range.end)?;
    let dst = s![range.clone()];
    match (&csr.data, dtype) {
        // Same type: write straight from the chunk, no copy.
        (TypedVec::F32(v), DataType::F32) => data_ds.write_slice(v.as_slice(), dst)?,
        (TypedVec::F64(v), DataType::F64) => data_ds.write_slice(v.as_slice(), dst)?,
        (TypedVec::I32(v), DataType::I32) => data_ds.write_slice(v.as_slice(), dst)?,
        (TypedVec::U32(v), DataType::U32) => data_ds.write_slice(v.as_slice(), dst)?,
        (v, DataType::F32) => data_ds.write_slice(&cast(v, |x| x as f32), dst)?,
        (v, DataType::F64) => data_ds.write_slice(&cast(v, |x| x), dst)?,
        (v, DataType::I32) => data_ds.write_slice(&cast(v, |x| x as i32), dst)?,
        (v, DataType::U32) => data_ds.write_slice(&cast(v, |x| x as u32), dst)?,
    }
    let idx_ds = grp.dataset("indices")?;
    idx_ds.resize(range.end)?;
    let indices = cast_slice(&csr.indices, |x| x as i32);
    idx_ds.write_slice(&indices, s![range])?;
    Ok(())
}

/// Write `indptr` into `grp` as i32, or as i64 when it doesn't fit.
fn write_indptr(grp: &Group, indptr: &[u64], compression: Option<u8>) -> Result<Dataset> {
    if indptr.last().is_some_and(|&n| n > i32::MAX as u64) {
        let v: Vec<i64> = indptr.iter().map(|&x| x as i64).collect();
        write_1d(grp, "indptr", Array1::from_vec(v), compression)
    } else {
        let v: Vec<i32> = indptr.iter().map(|&x| x as i32).collect();
        write_1d(grp, "indptr", Array1::from_vec(v), compression)
    }
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
    write_str_attr(&grp, "_index", "_index")?;

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

    let col_names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    write_str_array_attr(&grp, "column-order", &col_names)?;

    // index dataset
    let idx_ds = write_strings(&grp, "_index", index)?;
    write_encoding_on_ds(&idx_ds, "string-array", "0.2.0")?;

    // columns
    for col in columns {
        write_column(&grp, col, compression)?;
    }

    Ok(())
}

/// A pandas nullable column (`nullable-integer` / `-boolean` /
/// `-string-array`): a group of `values` + `mask`, mask true = NA.
fn write_nullable<T: hdf5::H5Type>(
    grp: &Group,
    name: &str,
    encoding: &str,
    values: Array1<T>,
    mask: &[bool],
    compression: Option<u8>,
) -> Result<Group> {
    let g = grp.create_group(name)?;
    write_encoding_on_group(&g, encoding, "0.1.0")?;
    let ds = write_1d(&g, "values", values, compression)?;
    write_encoding_on_ds(&ds, "array", "0.2.0")?;
    write_nullable_mask(&g, mask, compression)?;
    Ok(g)
}

/// The `mask` of a nullable group: true = NA, the inverse of `Column::mask`.
fn write_nullable_mask(g: &Group, mask: &[bool], compression: Option<u8>) -> Result<()> {
    let missing: Vec<bool> = mask.iter().map(|&present| !present).collect();
    let ds = write_1d(g, "mask", Array1::from_vec(missing), compression)?;
    write_encoding_on_ds(&ds, "array", "0.2.0")
}

fn write_column(grp: &Group, col: &Column, compression: Option<u8>) -> Result<()> {
    let name = col.name.as_str();
    match (&col.data, &col.mask) {
        (ColumnData::Float(v), mask) => {
            // Float NA is NaN, so no mask group is needed.
            let vals: Vec<f64> = match mask {
                None => v.clone(),
                Some(m) => v
                    .iter()
                    .zip(m)
                    .map(|(&x, &p)| if p { x } else { f64::NAN })
                    .collect(),
            };
            let ds = write_1d(grp, name, Array1::from_vec(vals), compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        (ColumnData::Int(v), Some(m)) => {
            write_nullable(
                grp,
                name,
                "nullable-integer",
                Array1::from_vec(v.clone()),
                m,
                compression,
            )?;
        }
        (ColumnData::Bool(v), Some(m)) => {
            write_nullable(
                grp,
                name,
                "nullable-boolean",
                Array1::from_vec(v.clone()),
                m,
                compression,
            )?;
        }
        (ColumnData::String(v), Some(m)) => {
            let g = grp.create_group(name)?;
            write_encoding_on_group(&g, "nullable-string-array", "0.1.0")?;
            let ds = write_strings(&g, "values", v)?;
            write_encoding_on_ds(&ds, "string-array", "0.2.0")?;
            write_nullable_mask(&g, m, compression)?;
        }
        (ColumnData::Int(v), None) => {
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()), compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        (ColumnData::Bool(v), None) => {
            // Write as Rust `bool`, which hdf5 maps to the H5T_ENUM {FALSE=0,
            // TRUE=1} that h5py/AnnData use for booleans. Writing a plain u8
            // here instead produces an H5T_INTEGER that readers key off dtype
            // — rhdf5 hands it back as `raw` rather than logical, which breaks
            // consumers downstream of a round-trip.
            let ds = write_1d(grp, name, Array1::from_vec(v.clone()), compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        (ColumnData::String(v), None) => {
            // VarLen strings don't support HDF5 filters — written uncompressed.
            let ds = write_strings(grp, name, v)?;
            write_encoding_on_ds(&ds, "string-array", "0.2.0")?;
        }
        (ColumnData::Categorical { codes, levels }, _) => {
            let cat_grp = grp.create_group(name)?;
            write_encoding_on_group(&cat_grp, "categorical", "0.2.0")?;
            // ordered = false (stored as uint8 boolean)
            let attr = cat_grp.new_attr::<u8>().create("ordered")?;
            attr.write_scalar(&0u8)?;

            // Codes are written as the narrowest signed integer that can hold
            // every level index. Signed because AnnData reserves -1 for NA.
            // The i32 arm matters: `as i16` silently wraps, so a column with
            // more than 32767 levels (cell barcodes, cell names) used to be
            // written as garbage codes that no reader could detect.
            let codes: Vec<i32> = codes
                .iter()
                .enumerate()
                .map(|(i, &c)| if col.is_na(i) { -1 } else { c as i32 })
                .collect();
            let ds = if levels.len() <= i8::MAX as usize {
                let c: Vec<i8> = codes.iter().map(|&x| x as i8).collect();
                write_1d(&cat_grp, "codes", Array1::from_vec(c), compression)?
            } else if levels.len() <= i16::MAX as usize {
                let c: Vec<i16> = codes.iter().map(|&x| x as i16).collect();
                write_1d(&cat_grp, "codes", Array1::from_vec(c), compression)?
            } else {
                write_1d(&cat_grp, "codes", Array1::from_vec(codes), compression)?
            };
            write_encoding_on_ds(&ds, "array", "0.2.0")?;

            let cat_ds = write_strings(&cat_grp, "categories", levels)?;
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
        write_dense_dict(&self.file, "obsm", &obsm.map, self.compression)
    }

    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()> {
        let grp = self.file.create_group("uns")?;
        write_encoding_on_group(&grp, "dict", "0.1.0")?;
        if let Some(obj) = uns.raw.as_object() {
            for (key, val) in obj {
                crate::h5_json::write_json(&grp, key, val, crate::h5_json::Encoding::AnnData)?;
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
        let mat_grp = dict_group(&self.file, group_prefix)?.create_group(name)?;
        write_encoding_on_group(&mat_grp, "csr_matrix", "0.1.0")?;
        // `data` waits for the first chunk, which fixes its type.
        let ds = init_resizable_1d::<i32>(&mat_grp, "indices", self.compression)?;
        write_encoding_on_ds(&ds, "array", "0.2.0")?;

        self.sparse_state = Some(SparseWriteState {
            group_path: format!("{group_prefix}/{name}"),
            indptr: vec![0u64],
            shape: meta.shape,
            dtype: None,
        });
        Ok(())
    }

    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let state = self.sparse_state.as_mut().ok_or_else(|| {
            ScxError::InvalidFormat("write_sparse_chunk called without begin_sparse".into())
        })?;
        let grp = self.file.group(&state.group_path)?;
        let dtype = match state.dtype {
            Some(dtype) => dtype,
            None => {
                let dtype = chunk.data.data.dtype();
                let ds = init_values(&grp, dtype, self.compression)?;
                write_encoding_on_ds(&ds, "array", "0.2.0")?;
                *state.dtype.insert(dtype)
            }
        };
        append_csr(&grp, &chunk.data, dtype)?;
        let base = *state.indptr.last().unwrap();
        state
            .indptr
            .extend(chunk.data.indptr[1..=chunk.nrows].iter().map(|&p| base + p));
        Ok(())
    }

    async fn end_sparse(&mut self) -> Result<()> {
        let state = self.sparse_state.take().ok_or_else(|| {
            ScxError::InvalidFormat("end_sparse called without begin_sparse".into())
        })?;
        let grp = self.file.group(&state.group_path)?;
        if state.dtype.is_none() {
            // No chunks: an empty matrix still needs its data dataset.
            let ds = init_values(&grp, DataType::F32, self.compression)?;
            write_encoding_on_ds(&ds, "array", "0.2.0")?;
        }
        write_shape(&grp, state.shape)?;
        let ds = write_indptr(&grp, &state.indptr, self.compression)?;
        write_encoding_on_ds(&ds, "array", "0.2.0")
    }

    async fn write_varm(&mut self, varm: &Varm) -> Result<()> {
        write_dense_dict(&self.file, "varm", &varm.map, self.compression)
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        if self.mode == WriterMode::Append {
            return Err(ScxError::InvalidFormat(
                "write_x_chunk must not be called on an append-mode H5AdWriter".into(),
            ));
        }
        append_csr(&self.file.group("X")?, &chunk.data, self.dtype)?;
        let base = *self.x_indptr.last().unwrap();
        self.x_indptr
            .extend(chunk.data.indptr[1..=chunk.nrows].iter().map(|&p| base + p));
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        if self.mode == WriterMode::Append {
            return Err(ScxError::InvalidFormat(
                "finalize must not be called on an append-mode H5AdWriter".into(),
            ));
        }
        let x_grp = self.file.group("X")?;
        write_indptr(&x_grp, &self.x_indptr, self.compression)?;
        write_shape(&x_grp, (self.n_obs, self.n_vars))?;
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
    col: &Column,
    compression: Option<u8>,
) -> Result<()> {
    let col_name = col.name.as_str();
    let grp = file.group(group_name)?;

    // Tolerate a missing column-order (older files).
    let mut order = if grp.attr("column-order").is_ok() {
        read_str_array_attr(&grp, "column-order")?
    } else {
        Vec::new()
    };
    // Only update column-order if this is a new column (not an overwrite).
    if !order.iter().any(|c| c == col_name) {
        order.push(col_name.to_string());
        let order: Vec<&str> = order.iter().map(String::as_str).collect();
        write_str_array_attr(&grp, "column-order", &order)?;
    }

    write_column(&grp, col, compression)
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
    let grp = dict_group(file, group_name)?;
    let arr = Array2::from_shape_vec(mat.shape, mat.data.clone())
        .map_err(|e| ScxError::InvalidFormat(format!("{group_name}['{entry_name}']: {e}")))?;
    let ds = write_2d_f64(&grp, entry_name, &arr, compression)?;
    write_encoding_on_ds(&ds, "array", "0.2.0")
}

/// Write every entry of `map` into the dict group `group_name`, in name order.
fn write_dense_dict(
    file: &File,
    group_name: &str,
    map: &HashMap<String, DenseMatrix>,
    compression: Option<u8>,
) -> Result<()> {
    dict_group(file, group_name)?;
    let mut names: Vec<&String> = map.keys().collect();
    names.sort();
    for name in names {
        add_dense_dict_entry(file, group_name, name, &map[name], compression)?;
    }
    Ok(())
}

/// The AnnData `shape` attribute `[rows, cols]` of a sparse matrix group.
fn write_shape(grp: &Group, shape: (usize, usize)) -> Result<()> {
    grp.new_attr::<i64>()
        .shape(2)
        .create("shape")?
        .write(&Array1::from_vec(vec![shape.0 as i64, shape.1 as i64]))?;
    Ok(())
}
