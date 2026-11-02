use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use hdf5::types::VarLenUnicode;
use hdf5::{File, Group, SimpleExtents};
use ndarray::{s, Array1, Array2};

use crate::h5_str::{write_str_array_attr, write_strings};
use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta,
        UnsTable, VarTable, Varm,
    },
    stream::DatasetWriter,
};

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
    /// Value dtype, taken from the first chunk; `data` is created then.
    dtype: Option<DataType>,
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
    /// Whether to emit the extra attributes `SeuratDisk::LoadH5Seurat` validates.
    seuratdisk_compat: bool,
    /// X was copied as stored (`copy_csr_from_h5ad`), indptr included.
    x_copied: bool,
}

impl H5SeuratWriter {
    /// Create a new H5Seurat file for writing.
    #[allow(clippy::too_many_arguments)]
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
        assay: Option<&str>,
        layer: Option<&str>,
        project: Option<&str>,
        seuratdisk_compat: bool,
    ) -> Result<Self> {
        let assay = assay.unwrap_or("RNA").to_string();
        let layer = layer.unwrap_or("counts").to_string();
        let project = project.unwrap_or("SeuratProject");

        let file = crate::h5::create_file(path.as_ref())?;

        if seuratdisk_compat {
            // Root-level attributes and empty groups required by SeuratDisk::LoadH5Seurat.
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
                crate::h5::create_group(&file, grp)?;
            }
        }

        crate::h5::create_group(&file, "assays")?;
        let assay_grp = crate::h5::create_group(&file, &format!("assays/{assay}"))?;
        if seuratdisk_compat {
            let key =
                VarLenUnicode::from_str(&format!("{}_", assay.to_lowercase())).unwrap_or_default();
            assay_grp
                .new_attr::<VarLenUnicode>()
                .create("key")?
                .write_scalar(&key)?;
        }
        crate::h5::create_group(&file, &format!("assays/{assay}/{layer}"))?;

        // Resizable datasets for streaming x-chunk writes.
        let data_path = format!("assays/{assay}/{layer}/data");
        let indices_path = format!("assays/{assay}/{layer}/indices");
        seurat_init_values(&file, &data_path, dtype)?;
        seurat_init_resizable::<i32>(&file, &indices_path)?;

        Ok(Self {
            file,
            assay,
            layer,
            n_obs,
            n_vars,
            dtype,
            x_indptr: vec![0u64],
            sparse_state: None,
            seuratdisk_compat,
            x_copied: false,
        })
    }

    /// Attach the attributes `SeuratDisk::LoadH5Seurat` reads off a reduction.
    ///
    /// Without them the load fails at validation ("Attribute does not exist"),
    /// so a file written with `--seuratdisk-compat` was still unloadable as soon
    /// as the source had any embeddings.
    fn attr_reduction(&self, grp: &Group, red_name: &str) -> Result<()> {
        if !self.seuratdisk_compat {
            return Ok(());
        }
        for (name, value) in [
            ("active.assay", self.assay.clone()),
            ("key", seurat_reduction_key(red_name)),
        ] {
            let v = VarLenUnicode::from_str(&value).unwrap_or_default();
            grp.new_attr::<VarLenUnicode>()
                .create(name)?
                .write_scalar(&v)?;
        }
        Ok(())
    }
}

/// Seurat's column-name prefix for a reduction ("PC_1", "UMAP_2", ...).
///
/// Seurat special-cases the classic reductions rather than deriving the prefix
/// from the name, so matching it means hardcoding the same handful.
fn seurat_reduction_key(red_name: &str) -> String {
    match red_name.to_lowercase().as_str() {
        "pca" => "PC_".to_string(),
        "ica" => "IC_".to_string(),
        "tsne" => "tSNE_".to_string(),
        other => format!("{}_", other.to_uppercase()),
    }
}

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

/// `indptr` of the h5ad CSR group `src` when its arrays can be copied into a
/// dgCMatrix slot as stored (`copy_csr_from_h5ad`): int32 indices, chunks R
/// can decode (unfiltered, as anndata writes by default, or deflate,
/// optionally after shuffle), `want` as the value dtype when given, no
/// userblock, and a well-formed indptr. `None` otherwise. Whether each row is
/// sorted is checked during the copy, by [`RowCheck`].
fn csr_copy_indptr(src: &Group, n_rows: usize, want: Option<DataType>) -> Result<Option<Vec<i64>>> {
    use crate::h5_chunk::chunk_plan;
    if crate::h5_str::read_str_attr(src, "encoding-type")
        .ok()
        .as_deref()
        != Some("csr_matrix")
        || src.file()?.fcpl()?.userblock() != 0
    {
        return Ok(None);
    }
    let (Ok(data), Ok(indices), Ok(indptr)) = (
        src.dataset("data"),
        src.dataset("indices"),
        src.dataset("indptr"),
    ) else {
        return Ok(None);
    };
    let dt = data.dtype()?;
    let dtype = if dt.is::<f32>() {
        DataType::F32
    } else if dt.is::<f64>() {
        DataType::F64
    } else if dt.is::<i32>() {
        DataType::I32
    } else if dt.is::<u32>() {
        DataType::U32
    } else {
        return Ok(None);
    };
    let copyable = |d: &hdf5::Dataset| {
        d.chunk().is_some_and(|c| c.len() == 1)
            && (d.filters().is_empty() || chunk_plan(d).is_some())
    };
    let ip = indptr.dtype()?;
    if want.is_some_and(|w| w != dtype)
        || !indices.dtype()?.is::<i32>()
        || !(ip.is::<i32>() || ip.is::<i64>())
        || ![&data, &indices, &indptr].into_iter().all(copyable)
    {
        return Ok(None);
    }
    let indptr: Vec<i64> = indptr.read_raw()?;
    let nnz = indices.size();
    let ok = indptr.len() == n_rows + 1
        && indptr[0] == 0
        && indptr[n_rows] as usize == nnz
        && data.size() == nnz
        && indptr.windows(2).all(|w| w[0] <= w[1]);
    Ok(ok.then_some(indptr))
}

/// Checks, a chunk of `indices` at a time, that every row's column indices
/// are strictly increasing and below `n_cols`. dgCMatrix requires it, and
/// the streaming path sorts rows, so a copy is only equivalent when it holds.
struct RowCheck<'a> {
    indptr: &'a [i64],
    n_cols: i64,
    row: usize,
    pos: i64,
    prev: i64,
}

impl<'a> RowCheck<'a> {
    fn new(indptr: &'a [i64], n_cols: usize) -> Self {
        Self {
            indptr,
            n_cols: n_cols as i64,
            row: 0,
            pos: 0,
            prev: -1,
        }
    }

    /// Feed the next indices, as little-endian int32 bytes.
    fn feed(&mut self, bytes: &[u8]) -> bool {
        for c in bytes.as_chunks::<4>().0 {
            let i = i32::from_le_bytes(*c) as i64;
            while self.pos == self.indptr[self.row + 1] {
                self.row += 1;
                self.prev = -1;
            }
            if i <= self.prev || i >= self.n_cols {
                return false;
            }
            self.prev = i;
            self.pos += 1;
        }
        true
    }
}

fn seurat_init_resizable<T: hdf5::H5Type>(file: &File, path: &str) -> Result<()> {
    file.new_dataset::<T>()
        .chunk(SEURAT_CHUNK_ELEMS)
        .shape(SimpleExtents::resizable([0usize]))
        .create(path)?;
    Ok(())
}

/// obsm/varm entries as `(reduction name, matrix)`: `X_` stripped, sorted
/// (HashMap order would change the file bytes). Two keys naming the same
/// reduction (`X_pca` and `pca`) are an error, not an HDF5 "already exists".
pub(crate) fn reductions<'a>(
    map: &'a std::collections::HashMap<String, DenseMatrix>,
    slot: &str,
) -> Result<Vec<(&'a str, &'a DenseMatrix)>> {
    let mut out: Vec<_> = map
        .iter()
        .map(|(k, m)| (k.strip_prefix("X_").unwrap_or(k), m))
        .collect();
    out.sort_by_key(|(name, _)| *name);
    if let Some(w) = out.windows(2).find(|w| w[0].0 == w[1].0) {
        let name = w[0].0;
        return Err(ScxError::InvalidFormat(format!(
            "{slot} keys `{name}` and `X_{name}` both map to H5Seurat reduction `{name}`; rename one"
        )));
    }
    Ok(out)
}

fn seurat_init_values(file: &File, path: &str, dtype: DataType) -> Result<()> {
    match dtype {
        DataType::F32 => seurat_init_resizable::<f32>(file, path),
        DataType::F64 => seurat_init_resizable::<f64>(file, path),
        DataType::I32 => seurat_init_resizable::<i32>(file, path),
        DataType::U32 => seurat_init_resizable::<u32>(file, path),
    }
}

/// Write `v` into `ds[at..at + v.len()]` in its own dtype.
fn write_typed_slice(ds: &hdf5::Dataset, v: &TypedVec, at: usize) -> Result<()> {
    let r = s![at..at + v.len()];
    match v {
        TypedVec::F32(x) => ds.write_slice(x.as_slice(), r)?,
        TypedVec::F64(x) => ds.write_slice(x.as_slice(), r)?,
        TypedVec::I32(x) => ds.write_slice(x.as_slice(), r)?,
        TypedVec::U32(x) => ds.write_slice(x.as_slice(), r)?,
    }
    Ok(())
}

/// Write a SeuratDisk data.frame group (`meta.data`, `meta.features`): its
/// columns plus the `colnames`, `logicals` and (for meta.data) `_class`
/// attributes.
pub(crate) fn seurat_write_meta_cols(grp: &Group, columns: &[Column]) -> Result<()> {
    let names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    write_str_array_attr(grp, "colnames", &names)?;
    let logicals: Vec<&str> = columns
        .iter()
        .filter(|c| matches!(c.data, ColumnData::Bool(_)))
        .map(|c| c.name.as_str())
        .collect();
    if !logicals.is_empty() {
        write_str_array_attr(grp, "logicals", &logicals)?;
    }
    if grp.name() == "/meta.data" {
        write_str_array_attr(grp, "_class", &["data.frame"])?;
    }
    for col in columns {
        seurat_write_col(grp, col)?;
    }
    Ok(())
}

/// R's NA for integers and factor codes.
const NA_INTEGER: i32 = i32::MIN;

fn seurat_write_col(grp: &Group, col: &Column) -> Result<()> {
    let name = col.name.as_str();
    let na = |i: usize| col.is_na(i);
    let write_i32 = |g: &Group, n: &str, v: Vec<i32>| -> Result<()> {
        g.new_dataset::<i32>()
            .shape(v.len())
            .create(n)?
            .write(&Array1::from_vec(v))?;
        Ok(())
    };
    let write_f64 = |v: Vec<f64>| -> Result<()> {
        // NA_real_ is a NaN.
        let v: Vec<f64> = v
            .into_iter()
            .enumerate()
            .map(|(i, x)| if na(i) { f64::NAN } else { x })
            .collect();
        grp.new_dataset::<f64>()
            .shape(v.len())
            .create(name)?
            .write(&Array1::from_vec(v))?;
        Ok(())
    };
    match &col.data {
        ColumnData::Float(v) => write_f64(v.clone())?,
        // R has no int64: integer when every value fits (NA_integer_ is
        // taken), else double, which is exact up to 2^53.
        ColumnData::Int64(v) => {
            let fits = v
                .iter()
                .enumerate()
                .all(|(i, &x)| na(i) || (x > NA_INTEGER as i64 && x <= i32::MAX as i64));
            if fits {
                write_i32(
                    grp,
                    name,
                    v.iter()
                        .enumerate()
                        .map(|(i, &x)| if na(i) { NA_INTEGER } else { x as i32 })
                        .collect(),
                )?;
            } else {
                write_f64(v.iter().map(|&x| x as f64).collect())?;
            }
        }
        ColumnData::Int(v) => {
            write_i32(
                grp,
                name,
                v.iter()
                    .enumerate()
                    .map(|(i, &x)| if na(i) { NA_INTEGER } else { x })
                    .collect(),
            )?;
        }
        ColumnData::Bool(v) => {
            // int32 0/1/2 (FALSE/TRUE/NA); the name goes in the `logicals` attr.
            write_i32(
                grp,
                name,
                v.iter()
                    .enumerate()
                    .map(|(i, &b)| if na(i) { 2 } else { b as i32 })
                    .collect(),
            )?;
        }
        // ponytail: a missing string is written as "" (SeuratDisk has no NA for
        // plain character columns); write it as a factor if NA must survive.
        ColumnData::String(v) => {
            // The value at a missing position is a placeholder, not data.
            let v: Vec<String> = v
                .iter()
                .enumerate()
                .map(|(i, x)| if na(i) { String::new() } else { x.clone() })
                .collect();
            write_strings(grp, name, &v)?;
        }
        // SeuratDisk saves any factor as values + levels; `ordered` has no
        // H5Seurat encoding and is dropped.
        ColumnData::Categorical { codes, levels, .. } => {
            let col_grp = crate::h5::create_group(grp, name)?;
            // 1-based codes, NA_integer_ for missing.
            let values = codes
                .iter()
                .enumerate()
                .map(|(i, &c)| if na(i) { NA_INTEGER } else { c as i32 + 1 })
                .collect();
            write_i32(&col_grp, "values", values)?;
            write_strings(&col_grp, "levels", levels)?;
        }
    }
    Ok(())
}

/// `uns` goes to Seurat's `misc` slot, the free-form list SeuratDisk saves
/// as a group.
pub(crate) fn seurat_write_uns(file: &File, uns: &UnsTable) -> Result<()> {
    let Some(map) = uns.raw.as_object() else {
        return Ok(());
    };
    let misc = match file.group("misc") {
        Ok(g) => g,
        Err(_) => crate::h5::create_group(file, "misc")?,
    };
    for (k, v) in map {
        if misc.link_exists(k) {
            misc.unlink(k)?;
        }
        crate::h5_json::write_json(&misc, k, v, crate::h5_json::Encoding::None)?;
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
        write_strings(&root, "cell.names", &obs.index)?;

        // /meta.data/ — always created even when obs.columns is empty
        let meta_grp = crate::h5::create_group(&self.file, "meta.data")?;
        seurat_write_meta_cols(&meta_grp, &obs.columns)?;

        Ok(())
    }

    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        // /assays/{assay}/features
        let assay_grp = self.file.group(&format!("assays/{}", self.assay))?;
        write_strings(&assay_grp, "features", &var.index)?;

        // /assays/{assay}/meta.features/ — only when var has columns
        if !var.columns.is_empty() {
            let mf_grp = crate::h5::create_group(&assay_grp, "meta.features")?;
            seurat_write_meta_cols(&mf_grp, &var.columns)?;
        }

        Ok(())
    }

    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()> {
        // reductions is pre-created in create(); open or create defensively.
        let reds_grp = match self.file.group("reductions") {
            Ok(g) => g,
            Err(_) => crate::h5::create_group(&self.file, "reductions")?,
        };
        if obsm.map.is_empty() {
            return Ok(());
        }
        for (red_name, mat) in crate::h5seurat::reductions(&obsm.map, "obsm")? {
            let red_grp = crate::h5::create_group(&reds_grp, red_name)?;
            self.attr_reduction(&red_grp, red_name)?;

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

    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()> {
        seurat_write_uns(&self.file, uns)
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
                    crate::h5::create_group(&self.file, &format!("assays/{}", self.assay))?;
                }
                format!("assays/{}/{}", self.assay, name)
            }
            "obsp" => {
                if self.file.group("graphs").is_err() {
                    crate::h5::create_group(&self.file, "graphs")?;
                }
                format!("graphs/{name}")
            }
            other => format!("{other}/{name}"),
        };

        crate::h5::create_group(&self.file, &group_path)?;

        // Pre-create indices so chunks can be appended; `data` waits for the
        // first chunk, so it keeps the source dtype.
        seurat_init_resizable::<i32>(&self.file, &format!("{group_path}/indices"))?;

        self.sparse_state = Some(SparseWriteState {
            group_path,
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

        let csr = &chunk.data;
        let nnz = csr.indices.len();
        let data_path = format!("{}/data", state.group_path);
        if state.dtype.is_none() {
            seurat_init_values(&self.file, &data_path, csr.data.dtype())?;
            state.dtype = Some(csr.data.dtype());
        }

        if nnz > 0 {
            let data_ds = self.file.dataset(&data_path)?;
            let old_len = data_ds.shape()[0];
            let new_len = old_len + nnz;
            data_ds.resize(new_len)?;
            write_typed_slice(&data_ds, &csr.data, old_len)?;

            let idx_ds = self
                .file
                .dataset(&format!("{}/indices", state.group_path))?;
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
        let state = self.sparse_state.take().ok_or_else(|| {
            ScxError::InvalidFormat("end_sparse called without begin_sparse".into())
        })?;

        let grp = self.file.group(&state.group_path)?;
        if state.dtype.is_none() {
            // No chunks at all: an empty matrix, so the dtype is moot.
            seurat_init_values(
                &self.file,
                &format!("{}/data", state.group_path),
                DataType::F32,
            )?;
        }

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

        // dims is R's [nrow, ncol] of the dgCMatrix. The IR matrix is
        // cells x genes CSR, which on disk is a genes x cells CSC, so the
        // order flips (as for X in write_x_chunk).
        let (nrows, ncols) = state.shape;
        let dims = vec![ncols as i32, nrows as i32];
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
            Ok(g) => g,
            Err(_) => crate::h5::create_group(&self.file, "reductions")?,
        };
        for (red_name, mat) in crate::h5seurat::reductions(&varm.map, "varm")? {
            // reduction sub-group may already exist from write_obsm
            let red_grp = match reds_grp.group(red_name) {
                Ok(g) => g,
                Err(_) => {
                    let g = crate::h5::create_group(&reds_grp, red_name)?;
                    self.attr_reduction(&g, red_name)?;
                    g
                }
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
            let data_path = format!("assays/{}/{}/data", self.assay, self.layer);
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
                    data_ds
                        .write_slice(&Array1::from_vec(csr.data.to_f64()), s![old_len..new_len])?;
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
        let max_ptr = self.x_indptr.iter().copied().max().unwrap_or(0);
        let indptr_path = format!("{layer_path}/indptr");
        if self.x_copied {
            // copy_csr_from_h5ad brought the source's indptr along.
        } else if max_ptr > i32::MAX as u64 {
            let v: Vec<i64> = self.x_indptr.iter().map(|&x| x as i64).collect();
            let ds = self
                .file
                .new_dataset::<i64>()
                .shape(v.len())
                .create(indptr_path.as_str())?;
            ds.write(&Array1::from_vec(v))?;
        } else {
            let v: Vec<i32> = self.x_indptr.iter().map(|&x| x as i32).collect();
            let ds = self
                .file
                .new_dataset::<i32>()
                .shape(v.len())
                .create(indptr_path.as_str())?;
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
            nnz    = self.file.dataset(&format!("{layer_path}/indices"))?.size(),
            assay  = %self.assay,
            layer  = %self.layer,
            "h5seurat finalized"
        );

        Ok(())
    }

    fn copy_csr_from_h5ad(&mut self, src: &Path, group: &str, slot: &str) -> Result<bool> {
        use crate::h5_chunk::copy_chunks;
        let Ok(src) = File::open(src).and_then(|f| f.group(group)) else {
            return Ok(false);
        };
        // X's dtype is the one asked for; other slots keep the source's.
        let is_x = slot == self.layer;
        let want = is_x.then_some(self.dtype);
        let Some(indptr) = csr_copy_indptr(&src, self.n_obs, want)? else {
            return Ok(false);
        };
        let path = format!("assays/{}/{}", self.assay, slot);
        // X's group exists, holding `create`'s empty streaming datasets.
        let grp = if is_x {
            self.file.group(&path)?
        } else {
            crate::h5::create_group(&self.file, &path)?
        };
        // Copy under temporary names, renamed once all three are in.
        let tmp = |name: &str| format!("{name}.copy");
        // indices first, checked as they go; an unsorted row abandons the copy.
        let mut rows = RowCheck::new(&indptr, self.n_vars);
        if !copy_chunks(
            &src.dataset("indices")?,
            &grp,
            &tmp("indices"),
            Some(&mut |b| rows.feed(b)),
        )? {
            // ponytail: the partial copy stays in the file as dead space (HDF5
            // does not reclaim it); only input with unsorted rows pays it.
            if is_x {
                grp.unlink(&tmp("indices"))?;
            } else {
                self.file.unlink(&path)?;
            }
            return Ok(false);
        }
        // CSR of cells x genes is, array for array, the CSC of genes x cells
        // that dgCMatrix stores; only `dims` is written fresh.
        copy_chunks(&src.dataset("data")?, &grp, &tmp("data"), None)?;
        copy_chunks(&src.dataset("indptr")?, &grp, &tmp("indptr"), None)?;
        for name in ["data", "indices", "indptr"] {
            if grp.link_exists(name) {
                grp.unlink(name)?;
            }
            grp.relink(&tmp(name), name)?;
        }
        if is_x {
            self.x_copied = true;
        } else {
            let dims = [self.n_vars as i32, self.n_obs as i32];
            grp.new_attr::<i32>()
                .shape(2)
                .create("dims")?
                .write(&dims)?;
        }
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
