use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use hdf5::types::VarLenUnicode;
use hdf5::{File, Group, SimpleExtents};
use ndarray::{s, Array1, Array2};

use crate::{
    dtype::DataType,
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta, UnsTable,
        VarTable, Varm,
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

        let file = File::create(path.as_ref())?;

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
                file.create_group(grp)?;
            }
        }

        file.create_group("assays")?;
        let assay_grp = file.create_group(&format!("assays/{assay}"))?;
        if seuratdisk_compat {
            let key =
                VarLenUnicode::from_str(&format!("{}_", assay.to_lowercase())).unwrap_or_default();
            assay_grp
                .new_attr::<VarLenUnicode>()
                .create("key")?
                .write_scalar(&key)?;
        }
        file.create_group(&format!("assays/{assay}/{layer}"))?;

        // Resizable datasets for streaming x-chunk writes.
        let data_path = format!("assays/{assay}/{layer}/data");
        let indices_path = format!("assays/{assay}/{layer}/indices");
        match dtype {
            DataType::F32 => seurat_init_resizable::<f32>(&file, &data_path)?,
            DataType::F64 => seurat_init_resizable::<f64>(&file, &data_path)?,
            DataType::I32 => seurat_init_resizable::<i32>(&file, &data_path)?,
            DataType::U32 => seurat_init_resizable::<u32>(&file, &data_path)?,
        }
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
    let ds = grp
        .new_dataset::<VarLenUnicode>()
        .shape(vals.len())
        .create(name)?;
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
        let attr = grp
            .new_attr::<VarLenUnicode>()
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
            let ds = col_grp
                .new_dataset::<i32>()
                .shape(values.len())
                .create("values")?;
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
        // reductions is pre-created in create(); open or create defensively.
        let reds_grp = match self.file.group("reductions") {
            Ok(g) => g,
            Err(_) => self.file.create_group("reductions")?,
        };
        if obsm.map.is_empty() {
            return Ok(());
        }
        for (key, mat) in &obsm.map {
            let red_name = key.strip_prefix("X_").unwrap_or(key.as_str());
            let red_grp = reds_grp.create_group(red_name)?;
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
            let vals: Vec<f64> = csr.data.to_f64();
            data_ds.write_slice(&Array1::from_vec(vals), s![old_len..new_len])?;

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
            Ok(g) => g,
            Err(_) => self.file.create_group("reductions")?,
        };
        for (key, mat) in &varm.map {
            let red_name = key.strip_prefix("X_").unwrap_or(key.as_str());
            // reduction sub-group may already exist from write_obsm
            let red_grp = match reds_grp.group(red_name) {
                Ok(g) => g,
                Err(_) => {
                    let g = reds_grp.create_group(red_name)?;
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
        if max_ptr > i32::MAX as u64 {
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
