use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::File;
use hdf5::types::{TypeDescriptor, FloatSize, VarLenUnicode};
use ndarray::s;

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, UnsTable, VarTable},
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
pub struct H5SeuratReader {
    path: PathBuf,
    assay: String,
    layer: String,
    n_obs: usize,
    n_vars: usize,
    /// Full CSC column pointer array (n_obs+1 entries).
    indptr: Vec<u64>,
    chunk_size: usize,
    dtype: DataType,
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

        // dims attribute on the sparse group: [ngenes, ncells]
        let dims_attr = dims_grp.attr("dims").map_err(|_| {
            ScxError::InvalidFormat(format!("missing 'dims' attribute on {dims_path}"))
        })?;
        let dims: Vec<i32> = dims_attr.read_1d::<i32>()?.to_vec();
        if dims.len() < 2 {
            return Err(ScxError::InvalidFormat("dims must have 2 elements".into()));
        }
        let n_vars = dims[0] as usize;
        let n_obs  = dims[1] as usize;

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

        Ok(Self { path, assay, layer, n_obs, n_vars, indptr, chunk_size, dtype })
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
            arr.t().to_owned()
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
// DatasetReader impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetReader for H5SeuratReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        self.dtype
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
        Ok(UnsTable::default())
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let path       = self.path.clone();
        let assay      = self.assay.clone();
        let layer      = self.layer.clone();
        let n_obs      = self.n_obs;
        let n_vars     = self.n_vars;
        let chunk_size = self.chunk_size;
        let indptr     = self.indptr.clone();
        let dtype      = self.dtype;

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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    const GOLDEN: &str = "../../tests/golden/pbmc3k.h5seurat";

    fn golden_exists() -> bool {
        std::path::Path::new(GOLDEN).exists()
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
}
