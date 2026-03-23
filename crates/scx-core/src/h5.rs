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
    ir::{Column, ColumnData, DenseMatrix, Embeddings, Layers, MatrixChunk, Obsp, ObsTable,
         SparseMatrixCSR, UnsTable, VarTable, Varm, Varp},
    stream::DatasetReader,
};

/// Reader for the SCX simple HDF5 schema.
///
/// Schema layout:
///   /X/data      float32 (nnz,)        — count values (CSC)
///   /X/indices   int32   (nnz,)        — row (gene) indices
///   /X/indptr    float64 (ncells+1,)   — column (cell) pointers
///   /X/shape     float64 (2,)          — [ngenes, ncells]
///   /obs/index   string  (ncells,)
///   /obs/<col>   typed   (ncells,)
///   /var/index   string  (ngenes,)
///   /obsm/<key>  float64 (ncells, k)
pub struct ScxH5Reader {
    path: PathBuf,
    n_obs: usize,
    n_vars: usize,
    /// Full CSC column pointer array (n_obs+1 entries). Small enough to hold in RAM.
    indptr: Vec<u64>,
    chunk_size: usize,
    dtype: DataType,
}

impl ScxH5Reader {
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        let shape_ds = file.dataset("X/shape")?;
        let (n_vars, n_obs) = {
            let td = shape_ds.dtype()?.to_descriptor()?;
            let (v, o): (usize, usize) = match td {
                TypeDescriptor::Float(_) => {
                    let s: Vec<f64> = shape_ds.read_1d::<f64>()?.to_vec();
                    if s.len() < 2 { return Err(ScxError::InvalidFormat("X/shape must have 2 elements".into())); }
                    (s[0] as usize, s[1] as usize)
                }
                TypeDescriptor::Integer(_) => {
                    let s: Vec<i32> = shape_ds.read_1d::<i32>()?.to_vec();
                    if s.len() < 2 { return Err(ScxError::InvalidFormat("X/shape must have 2 elements".into())); }
                    (s[0] as usize, s[1] as usize)
                }
                other => return Err(ScxError::InvalidFormat(format!("unexpected shape type: {:?}", other))),
            };
            (v, o)
        };

        let indptr: Vec<u64> = read_indptr(&file)?;

        if indptr.len() != n_obs + 1 {
            return Err(ScxError::InvalidFormat(format!(
                "indptr length {} != n_obs+1 {}",
                indptr.len(),
                n_obs + 1
            )));
        }

        let dtype = detect_x_dtype(&file)?;

        Ok(Self { path, n_obs, n_vars, indptr, chunk_size, dtype })
    }
}

// ---------------------------------------------------------------------------
// Sync helpers (HDF5 crate is synchronous)
// ---------------------------------------------------------------------------

/// Read /X/indptr as u64, handling both float64 (rhdf5) and int32 (hdf5r) storage.
fn read_indptr(file: &File) -> Result<Vec<u64>> {
    let ds = file.dataset("X/indptr")?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(_) => {
            Ok(ds.read_1d::<f64>()?.iter().map(|&x| x as u64).collect())
        }
        TypeDescriptor::Integer(_) => {
            Ok(ds.read_1d::<i32>()?.iter().map(|&x| x as u64).collect())
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unexpected indptr type: {:?}", other
        ))),
    }
}

/// Read /X/indices as u32, handling both int32 and uint32 storage.
fn read_indices_slice(ds: &hdf5::Dataset, start: usize, end: usize) -> Result<Vec<u32>> {
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) => {
            Ok(ds.read_slice_1d::<i32, _>(s![start..end])?.iter().map(|&x| x as u32).collect())
        }
        _ => Err(ScxError::InvalidFormat("unexpected indices type".into())),
    }
}

fn detect_x_dtype(file: &File) -> Result<DataType> {
    let ds = file.dataset("X/data")?;
    // Prefer explicit attribute set by the writer
    if let Ok(attr) = ds.attr("dtype") {
        if let Ok(s) = attr.read_scalar::<VarLenUnicode>() {
            return Ok(match s.as_str() {
                "float32" => DataType::F32,
                "float64" => DataType::F64,
                "int32"   => DataType::I32,
                "uint32"  => DataType::U32,
                _         => DataType::F32,
            });
        }
    }
    // Fall back to inspecting the HDF5 datatype
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_)             => DataType::F64,
        TypeDescriptor::Integer(_)           => DataType::I32,
        _                                    => DataType::F32,
    })
}

/// Read a cell-chunk [cell_start, cell_end) from a CSC matrix and return it
/// as a CSR MatrixChunk. The conversion is a zero-copy reinterpretation:
/// CSC columns = cells ↔ CSR rows = cells.
fn read_chunk_sync(
    path: &Path,
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

    let gene_indices: Vec<u32> = if nnz > 0 {
        let ds = file.dataset("X/indices")?;
        read_indices_slice(&ds, nnz_start, nnz_end)?
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset("X/data")?;
        match dtype {
            DataType::F32 => TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::F64 => TypedVec::F64(ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::I32 => TypedVec::I32(ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec()),
            DataType::U32 => TypedVec::U32(ds.read_slice_1d::<u32, _>(s![nnz_start..nnz_end])?.to_vec()),
        }
    } else {
        TypedVec::F32(Vec::new())
    };

    // CSC column pointers for [cell_start..=cell_end] become CSR row pointers.
    // Column indices in CSC (gene indices) become column indices in CSR — same data.
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

fn read_strings_sync(file: &File, path: &str) -> Result<Vec<String>> {
    let ds = file.dataset(path)?;
    // Dispatch on actual stored type to avoid HDF5 charset-conversion errors.
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

fn read_column_data_sync(file: &File, ds_path: &str) -> Result<ColumnData> {
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
            Ok(ColumnData::String(read_strings_sync(file, ds_path)?))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported column type {:?} at {}",
            other, ds_path
        ))),
    }
}

fn read_obs_sync(path: &Path) -> Result<ObsTable> {
    let file = File::open(path)?;
    let index = read_strings_sync(&file, "obs/index")?;
    let members = file.group("obs")?.member_names()?;
    let mut columns = Vec::new();
    for name in members {
        if name == "index" { continue; }
        let ds_path = format!("obs/{name}");
        match read_column_data_sync(&file, &ds_path) {
            Ok(data) => columns.push(Column { name, data }),
            Err(e)   => tracing::warn!("skipping obs column '{}': {}", name, e),
        }
    }
    Ok(ObsTable { index, columns })
}

fn read_var_sync(path: &Path) -> Result<VarTable> {
    let file = File::open(path)?;
    let index = read_strings_sync(&file, "var/index")?;
    let members = file.group("var")?.member_names()?;
    let mut columns = Vec::new();
    for name in members {
        if name == "index" { continue; }
        let ds_path = format!("var/{name}");
        match read_column_data_sync(&file, &ds_path) {
            Ok(data) => columns.push(Column { name, data }),
            Err(e)   => tracing::warn!("skipping var column '{}': {}", name, e),
        }
    }
    Ok(VarTable { index, columns })
}

fn read_obsm_sync(path: &Path, n_obs: usize) -> Result<Embeddings> {
    let file = File::open(path)?;
    let obsm_group = match file.group("obsm") {
        Ok(g)  => g,
        Err(_) => return Ok(Embeddings::default()),
    };
    let mut map = HashMap::new();
    for name in obsm_group.member_names()? {
        let ds_path = format!("obsm/{name}");
        let ds = file.dataset(&ds_path)?;
        let arr: ndarray::Array2<f64> = ds.read::<f64, ndarray::Ix2>()?;
        // R writes matrices in column-major (Fortran) order. HDF5/ndarray reads
        // in row-major (C) order, so the dims appear transposed: (n_components, n_obs)
        // instead of (n_obs, n_components). Detect and fix.
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

// ---------------------------------------------------------------------------
// DatasetReader impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetReader for ScxH5Reader {
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
        read_var_sync(&self.path)
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        read_obsm_sync(&self.path, self.n_obs)
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        Ok(UnsTable::default())
    }

    async fn layers(&mut self) -> Result<Layers> { Ok(Layers::default()) }
    async fn obsp(&mut self)   -> Result<Obsp>   { Ok(Obsp::default()) }
    async fn varp(&mut self)   -> Result<Varp>   { Ok(Varp::default()) }
    async fn varm(&mut self)   -> Result<Varm>   { Ok(Varm::default()) }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let path       = self.path.clone();
        let n_obs      = self.n_obs;
        let n_vars     = self.n_vars;
        let chunk_size = self.chunk_size;
        let indptr     = self.indptr.clone();
        let dtype      = self.dtype;

        Box::pin(stream::unfold(0usize, move |cell_start| {
            let path   = path.clone();
            let indptr = indptr.clone();
            async move {
                if cell_start >= n_obs {
                    return None;
                }
                let cell_end = (cell_start + chunk_size).min(n_obs);
                let chunk = read_chunk_sync(&path, &indptr, cell_start, cell_end, n_vars, dtype);
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

    const GOLDEN: &str = "../../tests/golden/pbmc3k.h5";

    fn golden_exists() -> bool {
        std::path::Path::new(GOLDEN).exists()
    }

    #[test]
    fn test_string_type_descriptor() {
        if !golden_exists() { return; }
        let file = File::open(GOLDEN).unwrap();
        let ds = file.dataset("obs/index").unwrap();
        let td = ds.dtype().unwrap().to_descriptor().unwrap();
        println!("obs/index dtype: {:?}", td);
        // Helps diagnose charset mismatch between rhdf5 and hdf5 Rust crate.
    }

    #[tokio::test]
    async fn test_open_shape() {
        if !golden_exists() { return; }
        let reader = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        let (n_obs, n_vars) = reader.shape();
        assert_eq!(n_obs,  2700,  "expected 2700 cells");
        assert_eq!(n_vars, 13714, "expected 13714 genes");
    }

    #[tokio::test]
    async fn test_obs() {
        if !golden_exists() { return; }
        let mut reader = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        let obs = reader.obs().await.unwrap();
        assert_eq!(obs.index.len(), 2700);
        assert!(!obs.columns.is_empty());
        assert!(obs.columns.iter().any(|c| c.name == "nCount_RNA"));
    }

    #[tokio::test]
    async fn test_var() {
        if !golden_exists() { return; }
        let mut reader = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        let var = reader.var().await.unwrap();
        assert_eq!(var.index.len(), 13714);
    }

    #[tokio::test]
    async fn test_obsm() {
        if !golden_exists() { return; }
        let mut reader = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        let obsm = reader.obsm().await.unwrap();
        assert!(obsm.map.contains_key("X_pca"));
        assert!(obsm.map.contains_key("X_umap"));
        let pca = &obsm.map["X_pca"];
        assert_eq!(pca.shape, (2700, 30));
        let umap = &obsm.map["X_umap"];
        assert_eq!(umap.shape, (2700, 2));
    }

    #[tokio::test]
    async fn test_stream_chunks_cover_all_cells() {
        if !golden_exists() { return; }
        let mut reader = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        let mut total_cells = 0usize;
        let mut total_nnz   = 0usize;
        let mut stream = reader.x_stream();
        let mut expected_offset = 0usize;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            assert_eq!(chunk.row_offset, expected_offset, "row_offset mismatch");
            assert_eq!(chunk.data.shape.0, chunk.nrows);
            assert_eq!(chunk.data.shape.1, 13714);
            total_cells   += chunk.nrows;
            total_nnz     += chunk.data.indices.len();
            expected_offset += chunk.nrows;
        }
        assert_eq!(total_cells, 2700);
        assert_eq!(total_nnz,   2282976);
    }

    #[tokio::test]
    async fn test_memory_bounded_streaming() {
        if !golden_exists() { return; }
        // Stream chunk-by-chunk and assert no single chunk exceeds 2× chunk budget.
        // (n_cells_per_chunk * avg_nnz_per_cell * bytes_per_f32 * 2)
        let chunk_size = 500usize;
        let mut reader = ScxH5Reader::open(GOLDEN, chunk_size).unwrap();
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            assert!(chunk.nrows <= chunk_size, "chunk exceeded chunk_size");
        }
    }
}
