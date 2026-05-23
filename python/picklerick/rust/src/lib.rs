#![allow(clippy::useless_conversion)]

use std::path::Path;

use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use scx_core::{
    bpcells::BpcellsDatasetReader,
    detect,
    detect::Format,
    dtype::{DataType, TypedVec},
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::H5SeuratWriter,
    ir::{ColumnData, MatrixChunk},
    stream::{DatasetReader, DatasetWriter},
};

fn py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    futures::executor::block_on(fut)
}

fn parse_dtype(dtype: &str) -> anyhow::Result<DataType> {
    match dtype {
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "i32" => Ok(DataType::I32),
        "u32" => Ok(DataType::U32),
        other => Err(anyhow::anyhow!(
            "unknown dtype '{other}': use f32, f64, i32, or u32"
        )),
    }
}

fn detect_format(path: &Path) -> Option<Format> {
    detect::sniff_dir(path)
        .or_else(|| detect::sniff(path))
        .or_else(|| match path.extension().and_then(|e| e.to_str()) {
            Some("h5seurat") => Some(Format::H5Seurat),
            Some("h5ad") => Some(Format::H5Ad),
            _ => Some(Format::ScxH5),
        })
}

fn open_reader(
    input_path: &Path,
    chunk_size: usize,
    assay: &str,
    layer: &str,
) -> anyhow::Result<Box<dyn DatasetReader>> {
    let fmt = detect_format(input_path);
    match fmt {
        Some(Format::H5Seurat) => {
            let reader = scx_core::h5seurat::open_h5seurat(
                input_path,
                chunk_size,
                Some(assay),
                Some(layer),
            )?;
            Ok(reader)
        }
        Some(Format::H5Ad) | None => {
            let reader = H5AdReader::open(input_path, chunk_size)?;
            Ok(Box::new(reader))
        }
        Some(Format::ScxH5) => {
            let reader = ScxH5Reader::open(input_path, chunk_size)?;
            Ok(Box::new(reader))
        }
        Some(Format::BPCells) => {
            let reader = BpcellsDatasetReader::open(input_path, chunk_size)?;
            Ok(Box::new(reader))
        }
        other => Err(anyhow::anyhow!("unsupported input format: {other:?}")),
    }
}

fn open_reader_metadata_only(input_path: &Path) -> anyhow::Result<Box<dyn DatasetReader>> {
    let fmt = detect_format(input_path);
    match fmt {
        Some(Format::H5Seurat) => {
            let reader = scx_core::h5seurat::open_h5seurat(input_path, 1, None, None)?;
            Ok(reader)
        }
        Some(Format::H5Ad) | None => Ok(Box::new(H5AdReader::open(input_path, 1)?)),
        Some(Format::ScxH5) => Ok(Box::new(ScxH5Reader::open(input_path, 1)?)),
        Some(Format::BPCells) => Ok(Box::new(BpcellsDatasetReader::open_metadata_only(
            input_path,
        )?)),
        other => Err(anyhow::anyhow!("unsupported input format: {other:?}")),
    }
}

async fn collect_inspect_info(
    reader: &mut dyn DatasetReader,
    format_name: &str,
    py: Python<'_>,
) -> anyhow::Result<PyObject> {
    let (n_obs, n_vars) = reader.shape();
    let obs = reader.obs().await?;
    let var = reader.var().await?;
    let obsm = reader.obsm().await?;
    let uns = reader.uns().await?;
    let layer_metas = reader.layer_metas().await?;
    let obsp_metas = reader.obsp_metas().await?;
    let varm = reader.varm().await?;

    let obs_cols: Vec<&str> = obs.columns.iter().map(|c| c.name.as_str()).collect();
    let var_cols: Vec<&str> = var.columns.iter().map(|c| c.name.as_str()).collect();
    let obsm_keys: Vec<&str> = obsm.map.keys().map(|s| s.as_str()).collect();
    let varm_keys: Vec<&str> = varm.map.keys().map(|s| s.as_str()).collect();
    let uns_keys: Vec<String> = uns
        .raw
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();

    let obs_dtypes: Vec<&str> = obs
        .columns
        .iter()
        .map(|c| match &c.data {
            ColumnData::Float(_) => "float64",
            ColumnData::Int(_) => "int32",
            ColumnData::Bool(_) => "bool",
            ColumnData::String(_) => "string",
            ColumnData::Categorical { .. } => "categorical",
        })
        .collect();
    let var_dtypes: Vec<&str> = var
        .columns
        .iter()
        .map(|c| match &c.data {
            ColumnData::Float(_) => "float64",
            ColumnData::Int(_) => "int32",
            ColumnData::Bool(_) => "bool",
            ColumnData::String(_) => "string",
            ColumnData::Categorical { .. } => "categorical",
        })
        .collect();

    // per-layer and per-obsp nnz stats (free from indptr, no matrix streaming)
    let layer_stats = pyo3::types::PyList::empty_bound(py);
    for m in &layer_metas {
        let nnz = m.indptr.last().copied().unwrap_or(0) as usize;
        let mut per_row: Vec<u64> = m.indptr.windows(2).map(|w| w[1] - w[0]).collect();
        per_row.sort_unstable();
        let n = per_row.len();
        let q = |p: f64| {
            if n == 0 {
                0u64
            } else {
                per_row[(p * (n - 1) as f64).round() as usize]
            }
        };
        let entry = pyo3::types::PyDict::new_bound(py);
        entry.set_item("name", m.name.as_str())?;
        entry.set_item("n_obs", m.shape.0 as i64)?;
        entry.set_item("n_vars", m.shape.1 as i64)?;
        entry.set_item("nnz", nnz as i64)?;
        entry.set_item("nnz_q1", q(0.25) as i64)?;
        entry.set_item("nnz_med", q(0.5) as i64)?;
        entry.set_item("nnz_q3", q(0.75) as i64)?;
        entry.set_item("nnz_max", per_row.last().copied().unwrap_or(0) as i64)?;
        layer_stats.append(entry)?;
    }

    let obsp_stats = pyo3::types::PyList::empty_bound(py);
    for m in &obsp_metas {
        let nnz = m.indptr.last().copied().unwrap_or(0) as usize;
        let mut per_row: Vec<u64> = m.indptr.windows(2).map(|w| w[1] - w[0]).collect();
        per_row.sort_unstable();
        let n = per_row.len();
        let q = |p: f64| {
            if n == 0 {
                0u64
            } else {
                per_row[(p * (n - 1) as f64).round() as usize]
            }
        };
        let entry = pyo3::types::PyDict::new_bound(py);
        entry.set_item("name", m.name.as_str())?;
        entry.set_item("n_obs", m.shape.0 as i64)?;
        entry.set_item("n_vars", m.shape.1 as i64)?;
        entry.set_item("nnz", nnz as i64)?;
        entry.set_item("nnz_q1", q(0.25) as i64)?;
        entry.set_item("nnz_med", q(0.5) as i64)?;
        entry.set_item("nnz_q3", q(0.75) as i64)?;
        entry.set_item("nnz_max", per_row.last().copied().unwrap_or(0) as i64)?;
        obsp_stats.append(entry)?;
    }

    // X nnz stats from indptr (empty slice for dense/BPCells)
    let x_indptr = reader.x_indptr();
    let x_nnz = x_indptr.last().copied().unwrap_or(0) as i64;
    let x_nnz_stats: Option<pyo3::Bound<'_, pyo3::types::PyDict>> = if x_indptr.len() > 1 {
        let mut per_row: Vec<u64> = x_indptr.windows(2).map(|w| w[1] - w[0]).collect();
        per_row.sort_unstable();
        let n = per_row.len();
        let q = |p: f64| per_row[(p * (n - 1) as f64).round() as usize] as i64;
        let s = pyo3::types::PyDict::new_bound(py);
        s.set_item("nnz", x_nnz)?;
        s.set_item("nnz_q1", q(0.25))?;
        s.set_item("nnz_med", q(0.5))?;
        s.set_item("nnz_q3", q(0.75))?;
        s.set_item("nnz_max", *per_row.last().unwrap() as i64)?;
        Some(s)
    } else {
        None
    };

    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("format", format_name)?;
    d.set_item("n_obs", n_obs as i64)?;
    d.set_item("n_vars", n_vars as i64)?;
    if let Some(s) = x_nnz_stats {
        d.set_item("x_stats", s)?;
    }
    d.set_item("obs_cols", obs_cols)?;
    d.set_item("obs_dtypes", obs_dtypes)?;
    d.set_item("var_cols", var_cols)?;
    d.set_item("var_dtypes", var_dtypes)?;
    d.set_item("obsm_keys", obsm_keys)?;
    d.set_item("layers", layer_stats)?;
    d.set_item("uns_keys", uns_keys)?;
    d.set_item("obsp", obsp_stats)?;
    d.set_item("varm_keys", varm_keys)?;
    Ok(d.unbind().into())
}

async fn write_aux_sparse_matrices(
    reader: &mut dyn DatasetReader,
    writer: &mut dyn DatasetWriter,
    chunk_size: usize,
) -> anyhow::Result<()> {
    let layer_metas = reader.layer_metas().await?;
    for meta in &layer_metas {
        writer.begin_sparse("layers", &meta.name, meta).await?;
        let mut stream = reader.layer_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    let obsp_metas = reader.obsp_metas().await?;
    for meta in &obsp_metas {
        writer.begin_sparse("obsp", &meta.name, meta).await?;
        let mut stream = reader.obsp_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    Ok(())
}

async fn do_convert(
    reader: &mut dyn DatasetReader,
    output: &Path,
    dtype: DataType,
    chunk_size: usize,
) -> anyhow::Result<()> {
    let (n_obs, n_vars) = reader.shape();
    let obs = reader.obs().await?;
    let var = reader.var().await?;
    let obsm = reader.obsm().await?;
    let uns = reader.uns().await?;
    let varm = reader.varm().await?;

    let mut writer = H5AdWriter::create(output, n_obs, n_vars, dtype)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;
    writer.write_varm(&varm).await?;

    write_aux_sparse_matrices(reader, &mut writer, chunk_size).await?;

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk?).await?;
    }
    writer.finalize().await?;
    Ok(())
}

async fn do_convert_h5seurat(
    reader: &mut dyn DatasetReader,
    output: &Path,
    dtype: DataType,
    assay: &str,
    chunk_size: usize,
) -> anyhow::Result<()> {
    let (n_obs, n_vars) = reader.shape();
    let obs = reader.obs().await?;
    let var = reader.var().await?;
    let obsm = reader.obsm().await?;
    let uns = reader.uns().await?;
    let varm = reader.varm().await?;

    let mut writer =
        H5SeuratWriter::create(output, n_obs, n_vars, dtype, Some(assay), None, None, false)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;
    writer.write_varm(&varm).await?;

    write_aux_sparse_matrices(reader, &mut writer, chunk_size).await?;

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk?).await?;
    }
    writer.finalize().await?;
    Ok(())
}

#[pyfunction]
fn scx_convert_native(
    input: &str,
    output: &str,
    chunk_size: usize,
    dtype: &str,
    assay: &str,
    layer: &str,
) -> PyResult<()> {
    let input_path = Path::new(input);
    let output_path = Path::new(output);

    let result = block_on(async {
        let out_dtype = parse_dtype(dtype)?;
        let mut reader = open_reader(input_path, chunk_size, assay, layer)?;
        do_convert(&mut *reader, output_path, out_dtype, chunk_size).await
    });

    result.map_err(py_err)
}

#[pyfunction]
fn scx_write_h5seurat_native(
    input: &str,
    output: &str,
    chunk_size: usize,
    assay: &str,
) -> PyResult<()> {
    let input_path = Path::new(input);
    let output_path = Path::new(output);

    let result = block_on(async {
        let mut reader = H5AdReader::open(input_path, chunk_size)?;
        do_convert_h5seurat(&mut reader, output_path, DataType::F32, assay, chunk_size).await
    });

    result.map_err(py_err)
}

#[pyfunction]
fn scx_inspect_native(py: Python<'_>, input: &str, _chunk_size: usize) -> PyResult<PyObject> {
    let input_path = Path::new(input);
    let fmt = detect_format(input_path);
    let format_name = match fmt {
        Some(Format::H5Seurat) => "H5Seurat",
        Some(Format::H5Ad) => "H5AD",
        Some(Format::ScxH5) => "ScxH5",
        Some(Format::BPCells) => "BPCells",
        _ => "unknown",
    };

    let result = block_on(async {
        let mut reader = open_reader_metadata_only(input_path)?;
        let fmt_name = if matches!(fmt, Some(Format::H5Seurat)) && reader.x_indptr().is_empty() {
            "H5Seurat (BPCells)"
        } else {
            format_name
        };
        collect_inspect_info(&mut *reader, fmt_name, py).await
    });

    result.map_err(py_err)
}

// ---------------------------------------------------------------------------
// Streaming iterator
// ---------------------------------------------------------------------------

/// Cast a typed slice to its raw byte representation.
///
/// Safety: `T` must be a plain-old-data type (no padding, no uninit bytes).
/// The caller must ensure the slice outlives the returned reference.
unsafe fn slice_as_bytes<T>(v: &[T]) -> &[u8] {
    std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
}

/// A single chunk of rows from a streaming matrix read.
///
/// Arrays are pre-computed as Python `bytes` objects; use
/// `numpy.frombuffer(chunk.indptr_bytes, dtype=numpy.uint64)` etc.
/// to obtain zero-copy numpy views.
#[pyclass]
pub struct PyMatrixChunk {
    #[pyo3(get)]
    pub row_offset: usize,
    #[pyo3(get)]
    pub nrows: usize,
    #[pyo3(get)]
    pub n_vars: usize,
    /// NumPy dtype string for the `data_bytes` array.
    #[pyo3(get)]
    pub dtype: &'static str,
    /// Raw bytes of a `(nrows+1,) uint64` CSR row-pointer array.
    #[pyo3(get)]
    pub indptr_bytes: PyObject,
    /// Raw bytes of a `(nnz,) uint32` column-index array.
    #[pyo3(get)]
    pub indices_bytes: PyObject,
    /// Raw bytes of a `(nnz,) <dtype>` values array.
    #[pyo3(get)]
    pub data_bytes: PyObject,
}

fn chunk_to_py(py: Python<'_>, chunk: MatrixChunk, n_vars: usize) -> PyResult<PyMatrixChunk> {
    let dtype = match &chunk.data.data {
        TypedVec::F32(_) => "float32",
        TypedVec::F64(_) => "float64",
        TypedVec::I32(_) => "int32",
        TypedVec::U32(_) => "uint32",
    };
    // Safety: Vec<u64/u32/f32/f64> are plain-old-data, no padding.
    let indptr_bytes: PyObject =
        pyo3::types::PyBytes::new_bound(py, unsafe { slice_as_bytes(&chunk.data.indptr) })
            .into_any()
            .unbind();
    let indices_bytes: PyObject =
        pyo3::types::PyBytes::new_bound(py, unsafe { slice_as_bytes(&chunk.data.indices) })
            .into_any()
            .unbind();
    let data_bytes: PyObject = match &chunk.data.data {
        TypedVec::F32(v) => pyo3::types::PyBytes::new_bound(py, unsafe { slice_as_bytes(v) }),
        TypedVec::F64(v) => pyo3::types::PyBytes::new_bound(py, unsafe { slice_as_bytes(v) }),
        TypedVec::I32(v) => pyo3::types::PyBytes::new_bound(py, unsafe { slice_as_bytes(v) }),
        TypedVec::U32(v) => pyo3::types::PyBytes::new_bound(py, unsafe { slice_as_bytes(v) }),
    }
    .into_any()
    .unbind();

    Ok(PyMatrixChunk {
        row_offset: chunk.row_offset,
        nrows: chunk.nrows,
        n_vars,
        dtype,
        indptr_bytes,
        indices_bytes,
        data_bytes,
    })
}

/// An iterator over `PyMatrixChunk` objects backed by a background reader thread.
///
/// Obtain via `scx_open_stream`; iterate in Python with a `for` loop.
#[pyclass]
pub struct PyMatrixStream {
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<std::result::Result<MatrixChunk, String>>>,
    n_vars: usize,
}

#[pymethods]
impl PyMatrixStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyMatrixChunk>> {
        // Release the GIL while waiting for the background reader thread.
        let result = py.allow_threads(|| self.rx.lock().unwrap().recv());
        match result {
            Ok(Ok(chunk)) => chunk_to_py(py, chunk, self.n_vars).map(Some),
            Ok(Err(msg)) => Err(PyRuntimeError::new_err(msg)),
            Err(_) => Ok(None), // channel closed — stream exhausted
        }
    }
}

/// Open a streaming iterator over the count matrix of a single-cell file.
///
/// Returns a `PyMatrixStream` whose `__next__` yields `PyMatrixChunk` objects.
/// Reading runs on a background thread; the GIL is released between chunks.
#[pyfunction]
fn scx_open_stream(
    path: &str,
    chunk_size: usize,
    assay: &str,
    layer: &str,
) -> PyResult<PyMatrixStream> {
    let input_path = std::path::PathBuf::from(path);
    let assay = assay.to_string();
    let layer = layer.to_string();

    let reader = open_reader(&input_path, chunk_size, &assay, &layer).map_err(py_err)?;
    let (_, n_vars) = reader.shape();

    let (tx, rx) = std::sync::mpsc::sync_channel::<std::result::Result<MatrixChunk, String>>(8);

    std::thread::spawn(move || {
        futures::executor::block_on(async move {
            let mut reader = reader;
            let mut stream = reader.x_stream();
            while let Some(chunk) = stream.next().await {
                let result = chunk.map_err(|e| e.to_string());
                if tx.send(result).is_err() {
                    break;
                }
            }
        });
    });

    Ok(PyMatrixStream {
        rx: std::sync::Mutex::new(rx),
        n_vars,
    })
}

#[pymodule]
fn picklerick_py_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scx_convert_native, m)?)?;
    m.add_function(wrap_pyfunction!(scx_write_h5seurat_native, m)?)?;
    m.add_function(wrap_pyfunction!(scx_inspect_native, m)?)?;
    m.add_function(wrap_pyfunction!(scx_open_stream, m)?)?;
    m.add_class::<PyMatrixChunk>()?;
    m.add_class::<PyMatrixStream>()?;
    Ok(())
}
