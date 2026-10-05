use std::path::Path;

use futures::executor::block_on;
use futures::StreamExt;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scx_core::{
    detect::{self, Format},
    dtype::{DataType, TypedVec},
    h5ad::H5AdWriter,
    h5seurat::H5SeuratWriter,
    ir::{Column, MatrixChunk, SparseMatrixMeta},
    stream::{DatasetReader, DatasetWriter},
};

pyo3::create_exception!(picklerick, PickleRickError, pyo3::exceptions::PyException);

fn py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PickleRickError::new_err(e.to_string())
}

fn open_reader(path: &str, opts: scx_core::OpenOptions) -> anyhow::Result<Box<dyn DatasetReader>> {
    Ok(block_on(scx_core::open(path, &opts))?)
}

fn stream_opts(chunk_size: usize, assay: &str, layer: &str) -> scx_core::OpenOptions {
    scx_core::OpenOptions {
        assay: Some(assay.to_string()),
        layer: Some(layer.to_string()),
        ..scx_core::OpenOptions::new(chunk_size)
    }
}

/// Pick the writer from the output extension. H5Seurat output is the
/// SeuratDisk-compatible dgCMatrix layout, not the CLI's BPCells default.
///
/// Same X-slot rule as `scx convert --x-slot auto`: when the source has a
/// `counts` layer, X is taken to be normalised and goes to the `data` slot.
fn create_writer(
    output: &Path,
    n_obs: usize,
    n_vars: usize,
    dtype: DataType,
    assay: &str,
    x_slot: &str,
) -> anyhow::Result<Box<dyn DatasetWriter>> {
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    Ok(match ext.as_deref() {
        Some("h5ad") => Box::new(H5AdWriter::create(output, n_obs, n_vars, dtype)?),
        Some("h5seurat") => Box::new(H5SeuratWriter::create(
            output,
            n_obs,
            n_vars,
            dtype,
            Some(assay),
            Some(x_slot),
            None,
            false,
        )?),
        _ => anyhow::bail!(
            "unsupported output extension for '{}': use .h5ad or .h5seurat",
            output.display()
        ),
    })
}

async fn convert(
    reader: &mut dyn DatasetReader,
    writer: &mut dyn DatasetWriter,
    layer_metas: &[SparseMatrixMeta],
    skip_layer: Option<&str>,
    chunk_size: usize,
) -> anyhow::Result<()> {
    writer.write_obs(&reader.obs().await?).await?;
    writer.write_var(&reader.var().await?).await?;
    writer.write_obsm(&reader.obsm().await?).await?;
    writer.write_uns(&reader.uns().await?).await?;
    writer.write_varm(&reader.varm().await?).await?;

    for meta in layer_metas {
        if Some(meta.name.as_str()) == skip_layer {
            continue;
        }
        writer.begin_sparse("layers", &meta.name, meta).await?;
        let mut stream = reader.layer_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }
    for meta in &reader.obsp_metas().await? {
        writer.begin_sparse("obsp", &meta.name, meta).await?;
        let mut stream = reader.obsp_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk?).await?;
    }
    writer.finalize().await?;
    Ok(())
}

#[pyfunction]
fn scx_convert(
    input: &str,
    output: &str,
    chunk_size: usize,
    dtype: &str,
    assay: &str,
    layer: &str,
) -> PyResult<()> {
    let run = || -> anyhow::Result<()> {
        let dtype = dtype.parse::<DataType>()?;
        let mut reader = open_reader(input, stream_opts(chunk_size, assay, layer))?;
        let (n_obs, n_vars) = reader.shape();
        let layer_metas = block_on(reader.layer_metas())?;
        let x_slot = if layer_metas.iter().any(|m| m.name == "counts") {
            "data"
        } else {
            "counts"
        };
        let output = Path::new(output);
        let mut writer = create_writer(output, n_obs, n_vars, dtype, assay, x_slot)?;
        // H5Seurat stores X in an assay slot; a layer of the same name can't coexist.
        let is_seurat = output
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("h5seurat"));
        let skip = is_seurat.then_some(x_slot);
        block_on(convert(
            &mut *reader,
            &mut *writer,
            &layer_metas,
            skip,
            chunk_size,
        ))
    };
    run().map_err(py_err)
}

/// Per-row nnz summary of a CSR matrix, computed from its indptr alone.
fn nnz_stats<'py>(py: Python<'py>, indptr: &[u64]) -> PyResult<Bound<'py, PyDict>> {
    let mut per_row: Vec<u64> = indptr.windows(2).map(|w| w[1] - w[0]).collect();
    per_row.sort_unstable();
    let q = |p: f64| match per_row.len() {
        0 => 0,
        n => per_row[(p * (n - 1) as f64).round() as usize],
    };
    let d = PyDict::new(py);
    d.set_item("nnz", indptr.last().copied().unwrap_or(0))?;
    d.set_item("nnz_q1", q(0.25))?;
    d.set_item("nnz_med", q(0.5))?;
    d.set_item("nnz_q3", q(0.75))?;
    d.set_item("nnz_max", per_row.last().copied().unwrap_or(0))?;
    Ok(d)
}

#[pyfunction]
fn scx_inspect(py: Python<'_>, input: &str) -> PyResult<Py<PyAny>> {
    let fmt = detect::detect(Path::new(input));
    let opts = scx_core::OpenOptions {
        metadata_only: true,
        ..scx_core::OpenOptions::new(1)
    };
    let mut reader = open_reader(input, opts).map_err(py_err)?;
    let reader = &mut *reader;

    let (obs, var, obsm, varm, uns, layer_metas, obsp_metas) = block_on(async {
        anyhow::Ok((
            reader.obs().await?,
            reader.var().await?,
            reader.obsm().await?,
            reader.varm().await?,
            reader.uns().await?,
            reader.layer_metas().await?,
            reader.obsp_metas().await?,
        ))
    })
    .map_err(py_err)?;

    // BPCells-backed H5Seurat has no CSR indptr for X.
    let x_indptr = reader.x_indptr();
    let format = match fmt {
        Some(Format::H5Seurat) if x_indptr.is_empty() => "H5Seurat (BPCells)",
        Some(f) => f.display_name(),
        None => "unknown",
    };

    let sparse_list = |metas: &[SparseMatrixMeta]| -> PyResult<Bound<'_, PyList>> {
        let list = PyList::empty(py);
        for m in metas {
            let entry = nnz_stats(py, &m.indptr)?;
            entry.set_item("name", &m.name)?;
            entry.set_item("n_obs", m.shape.0)?;
            entry.set_item("n_vars", m.shape.1)?;
            list.append(entry)?;
        }
        Ok(list)
    };

    let (n_obs, n_vars) = reader.shape();
    let d = PyDict::new(py);
    d.set_item("format", format)?;
    d.set_item("n_obs", n_obs)?;
    d.set_item("n_vars", n_vars)?;
    if x_indptr.len() > 1 {
        d.set_item("x_stats", nnz_stats(py, x_indptr)?)?;
    }
    let names = |cols: &[Column]| -> Vec<String> { cols.iter().map(|c| c.name.clone()).collect() };
    let dtypes = |cols: &[Column]| -> Vec<&'static str> {
        cols.iter().map(|c| c.data.dtype_str()).collect()
    };
    d.set_item("obs_cols", names(&obs.columns))?;
    d.set_item("obs_dtypes", dtypes(&obs.columns))?;
    d.set_item("var_cols", names(&var.columns))?;
    d.set_item("var_dtypes", dtypes(&var.columns))?;
    d.set_item("obsm_keys", obsm.map.keys().collect::<Vec<_>>())?;
    d.set_item("varm_keys", varm.map.keys().collect::<Vec<_>>())?;
    let uns_keys: Vec<&String> = uns
        .raw
        .as_object()
        .map(|o| o.keys().collect())
        .unwrap_or_default();
    d.set_item("uns_keys", uns_keys)?;
    d.set_item("layers", sparse_list(&layer_metas)?)?;
    d.set_item("obsp", sparse_list(&obsp_metas)?)?;
    Ok(d.into_any().unbind())
}

/// A chunk of consecutive rows of the matrix, as CSR numpy arrays.
///
/// The arrays take ownership of the buffers the reader decoded into, so
/// there is no per-chunk copy. They are writable and outlive the chunk.
#[pyclass(name = "MatrixChunk", module = "picklerick")]
pub struct PyMatrixChunk {
    /// Index of the first row of this chunk in the full matrix.
    #[pyo3(get)]
    pub row_offset: usize,
    /// Number of rows in this chunk.
    #[pyo3(get)]
    pub nrows: usize,
    /// Number of columns (features) of the full matrix.
    #[pyo3(get)]
    pub n_vars: usize,
    /// NumPy dtype name of `data`, e.g. ``"float32"``.
    #[pyo3(get)]
    pub dtype: String,
    /// ``(nrows + 1,)`` uint64 row pointers.
    #[pyo3(get)]
    pub indptr: Py<PyAny>,
    /// ``(nnz,)`` uint32 column indices.
    #[pyo3(get)]
    pub indices: Py<PyAny>,
    /// ``(nnz,)`` values.
    #[pyo3(get)]
    pub data: Py<PyAny>,
}

fn chunk_to_py(py: Python<'_>, chunk: MatrixChunk, n_vars: usize) -> PyMatrixChunk {
    let MatrixChunk {
        row_offset,
        nrows,
        data: csr,
    } = chunk;
    let dtype = csr.data.dtype().to_string();
    let data = match csr.data {
        TypedVec::F32(v) => v.into_pyarray(py).into_any().unbind(),
        TypedVec::F64(v) => v.into_pyarray(py).into_any().unbind(),
        TypedVec::I32(v) => v.into_pyarray(py).into_any().unbind(),
        TypedVec::U32(v) => v.into_pyarray(py).into_any().unbind(),
    };
    PyMatrixChunk {
        row_offset,
        nrows,
        n_vars,
        dtype,
        indptr: csr.indptr.into_pyarray(py).into_any().unbind(),
        indices: csr.indices.into_pyarray(py).into_any().unbind(),
        data,
    }
}

/// Iterator over `MatrixChunk`s, filled by a background reader thread.
#[pyclass(name = "MatrixStream", module = "picklerick")]
pub struct PyMatrixStream {
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<Result<MatrixChunk, String>>>,
    n_vars: usize,
}

#[pymethods]
impl PyMatrixStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyMatrixChunk>> {
        let received = py.detach(|| self.rx.lock().unwrap().recv());
        match received {
            Ok(Ok(chunk)) => Ok(Some(chunk_to_py(py, chunk, self.n_vars))),
            Ok(Err(msg)) => Err(PickleRickError::new_err(msg)),
            Err(_) => Ok(None), // reader thread finished
        }
    }
}

#[pyfunction]
fn scx_open_stream(
    path: &str,
    chunk_size: usize,
    assay: &str,
    layer: &str,
) -> PyResult<PyMatrixStream> {
    let mut reader = open_reader(path, stream_opts(chunk_size, assay, layer)).map_err(py_err)?;
    let (_, n_vars) = reader.shape();
    // Bounded so the reader stays at most 8 chunks ahead of the consumer.
    let (tx, rx) = std::sync::mpsc::sync_channel(8);

    std::thread::spawn(move || {
        block_on(async move {
            let mut stream = reader.x_stream();
            while let Some(chunk) = stream.next().await {
                if tx.send(chunk.map_err(|e| e.to_string())).is_err() {
                    break; // consumer dropped the iterator
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
fn picklerick_py_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PickleRickError", m.py().get_type::<PickleRickError>())?;
    m.add_function(wrap_pyfunction!(scx_convert, m)?)?;
    m.add_function(wrap_pyfunction!(scx_inspect, m)?)?;
    m.add_function(wrap_pyfunction!(scx_open_stream, m)?)?;
    m.add_class::<PyMatrixChunk>()?;
    m.add_class::<PyMatrixStream>()?;
    Ok(())
}
