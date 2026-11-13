use std::ops::Range;
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
    h5seurat::{auto_counts_source, CountsSource, H5SeuratWriter},
    ir::{mask_to_ranges, Column, MatrixChunk, SparseMatrixMeta},
    stream::{DatasetReader, DatasetWriter},
    subset::SubsetReader,
    zarr_ad::ZarrAdWriter,
};

pyo3::create_exception!(
    picklerick,
    PickleRickError,
    pyo3::exceptions::PyException,
    "Raised when the SCX engine cannot read, convert or write a dataset."
);

fn py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PickleRickError::new_err(e.to_string())
}

fn open_reader(path: &str, opts: scx_core::OpenOptions) -> anyhow::Result<Box<dyn DatasetReader>> {
    Ok(block_on(scx_core::open(path, &opts))?)
}

/// `assay: None` leaves the choice to the reader: RNA for H5Seurat, the
/// object's active assay for a Seurat `.rds`.
fn stream_opts(chunk_size: usize, assay: Option<&str>, layer: &str) -> scx_core::OpenOptions {
    scx_core::OpenOptions {
        source_hash: Default::default(),
        assay: assay.map(str::to_string),
        layer: Some(layer.to_string()),
        ..scx_core::OpenOptions::new(chunk_size)
    }
}

/// Pick the writer from the output extension. H5Seurat output is the
/// SeuratDisk-compatible dgCMatrix layout, not the CLI's BPCells default.
///
/// `x_slot` is the Seurat slot for H5Seurat output, or `"X"` / `"layers/<name>"`
/// for H5AD output.
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
        Some("h5ad") => Box::new(H5AdWriter::create_at(
            output, n_obs, n_vars, dtype, None, x_slot,
        )?),
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
        Some("zarr") => Box::new(ZarrAdWriter::create_at(
            output, n_obs, n_vars, dtype, x_slot,
        )?),
        _ => anyhow::bail!(
            "unsupported output extension for '{}': use .h5ad, .h5seurat or .zarr",
            output.display()
        ),
    })
}

async fn convert(
    reader: &mut dyn DatasetReader,
    writer: &mut dyn DatasetWriter,
    layer_metas: &[SparseMatrixMeta],
    skip_layer: Option<&str>,
    raw_counts: Option<&SparseMatrixMeta>,
    chunk_size: usize,
    csc_mem: Option<usize>,
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
    // AnnData's raw.X is the counts of a normalised X: Seurat's `counts` slot.
    if let Some(meta) = raw_counts {
        writer.begin_sparse("layers", "counts", meta).await?;
        let mut stream = reader.raw_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    } else {
        // Elsewhere raw stays raw, in formats that have it.
        scx_core::stream::copy_raw(reader, writer, chunk_size).await?;
    }
    for meta in &reader.obsp_metas().await? {
        writer.begin_sparse("obsp", &meta.name, meta).await?;
        let mut stream = reader.obsp_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    if let Some(mem) = csc_mem {
        // Gene-major X, holding at most `mem` bytes of transposed nonzeros.
        scx_core::transpose::write_x_csc(reader, writer, mem).await?;
    } else {
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            writer.write_x_chunk(&chunk?).await?;
        }
    }
    writer.finalize().await?;
    Ok(())
}

/// H5Seurat output follows `scx convert --x-slot auto`: when the source has a
/// `counts` layer, or a raw.X with X's genes, X is taken to be normalised and
/// goes to the `data` slot (`scx_core::h5seurat::auto_counts_source`).
/// H5AD output writes the matrix to `x_slot` (`"X"` or `"layers/<name>"`).
/// `transpose_mem` (bytes), when set, writes X gene-major (CSC), as
/// `scx convert --gene-major` does; H5AD output only. `where` keeps only the
/// matching cells (`scx_core::subset::SubsetReader`), as `--where` does.
#[pyfunction]
#[pyo3(signature = (input, output, chunk_size, dtype, assay, layer, x_slot = "X", transpose_mem = None, r#where = None))]
#[allow(clippy::too_many_arguments)]
fn scx_convert(
    input: &str,
    output: &str,
    chunk_size: usize,
    dtype: &str,
    assay: Option<&str>,
    layer: &str,
    x_slot: &str,
    transpose_mem: Option<usize>,
    r#where: Option<Vec<(String, Vec<String>)>>,
) -> PyResult<()> {
    let run = || -> anyhow::Result<()> {
        let dtype = dtype.parse::<DataType>()?;
        let is_h5ad = Path::new(output)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("h5ad"));
        if transpose_mem.is_some() && !is_h5ad {
            anyhow::bail!("gene_major only applies to .h5ad output");
        }
        let mut reader = open_reader(input, stream_opts(chunk_size, assay, layer))?;
        if let Some(filters) = r#where.filter(|f| !f.is_empty()) {
            reader = Box::new(block_on(SubsetReader::new(reader, &filters, chunk_size))?);
        }
        let (n_obs, n_vars) = reader.shape();
        let layer_metas = block_on(reader.layer_metas())?;
        let output = Path::new(output);
        let is_seurat = output
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("h5seurat"));
        let counts_source = if is_seurat {
            block_on(auto_counts_source(&mut *reader, &layer_metas))?
        } else {
            None
        };
        let slot = if !is_seurat {
            x_slot
        } else if counts_source.is_some() {
            "data"
        } else {
            "counts"
        };
        let raw_counts = match &counts_source {
            Some(CountsSource::Raw(meta)) => Some(meta),
            _ => None,
        };
        let mut writer = create_writer(output, n_obs, n_vars, dtype, assay.unwrap_or("RNA"), slot)?;
        // A source layer with the destination's name can't coexist with it.
        let skip = if is_seurat {
            Some(slot)
        } else {
            slot.strip_prefix("layers/")
        };
        block_on(convert(
            &mut *reader,
            &mut *writer,
            &layer_metas,
            skip,
            raw_counts,
            chunk_size,
            transpose_mem,
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
#[pyo3(signature = (input, with_names = false))]
fn scx_inspect(py: Python<'_>, input: &str, with_names: bool) -> PyResult<Py<PyAny>> {
    let fmt = detect::detect(Path::new(input));
    let opts = scx_core::OpenOptions {
        source_hash: Default::default(),
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
    // The axis X streams along: per-gene stats for gene-major X.
    let (x_axis, x_ptr) = match reader.x_col_ptr() {
        [] => ("obs", x_indptr),
        col_ptr => ("var", col_ptr),
    };
    d.set_item("x_axis", x_axis)?;
    if x_ptr.len() > 1 {
        d.set_item("x_stats", nnz_stats(py, x_ptr)?)?;
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
    d.set_item("uns_keys", uns.keys())?;
    if with_names {
        d.set_item("obs_names", &obs.index)?;
        d.set_item("var_names", &var.index)?;
    }
    d.set_item("layers", sparse_list(&layer_metas)?)?;
    d.set_item("obsp", sparse_list(&obsp_metas)?)?;
    Ok(d.into_any().unbind())
}

/// A block of the matrix, as compressed numpy arrays: consecutive rows
/// (``axis == "obs"``, CSR, every column) or consecutive genes
/// (``axis == "var"``, CSC, every row). The block is rows
/// ``row_offset:row_offset + nrows`` by columns
/// ``var_offset:var_offset + ncols``.
///
/// The arrays take ownership of the buffers the reader decoded into, so
/// there is no per-chunk copy. They are writable and outlive the chunk.
#[pyclass(name = "MatrixChunk", module = "picklerick")]
pub struct PyMatrixChunk {
    /// ``"obs"`` for a block of rows (CSR), ``"var"`` for a block of genes (CSC).
    #[pyo3(get)]
    pub axis: &'static str,
    /// Index of the first row of this chunk in the full matrix.
    #[pyo3(get)]
    pub row_offset: usize,
    /// Number of rows in this chunk.
    #[pyo3(get)]
    pub nrows: usize,
    /// Index of the first column of this chunk in the full matrix.
    #[pyo3(get)]
    pub var_offset: usize,
    /// Number of columns in this chunk.
    #[pyo3(get)]
    pub ncols: usize,
    /// Number of columns (features) of the full matrix.
    #[pyo3(get)]
    pub n_vars: usize,
    /// NumPy dtype name of `data`, e.g. ``"float32"``.
    #[pyo3(get)]
    pub dtype: String,
    /// uint64 pointers: ``(nrows + 1,)`` per row for ``"obs"``,
    /// ``(ncols + 1,)`` per gene for ``"var"``.
    #[pyo3(get)]
    pub indptr: Py<PyAny>,
    /// ``(nnz,)`` uint32 column indices (``"obs"``) or row indices (``"var"``).
    #[pyo3(get)]
    pub indices: Py<PyAny>,
    /// ``(nnz,)`` values.
    #[pyo3(get)]
    pub data: Py<PyAny>,
}

#[pymethods]
impl PyMatrixChunk {
    /// The chunk as a ``scipy.sparse.csr_array`` (``"obs"``) or
    /// ``csc_array`` (``"var"``) of shape ``(nrows, ncols)`` that shares
    /// ``data`` and ``indices`` with this chunk. scipy wants signed indices
    /// of one dtype: ``indices`` is viewed as int32 (same bits, never
    /// negative) and ``indptr`` is cast to int32, a copy of only its few
    /// entries. Given an int64 indptr, scipy would upcast, and so copy,
    /// ``indices``.
    fn to_scipy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (minor, what, class) = match self.axis {
            "var" => (self.nrows, "rows", "csc_array"),
            _ => (self.ncols, "columns", "csr_array"),
        };
        if minor > i32::MAX as usize {
            return Err(PickleRickError::new_err(format!(
                "to_scipy: {minor} {what} don't fit int32 indices"
            )));
        }
        let indices = self.indices.bind(py).call_method1("view", ("int32",))?;
        let nnz = self.data.bind(py).len()?;
        let indptr = if nnz <= i32::MAX as usize {
            self.indptr.bind(py).call_method1("astype", ("int32",))?
        } else {
            self.indptr.bind(py).call_method1("view", ("int64",))?
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", (self.nrows, self.ncols))?;
        kwargs.set_item("copy", false)?;
        py.import("scipy.sparse")?
            .getattr(class)?
            .call(((self.data.bind(py), indices, indptr),), Some(&kwargs))
    }
}

/// `n_obs` is set for a gene-axis stream, whose chunks span every row.
fn chunk_to_py(
    py: Python<'_>,
    chunk: MatrixChunk,
    n_vars: usize,
    n_obs: Option<usize>,
) -> PyMatrixChunk {
    let MatrixChunk {
        row_offset: offset,
        nrows: n,
        data: csr,
    } = chunk;
    // A gene-axis chunk is the CSR of its block's transpose: n genes from
    // `offset`, over every row.
    let (axis, row_offset, nrows, var_offset, ncols) = match n_obs {
        Some(n_obs) => ("var", 0, n_obs, offset, n),
        None => ("obs", offset, n, 0, n_vars),
    };
    let dtype = csr.data.dtype().to_string();
    let data = match csr.data {
        TypedVec::F32(v) => v.into_pyarray(py).into_any().unbind(),
        TypedVec::F64(v) => v.into_pyarray(py).into_any().unbind(),
        TypedVec::I32(v) => v.into_pyarray(py).into_any().unbind(),
        TypedVec::U32(v) => v.into_pyarray(py).into_any().unbind(),
    };
    PyMatrixChunk {
        axis,
        row_offset,
        nrows,
        var_offset,
        ncols,
        n_vars,
        dtype,
        indptr: csr.indptr.into_pyarray(py).into_any().unbind(),
        indices: csr.indices.into_pyarray(py).into_any().unbind(),
        data,
    }
}

/// Most chunks the reader thread may decode ahead of the consumer.
const READAHEAD_CHUNKS: usize = 8;

/// Bytes a decoded chunk holds (indptr, indices, values).
fn chunk_bytes(c: &MatrixChunk) -> usize {
    let m = &c.data;
    m.indptr.len() * 8 + m.indices.len() * 4 + m.data.len() * m.data.dtype().size()
}

/// The hand-off between the reader thread and the consumer: chunks decoded
/// ahead, bounded by count and by bytes. A count alone lets the queue grow
/// with the chunk size (8 chunks of 5,000 atlas cells are over 600 MB behind
/// a slow consumer). One chunk is always let in, so a chunk larger than the
/// budget still streams.
struct Readahead {
    state: std::sync::Mutex<ReadaheadState>,
    changed: std::sync::Condvar,
    budget: usize,
}

#[derive(Default)]
struct ReadaheadState {
    queue: std::collections::VecDeque<(usize, Result<MatrixChunk, String>)>,
    bytes: usize,
    /// The reader thread has nothing more to push.
    done: bool,
    /// The consumer dropped the stream.
    closed: bool,
}

impl Readahead {
    fn new(budget: usize) -> Self {
        Self {
            state: Default::default(),
            changed: Default::default(),
            budget,
        }
    }

    /// Blocks while the queue is full; false once the consumer is gone.
    fn push(&self, item: Result<MatrixChunk, String>) -> bool {
        let n = item.as_ref().map_or(0, chunk_bytes);
        let mut s = self.state.lock().unwrap();
        while !s.closed
            && !s.queue.is_empty()
            && (s.queue.len() >= READAHEAD_CHUNKS || s.bytes + n > self.budget)
        {
            s = self.changed.wait(s).unwrap();
        }
        if s.closed {
            return false;
        }
        s.bytes += n;
        s.queue.push_back((n, item));
        self.changed.notify_all();
        true
    }

    /// Blocks until a chunk is there; `None` when the reader has finished.
    fn pop(&self) -> Option<Result<MatrixChunk, String>> {
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some((n, item)) = s.queue.pop_front() {
                s.bytes -= n;
                self.changed.notify_all();
                return Some(item);
            }
            if s.done {
                return None;
            }
            s = self.changed.wait(s).unwrap();
        }
    }

    fn set(&self, f: impl FnOnce(&mut ReadaheadState)) {
        f(&mut self.state.lock().unwrap());
        self.changed.notify_all();
    }
}

/// Iterator over `MatrixChunk`s, filled by a background reader thread.
#[pyclass(name = "MatrixStream", module = "picklerick")]
pub struct PyMatrixStream {
    ahead: std::sync::Arc<Readahead>,
    n_vars: usize,
    /// Set when streaming by gene: every chunk spans these rows.
    n_obs: Option<usize>,
}

impl Drop for PyMatrixStream {
    fn drop(&mut self) {
        // The reader thread must stop, not wait on a full queue.
        self.ahead.set(|s| s.closed = true);
    }
}

#[pymethods]
impl PyMatrixStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyMatrixChunk>> {
        match py.detach(|| self.ahead.pop()) {
            Some(Ok(chunk)) => Ok(Some(chunk_to_py(py, chunk, self.n_vars, self.n_obs))),
            Some(Err(msg)) => Err(PickleRickError::new_err(msg)),
            None => Ok(None), // reader thread finished
        }
    }

    /// Bytes of decoded chunks waiting for the consumer (for tests).
    fn _queued_bytes(&self) -> usize {
        self.ahead.state.lock().unwrap().bytes
    }
}

/// `rows` are (start, stop) half-open row ranges; `where` maps obs columns to
/// the accepted values, all of which must hold. Both are turned into row
/// ranges; `rows` and `where` together take their intersection. `by_gene`
/// streams by gene instead, over the (start, stop) ranges in `genes`, or
/// every gene when that is `None`. `readahead_bytes` bounds the chunks decoded
/// ahead of the consumer.
#[pyfunction]
#[pyo3(signature = (path, chunk_size, assay, layer, rows = None, r#where = None, by_gene = false, genes = None, readahead_bytes = 128 << 20))]
#[allow(clippy::too_many_arguments)]
fn scx_open_stream(
    path: &str,
    chunk_size: usize,
    assay: Option<&str>,
    layer: &str,
    rows: Option<Vec<(usize, usize)>>,
    r#where: Option<Vec<(String, Vec<String>)>>,
    by_gene: bool,
    genes: Option<Vec<(usize, usize)>>,
    readahead_bytes: usize,
) -> PyResult<PyMatrixStream> {
    let mut reader = open_reader(path, stream_opts(chunk_size, assay, layer)).map_err(py_err)?;
    let (n_obs, n_vars) = reader.shape();
    let genes: Option<Vec<Range<usize>>> = by_gene.then(|| {
        genes.map_or(std::iter::once(0..n_vars).collect(), |g| {
            g.into_iter().map(|(a, b)| a..b).collect()
        })
    });
    let stream_n_obs = genes.is_some().then_some(n_obs);
    if genes.is_none() && !reader.x_col_ptr().is_empty() {
        return Err(py_err(
            "X is gene-major (CSC): stream it by gene with open_stream(axis=\"var\"), \
             or convert it to CSR to stream by cell",
        ));
    }
    if let Some((a, b)) = rows.iter().flatten().find(|&&(_, b)| b > n_obs) {
        return Err(py_err(format!(
            "row range {a}..{b} is beyond the {n_obs} rows"
        )));
    }
    let ranges = match (rows, r#where) {
        (None, None) => None,
        // No filter: the reader takes the ranges as they are, so a dask block
        // costs nothing in n_obs here.
        (Some(rows), None) => Some(rows.into_iter().map(|(a, b)| a..b).collect()),
        (rows, Some(filters)) => {
            let mut mask = vec![false; n_obs];
            for (a, b) in rows.unwrap_or_else(|| vec![(0, n_obs)]) {
                mask[a..b.max(a)].fill(true);
            }
            let obs = block_on(reader.obs()).map_err(py_err)?;
            let hit = obs.rows_matching(n_obs, &filters).map_err(py_err)?;
            mask.iter_mut().zip(hit).for_each(|(m, h)| *m &= h);
            Some(mask_to_ranges(&mask))
        }
    };
    let ahead = std::sync::Arc::new(Readahead::new(readahead_bytes));
    let tx = ahead.clone();

    std::thread::spawn(move || {
        block_on(async {
            let mut stream = match (genes, ranges) {
                (Some(genes), _) => reader.x_stream_cols(genes),
                (None, Some(ranges)) => reader.x_stream_rows(ranges),
                (None, None) => reader.x_stream(),
            };
            while let Some(chunk) = stream.next().await {
                if !tx.push(chunk.map_err(|e| e.to_string())) {
                    break; // consumer dropped the iterator
                }
            }
        });
        tx.set(|s| s.done = true);
    });

    Ok(PyMatrixStream {
        ahead,
        n_vars,
        n_obs: stream_n_obs,
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
