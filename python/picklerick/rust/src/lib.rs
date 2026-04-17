use std::path::Path;

use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use scx_core::{
    detect,
    detect::Format,
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::H5SeuratWriter,
    stream::{DatasetReader, DatasetWriter},
};

fn py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(fut)
}

fn parse_dtype(dtype: &str) -> PyResult<DataType> {
    match dtype {
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "i32" => Ok(DataType::I32),
        "u32" => Ok(DataType::U32),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown dtype '{other}': use f32, f64, i32, or u32"
        ))),
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
        other => Err(anyhow::anyhow!(
            "unsupported input format for native conversion: {other:?}"
        )),
    }
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

    let mut writer = H5SeuratWriter::create(output, n_obs, n_vars, dtype, Some(assay), None, None, false)?;
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
    let out_dtype = parse_dtype(dtype)?;
    let input_path = Path::new(input);
    let output_path = Path::new(output);

    let result = block_on(async {
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
        let mut reader = H5AdReader::open(input_path, chunk_size).map_err(anyhow::Error::from)?;
        do_convert_h5seurat(&mut reader, output_path, DataType::F32, assay, chunk_size).await
    });

    result.map_err(py_err)
}

#[pymodule]
fn picklerick_py_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scx_convert_native, m)?)?;
    m.add_function(wrap_pyfunction!(scx_write_h5seurat_native, m)?)?;
    Ok(())
}