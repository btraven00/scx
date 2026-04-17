#![allow(clippy::useless_conversion)]

use std::path::Path;

use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use scx_core::{
    bpcells::BpcellsDatasetReader,
    detect,
    detect::Format,
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::H5SeuratWriter,
    ir::ColumnData,
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
        other => Err(anyhow::anyhow!(
            "unsupported input format: {other:?}"
        )),
    }
}

fn open_reader_metadata_only(input_path: &Path) -> anyhow::Result<Box<dyn DatasetReader>> {
    let fmt = detect_format(input_path);
    match fmt {
        Some(Format::H5Seurat) => {
            let reader = scx_core::h5seurat::open_h5seurat(input_path, 1, None, None)?;
            Ok(reader)
        }
        Some(Format::H5Ad) | None => {
            Ok(Box::new(H5AdReader::open(input_path, 1)?))
        }
        Some(Format::ScxH5) => {
            Ok(Box::new(ScxH5Reader::open(input_path, 1)?))
        }
        Some(Format::BPCells) => {
            Ok(Box::new(BpcellsDatasetReader::open_metadata_only(input_path)?))
        }
        other => Err(anyhow::anyhow!("unsupported input format: {other:?}")),
    }
}

async fn collect_inspect_info(
    reader: &mut dyn DatasetReader,
    format_name: &str,
    py: Python<'_>,
) -> anyhow::Result<PyObject> {
    let (n_obs, n_vars) = reader.shape();
    let obs         = reader.obs().await?;
    let var         = reader.var().await?;
    let obsm        = reader.obsm().await?;
    let uns         = reader.uns().await?;
    let layer_metas = reader.layer_metas().await?;
    let obsp_metas  = reader.obsp_metas().await?;
    let varm        = reader.varm().await?;

    let obs_cols:   Vec<&str> = obs.columns.iter().map(|c| c.name.as_str()).collect();
    let var_cols:   Vec<&str> = var.columns.iter().map(|c| c.name.as_str()).collect();
    let obsm_keys:  Vec<&str> = obsm.map.keys().map(|s| s.as_str()).collect();
    let layer_keys: Vec<&str> = layer_metas.iter().map(|m| m.name.as_str()).collect();
    let obsp_keys:  Vec<&str> = obsp_metas.iter().map(|m| m.name.as_str()).collect();
    let varm_keys:  Vec<&str> = varm.map.keys().map(|s| s.as_str()).collect();
    let uns_keys:   Vec<String> = uns.raw
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();

    // obs col dtypes
    let obs_dtypes: Vec<&str> = obs.columns.iter().map(|c| match &c.data {
        ColumnData::Float(_)        => "float64",
        ColumnData::Int(_)          => "int32",
        ColumnData::Bool(_)         => "bool",
        ColumnData::String(_)       => "string",
        ColumnData::Categorical {..}=> "categorical",
    }).collect();
    let var_dtypes: Vec<&str> = var.columns.iter().map(|c| match &c.data {
        ColumnData::Float(_)        => "float64",
        ColumnData::Int(_)          => "int32",
        ColumnData::Bool(_)         => "bool",
        ColumnData::String(_)       => "string",
        ColumnData::Categorical {..}=> "categorical",
    }).collect();

    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("format",     format_name)?;
    d.set_item("n_obs",      n_obs as i64)?;
    d.set_item("n_vars",     n_vars as i64)?;
    d.set_item("obs_cols",   obs_cols)?;
    d.set_item("obs_dtypes", obs_dtypes)?;
    d.set_item("var_cols",   var_cols)?;
    d.set_item("var_dtypes", var_dtypes)?;
    d.set_item("obsm_keys",  obsm_keys)?;
    d.set_item("layers",     layer_keys)?;
    d.set_item("uns_keys",   uns_keys)?;
    d.set_item("obsp_keys",  obsp_keys)?;
    d.set_item("varm_keys",  varm_keys)?;
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
        Some(Format::H5Ad)     => "H5AD",
        Some(Format::ScxH5)    => "ScxH5",
        Some(Format::BPCells)  => "BPCells",
        _                      => "unknown",
    };

    let result = block_on(async {
        let mut reader = open_reader_metadata_only(input_path)?;
        collect_inspect_info(&mut *reader, format_name, py).await
    });

    result.map_err(py_err)
}

#[pymodule]
fn picklerick_py_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scx_convert_native, m)?)?;
    m.add_function(wrap_pyfunction!(scx_write_h5seurat_native, m)?)?;
    m.add_function(wrap_pyfunction!(scx_inspect_native, m)?)?;
    Ok(())
}
