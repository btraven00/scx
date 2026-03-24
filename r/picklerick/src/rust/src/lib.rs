//! picklerick-r — extendr bindings to scx-core
//!
//! # Status
//!
//! Phase B: `scx_convert` runs in-process via extendr FFI.
//! HDF5 is statically linked (hdf5-sys features=["static"]) so this .so has
//! its own isolated HDF5 instance that does not share global state with R's
//! rhdf5 / hdf5r shared libhdf5.so.

use extendr_api::prelude::*;
use futures::StreamExt;
use scx_core::{
    detect,
    detect::Format,
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::{H5SeuratReader, H5SeuratWriter},
    stream::{DatasetReader, DatasetWriter},
};
use std::path::Path;

// ---------------------------------------------------------------------------
// Async bridge: spin up a single-threaded tokio runtime per call.
// This is appropriate for one-shot CLI-style invocations from R.
// ---------------------------------------------------------------------------

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(fut)
}

// ---------------------------------------------------------------------------
// Core conversion helper (mirrors scx-cli/src/main.rs:convert_with_reader)
// ---------------------------------------------------------------------------

async fn do_convert(
    reader: &mut dyn DatasetReader,
    output: &Path,
    dtype: DataType,
) -> anyhow::Result<()> {
    let (n_obs, n_vars) = reader.shape();
    let obs    = reader.obs().await?;
    let var    = reader.var().await?;
    let obsm   = reader.obsm().await?;
    let uns    = reader.uns().await?;
    let layers = reader.layers().await?;
    let obsp   = reader.obsp().await?;
    let varp   = reader.varp().await?;
    let varm   = reader.varm().await?;

    let mut writer = H5AdWriter::create(output, n_obs, n_vars, dtype)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;
    writer.write_layers(&layers).await?;
    writer.write_obsp(&obsp).await?;
    writer.write_varp(&varp).await?;
    writer.write_varm(&varm).await?;

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk?).await?;
    }
    writer.finalize().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Exported function: scx_convert
//
// Called from R as `.Call(picklerick:::scx_convert, ...)` once Phase B is
// active.  During Phase A this function compiles but is never called from R;
// `R/convert.R` uses `system2("scx", ...)` instead.
// ---------------------------------------------------------------------------

/// Convert any supported single-cell format to H5AD.
///
/// @param input Path to input file (.h5seurat, .h5ad, .h5).
/// @param output Path to output .h5ad file.
/// @param chunk_size Number of cells per streaming chunk.
/// @param dtype Output numeric type: "f32", "f64", "i32", "u32".
/// @param assay Seurat assay name (H5Seurat only).
/// @param layer Seurat layer name (H5Seurat only).
/// @noRd
#[extendr]
fn scx_convert(
    input:      &str,
    output:     &str,
    chunk_size: i32,
    dtype:      &str,
    assay:      &str,
    layer:      &str,
) -> Result<()> {
    let dtype = match dtype {
        "f32" => DataType::F32,
        "f64" => DataType::F64,
        "i32" => DataType::I32,
        "u32" => DataType::U32,
        other => return Err(Error::from(format!("unknown dtype '{other}'"))),
    };

    let chunk = chunk_size as usize;
    let input_path = Path::new(input);

    let fmt = detect::sniff(input_path).or_else(|| {
        match input_path.extension().and_then(|e| e.to_str()) {
            Some("h5seurat") => Some(Format::H5Seurat),
            Some("h5ad")     => Some(Format::H5Ad),
            _                => Some(Format::ScxH5),
        }
    });

    let result = block_on(async {
        match fmt {
            Some(Format::H5Seurat) => {
                let mut r = H5SeuratReader::open(input_path, chunk, Some(assay), Some(layer))
                    .map_err(anyhow::Error::from)?;
                do_convert(&mut r, Path::new(output), dtype).await
            }
            Some(Format::H5Ad) | None => {
                let mut r = H5AdReader::open(input_path, chunk)
                    .map_err(anyhow::Error::from)?;
                do_convert(&mut r, Path::new(output), dtype).await
            }
            Some(Format::ScxH5) => {
                let mut r = ScxH5Reader::open(input_path, chunk)
                    .map_err(anyhow::Error::from)?;
                do_convert(&mut r, Path::new(output), dtype).await
            }
        }
    });

    result.map_err(|e| Error::from(e.to_string()))
}

// ---------------------------------------------------------------------------
// H5Seurat write helper
// ---------------------------------------------------------------------------

async fn do_convert_h5seurat(
    reader: &mut dyn DatasetReader,
    output: &Path,
    dtype: DataType,
    assay: &str,
) -> anyhow::Result<()> {
    let (n_obs, n_vars) = reader.shape();
    let obs    = reader.obs().await?;
    let var    = reader.var().await?;
    let obsm   = reader.obsm().await?;
    let uns    = reader.uns().await?;
    let layers = reader.layers().await?;
    let obsp   = reader.obsp().await?;
    let varp   = reader.varp().await?;
    let varm   = reader.varm().await?;

    let mut writer = H5SeuratWriter::create(output, n_obs, n_vars, dtype, Some(assay), None)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;
    writer.write_layers(&layers).await?;
    writer.write_obsp(&obsp).await?;
    writer.write_varp(&varp).await?;
    writer.write_varm(&varm).await?;

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        writer.write_x_chunk(&chunk?).await?;
    }
    writer.finalize().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Exported function: scx_write_h5seurat
// ---------------------------------------------------------------------------

/// Convert an H5AD file to H5Seurat format.
///
/// @param input  Path to the input `.h5ad` file.
/// @param output Path to the output `.h5seurat` file.
/// @param chunk_size Number of cells per streaming chunk.
/// @param assay Seurat assay name to write (default `"RNA"`).
/// @noRd
#[extendr]
fn scx_write_h5seurat(
    input:      &str,
    output:     &str,
    chunk_size: i32,
    assay:      &str,
) -> Result<()> {
    let chunk = chunk_size as usize;
    let input_path  = Path::new(input);
    let output_path = Path::new(output);

    let result = block_on(async {
        let mut r = H5AdReader::open(input_path, chunk).map_err(anyhow::Error::from)?;
        do_convert_h5seurat(&mut r, output_path, DataType::F32, assay).await
    });

    result.map_err(|e| Error::from(e.to_string()))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod picklerick;
    fn scx_convert;
    fn scx_write_h5seurat;
}
