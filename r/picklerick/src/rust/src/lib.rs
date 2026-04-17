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
    bpcells::BpcellsDatasetReader,
    detect,
    detect::Format,
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::H5SeuratReader,
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
    let obs          = reader.obs().await?;
    let var          = reader.var().await?;
    let obsm         = reader.obsm().await?;
    let uns          = reader.uns().await?;
    let varm         = reader.varm().await?;
    let layer_metas  = reader.layer_metas().await?;
    let obsp_metas   = reader.obsp_metas().await?;

    let mut writer = H5AdWriter::create(output, n_obs, n_vars, dtype)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;
    writer.write_varm(&varm).await?;

    for meta in &layer_metas {
        writer.begin_sparse("layers", &meta.name, meta).await?;
        let mut stream = reader.layer_stream(meta, n_obs);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    for meta in &obsp_metas {
        writer.begin_sparse("obsp", &meta.name, meta).await?;
        let mut stream = reader.obsp_stream(meta, n_obs);
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
            Some(Format::BPCells) => {
                let mut r = BpcellsDatasetReader::open(input_path, chunk)
                    .map_err(anyhow::Error::from)?;
                do_convert(&mut r, Path::new(output), dtype).await
            }
            Some(Format::NpyDir) => {
                Err(anyhow::anyhow!("NpyDir format is not supported"))
            }
        }
    });

    result.map_err(|e| Error::from(e.to_string()))
}

// ---------------------------------------------------------------------------
// Exported function: scx_inspect
// ---------------------------------------------------------------------------

/// Inspect a single-cell file and return metadata as a named list.
///
/// @param input      Path to the file (.h5seurat, .h5ad, .h5).
/// @param chunk_size Cells per streaming chunk (affects obs/var read only).
/// @return A named list with format, n_obs, n_vars, obs_cols, var_cols,
///   obsm_keys, layers, uns_keys, obsp_keys, varp_keys, varm_keys.
/// @noRd
#[extendr]
fn scx_inspect(input: &str, chunk_size: i32) -> Result<Robj> {
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
                let mut r = H5SeuratReader::open(input_path, chunk, None, None)
                    .map_err(anyhow::Error::from)?;
                collect_info(&mut r, "H5Seurat").await
            }
            Some(Format::H5Ad) | None => {
                let mut r = H5AdReader::open(input_path, chunk)
                    .map_err(anyhow::Error::from)?;
                collect_info(&mut r, "H5AD").await
            }
            Some(Format::ScxH5) => {
                let mut r = ScxH5Reader::open(input_path, chunk)
                    .map_err(anyhow::Error::from)?;
                collect_info(&mut r, "ScxH5").await
            }
            Some(Format::BPCells) => {
                let mut r = BpcellsDatasetReader::open_metadata_only(input_path)
                    .map_err(anyhow::Error::from)?;
                collect_info(&mut r, "BPCells").await
            }
            Some(Format::NpyDir) => {
                Err(anyhow::anyhow!("NpyDir format is not supported"))
            }
        }
    });

    result.map_err(|e| Error::from(e.to_string()))
}

async fn collect_info(
    reader: &mut dyn DatasetReader,
    format_name: &str,
) -> anyhow::Result<Robj> {
    let (n_obs, n_vars) = reader.shape();
    let obs          = reader.obs().await?;
    let var          = reader.var().await?;
    let obsm         = reader.obsm().await?;
    let uns          = reader.uns().await?;
    let layer_metas  = reader.layer_metas().await?;
    let obsp_metas   = reader.obsp_metas().await?;
    let varm         = reader.varm().await?;

    let obs_cols:   Vec<String> = obs.columns.iter().map(|c| c.name.clone()).collect();
    let var_cols:   Vec<String> = var.columns.iter().map(|c| c.name.clone()).collect();
    let obsm_keys:  Vec<String> = obsm.map.keys().cloned().collect();
    let layer_keys: Vec<String> = layer_metas.iter().map(|m| m.name.clone()).collect();
    let obsp_keys:  Vec<String> = obsp_metas.iter().map(|m| m.name.clone()).collect();
    let varm_keys:  Vec<String> = varm.map.keys().cloned().collect();
    let uns_keys:   Vec<String> = uns.raw
        .as_object()
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    Ok(list!(
        format    = format_name,
        n_obs     = n_obs as i32,
        n_vars    = n_vars as i32,
        obs_cols  = obs_cols,
        var_cols  = var_cols,
        obsm_keys = obsm_keys,
        layers    = layer_keys,
        uns_keys  = uns_keys,
        obsp_keys = obsp_keys,
        varm_keys = varm_keys
    ).into_robj())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod picklerick;
    fn scx_convert;
    fn scx_inspect;
}
