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
    dtype::{DataType, TypedVec},
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::H5SeuratReader,
    ir::{
        Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        UnsTable, VarTable, Varm,
    },
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
            Some(Format::TenxH5) | Some(Format::PlainH5) => {
                Err(anyhow::anyhow!("10x / plain H5 input is not supported by picklerick"))
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
                let fmt_name = if r.x_indptr().is_empty() { "H5Seurat (BPCells)" } else { "H5Seurat" };
                collect_info(&mut r, fmt_name).await
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
            Some(Format::TenxH5) | Some(Format::PlainH5) => {
                Err(anyhow::anyhow!("10x / plain H5 input is not supported by picklerick"))
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

    let obs_cols:  Vec<String> = obs.columns.iter().map(|c| c.name.clone()).collect();
    let var_cols:  Vec<String> = var.columns.iter().map(|c| c.name.clone()).collect();
    let obsm_keys: Vec<String> = obsm.map.keys().cloned().collect();
    let varm_keys: Vec<String> = varm.map.keys().cloned().collect();
    let uns_keys:  Vec<String> = uns.raw
        .as_object()
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    // per-layer and per-obsp nnz stats (free from indptr, no matrix streaming)
    // Returned as parallel named vectors — R-idiomatic, avoids nested list issues.
    fn sparse_stats(metas: &[scx_core::ir::SparseMatrixMeta]) -> Robj {
        let mut names: Vec<String>  = Vec::new();
        let mut nnz_vec: Vec<i32>   = Vec::new();
        let mut q1_vec:  Vec<i32>   = Vec::new();
        let mut med_vec: Vec<i32>   = Vec::new();
        let mut q3_vec:  Vec<i32>   = Vec::new();
        let mut max_vec: Vec<i32>   = Vec::new();
        for m in metas {
            let mut per_row: Vec<u64> = m.indptr.windows(2).map(|w| w[1] - w[0]).collect();
            per_row.sort_unstable();
            let n = per_row.len();
            let q = |p: f64| if n == 0 { 0i32 } else { per_row[(p * (n - 1) as f64).round() as usize] as i32 };
            names.push(m.name.clone());
            nnz_vec.push(m.indptr.last().copied().unwrap_or(0) as i32);
            q1_vec.push(q(0.25));
            med_vec.push(q(0.5));
            q3_vec.push(q(0.75));
            max_vec.push(per_row.last().copied().unwrap_or(0) as i32);
        }
        list!(name = names, nnz = nnz_vec, nnz_q1 = q1_vec,
              nnz_med = med_vec, nnz_q3 = q3_vec, nnz_max = max_vec).into_robj()
    }
    let layer_list = sparse_stats(&layer_metas);
    let obsp_list  = sparse_stats(&obsp_metas);

    // X nnz stats from indptr (empty for dense/BPCells)
    let x_indptr = reader.x_indptr();
    let x_stats: Robj = if x_indptr.len() > 1 {
        let nnz = *x_indptr.last().unwrap() as i32;
        let mut per_row: Vec<u64> = x_indptr.windows(2).map(|w| w[1] - w[0]).collect();
        per_row.sort_unstable();
        let n = per_row.len();
        let q = |p: f64| per_row[(p * (n - 1) as f64).round() as usize] as i32;
        list!(nnz = nnz, nnz_q1 = q(0.25), nnz_med = q(0.5),
              nnz_q3 = q(0.75), nnz_max = *per_row.last().unwrap() as i32).into_robj()
    } else {
        ().into_robj()
    };

    Ok(list!(
        format    = format_name,
        n_obs     = n_obs as i32,
        n_vars    = n_vars as i32,
        x_stats   = x_stats,
        obs_cols  = obs_cols,
        var_cols  = var_cols,
        obsm_keys = obsm_keys,
        layers    = layer_list,
        uns_keys  = uns_keys,
        obsp      = obsp_list,
        varm_keys = varm_keys
    ).into_robj())
}

// ---------------------------------------------------------------------------
// Exported function: scx_write_h5ad
//
// In-memory SCE → H5AD path. The R caller materializes counts(sce) as a
// dgCMatrix (CSC of genes×cells, which is structurally identical to CSR of
// cells×genes — same indptr/indices/data) and hands the triplet over with
// obs/var column lists and a JSON blob for uns.
// ---------------------------------------------------------------------------

fn parse_dtype(dtype: &str) -> Result<DataType> {
    Ok(match dtype {
        "f32" => DataType::F32,
        "f64" => DataType::F64,
        "i32" => DataType::I32,
        "u32" => DataType::U32,
        other => return Err(Error::from(format!("unknown dtype '{other}'"))),
    })
}

/// Coerce an R column (Robj) into scx-core's ColumnData.
fn robj_to_column_data(name: &str, obj: &Robj) -> Result<ColumnData> {
    // Factor: integer vector with class "factor" + levels attr.
    if obj.is_factor() {
        let levels_obj = obj.get_attrib("levels").unwrap_or_else(|| ().into());
        let levels: Vec<String> = levels_obj
            .as_str_vector()
            .ok_or_else(|| Error::from(format!("column '{name}': factor missing levels")))?
            .into_iter()
            .map(String::from)
            .collect();
        let codes: Vec<u32> = obj
            .as_integer_vector()
            .ok_or_else(|| Error::from(format!("column '{name}': factor codes not integer")))?
            .into_iter()
            .map(|x| if x < 1 { 0u32 } else { (x - 1) as u32 })
            .collect();
        return Ok(ColumnData::Categorical { codes, levels });
    }
    if let Some(v) = obj.as_logical_slice() {
        let bools: Vec<bool> = v.iter().map(|b| b.is_true()).collect();
        return Ok(ColumnData::Bool(bools));
    }
    if let Some(v) = obj.as_integer_vector() {
        return Ok(ColumnData::Int(v));
    }
    if let Some(v) = obj.as_real_vector() {
        return Ok(ColumnData::Float(v));
    }
    if let Some(v) = obj.as_str_vector() {
        return Ok(ColumnData::String(v.into_iter().map(String::from).collect()));
    }
    Err(Error::from(format!(
        "column '{name}': unsupported R type — expected logical/integer/double/character/factor",
    )))
}

fn list_to_columns(list: &List) -> Result<Vec<Column>> {
    let mut out = Vec::with_capacity(list.len());
    for (name, obj) in list.iter() {
        let data = robj_to_column_data(name, &obj)?;
        out.push(Column {
            name: name.to_string(),
            data,
        });
    }
    Ok(out)
}

/// Write an in-memory matrix + tables to an H5AD file.
///
/// @param output    Path to output `.h5ad` file.
/// @param n_obs     Number of cells (rows of X).
/// @param n_vars    Number of genes (columns of X).
/// @param x_indptr  CSR row pointers (length `n_obs + 1`). Same layout as
///   `dgCMatrix@p` for the genes×cells counts matrix.
/// @param x_indices CSR column (gene) indices (length `nnz`, 0-based).
/// @param x_data    Numeric/integer R vector of nnz values.
/// @param obs_index Cell barcodes (length `n_obs`).
/// @param var_index Feature names (length `n_vars`).
/// @param obs_cols  Named list of obs columns (length n_obs each).
/// @param var_cols  Named list of var columns (length n_vars each).
/// @param uns_json  JSON-encoded uns object (`{}` for empty).
/// @param dtype     Output X dtype: `"f32"` | `"f64"` | `"i32"` | `"u32"`.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn scx_write_h5ad(
    output:    &str,
    n_obs:     i32,
    n_vars:    i32,
    x_indptr:  Vec<i32>,
    x_indices: Vec<i32>,
    x_data:    Robj,
    obs_index: Vec<String>,
    var_index: Vec<String>,
    obs_cols:  List,
    var_cols:  List,
    uns_json:  &str,
    dtype:     &str,
) -> Result<()> {
    let dtype = parse_dtype(dtype)?;
    let n_obs  = n_obs  as usize;
    let n_vars = n_vars as usize;

    if x_indptr.len() != n_obs + 1 {
        return Err(Error::from(format!(
            "x_indptr length {} != n_obs+1 ({})",
            x_indptr.len(),
            n_obs + 1
        )));
    }
    if x_indices.len() != *x_indptr.last().unwrap() as usize {
        return Err(Error::from(
            "x_indices length does not match indptr last entry".to_string(),
        ));
    }

    // X data: accept R numeric (REAL → f64) or integer (INTSXP → i32).
    let data: TypedVec = if let Some(v) = x_data.as_real_vector() {
        TypedVec::F64(v)
    } else if let Some(v) = x_data.as_integer_vector() {
        TypedVec::I32(v)
    } else {
        return Err(Error::from(
            "x_data must be a numeric or integer vector".to_string(),
        ));
    };
    if data.len() != x_indices.len() {
        return Err(Error::from(format!(
            "x_data length {} != x_indices length {}",
            data.len(),
            x_indices.len()
        )));
    }

    let csr = SparseMatrixCSR {
        shape: (n_obs, n_vars),
        indptr: x_indptr.iter().map(|&x| x as u64).collect(),
        indices: x_indices.iter().map(|&x| x as u32).collect(),
        data,
    };

    if obs_index.len() != n_obs {
        return Err(Error::from("obs_index length mismatch".to_string()));
    }
    if var_index.len() != n_vars {
        return Err(Error::from("var_index length mismatch".to_string()));
    }

    let obs = ObsTable {
        index: obs_index,
        columns: list_to_columns(&obs_cols)?,
    };
    let var = VarTable {
        index: var_index,
        columns: list_to_columns(&var_cols)?,
    };

    let uns_value: serde_json::Value =
        serde_json::from_str(uns_json).map_err(|e| Error::from(format!("uns JSON: {e}")))?;
    let uns = UnsTable { raw: uns_value };

    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: n_obs,
        data: csr,
    };

    let result = block_on(async {
        let mut writer = H5AdWriter::create(Path::new(output), n_obs, n_vars, dtype)?;
        writer.write_obs(&obs).await?;
        writer.write_var(&var).await?;
        writer.write_obsm(&Embeddings::default()).await?;
        writer.write_uns(&uns).await?;
        writer.write_varm(&Varm::default()).await?;
        writer.write_x_chunk(&chunk).await?;
        writer.finalize().await?;
        anyhow::Ok(())
    });

    result.map_err(|e| Error::from(e.to_string()))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod picklerick;
    fn scx_convert;
    fn scx_inspect;
    fn scx_write_h5ad;
}
