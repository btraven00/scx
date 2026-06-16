use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use futures::StreamExt;

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{ColumnData, SingleCellDataset, SparseMatrixMeta},
    stream::DatasetReader,
};

use super::format::*;
use super::meta::*;

// ---------------------------------------------------------------------------

pub struct NpyIrWriter;

impl NpyIrWriter {
    /// Write `dataset` to `dir`, subject to `filter`.  `dir` is created if absent.
    pub fn write(dir: &Path, dataset: &SingleCellDataset, filter: &SlotFilter) -> Result<()> {
        fs::create_dir_all(dir)?;
        let (n_obs, n_vars) = dataset.x.shape;
        let x_dtype = dataset.x_dtype;
        let mut meta = Meta {
            scxd_version: "0.1".to_string(),
            n_obs,
            n_vars,
            x: None,
            obs_index: None,
            var_index: None,
            obs: Vec::new(),
            var: Vec::new(),
            obsm: HashMap::new(),
            varm: HashMap::new(),
            layers: HashMap::new(),
            obsp: HashMap::new(),
            varp: HashMap::new(),
            uns: None,
        };

        // --- X ---
        if filter.includes("X") {
            write_sparse(&x_dir(dir), &dataset.x)?;
            meta.x = Some(sparse_meta(&dataset.x, x_dtype));
        }

        // --- obs/var index ---
        if filter.includes("obs_index") {
            write_txt(&dir.join("obs_index.txt"), &dataset.obs.index)?;
            meta.obs_index = Some(IndexMeta { n: n_obs });
        }
        if filter.includes("var_index") {
            write_txt(&dir.join("var_index.txt"), &dataset.var.index)?;
            meta.var_index = Some(IndexMeta { n: n_vars });
        }

        // --- obs columns ---
        let od = obs_dir(dir);
        for col in &dataset.obs.columns {
            if filter.includes(&format!("obs:{}", col.name)) {
                fs::create_dir_all(&od)?;
                write_col(&od, col)?;
                meta.obs.push(ColumnMeta {
                    name: col.name.clone(),
                    kind: col_kind(&col.data).to_string(),
                    shape: [col.data.len()],
                    n_levels: if let ColumnData::Categorical { levels, .. } = &col.data {
                        Some(levels.len())
                    } else {
                        None
                    },
                });
            }
        }

        // --- var columns ---
        let vd = var_dir(dir);
        for col in &dataset.var.columns {
            if filter.includes(&format!("var:{}", col.name)) {
                fs::create_dir_all(&vd)?;
                write_col(&vd, col)?;
                meta.var.push(ColumnMeta {
                    name: col.name.clone(),
                    kind: col_kind(&col.data).to_string(),
                    shape: [col.data.len()],
                    n_levels: if let ColumnData::Categorical { levels, .. } = &col.data {
                        Some(levels.len())
                    } else {
                        None
                    },
                });
            }
        }

        // --- obsm ---
        let om = obsm_dir(dir);
        for (key, m) in &dataset.obsm.map {
            if filter.includes(&format!("obsm:{key}")) {
                fs::create_dir_all(&om)?;
                write_2d_f64(&om.join(format!("{key}.npy")), &m.data, m.shape)?;
                meta.obsm.insert(
                    key.clone(),
                    DenseArrayMeta {
                        shape: [m.shape.0, m.shape.1],
                        dtype: "f64".to_string(),
                    },
                );
            }
        }

        // --- varm ---
        let vm = varm_dir(dir);
        for (key, m) in &dataset.varm.map {
            if filter.includes(&format!("varm:{key}")) {
                fs::create_dir_all(&vm)?;
                write_2d_f64(&vm.join(format!("{key}.npy")), &m.data, m.shape)?;
                meta.varm.insert(
                    key.clone(),
                    DenseArrayMeta {
                        shape: [m.shape.0, m.shape.1],
                        dtype: "f64".to_string(),
                    },
                );
            }
        }

        // --- layers ---
        for (key, csr) in &dataset.layers.map {
            if filter.includes(&format!("layers:{key}")) {
                write_sparse(&layers_key_dir(dir, key), csr)?;
                meta.layers.insert(key.clone(), sparse_meta(csr, x_dtype));
            }
        }

        // --- obsp ---
        for (key, csr) in &dataset.obsp.map {
            if filter.includes(&format!("obsp:{key}")) {
                write_sparse(&obsp_key_dir(dir, key), csr)?;
                meta.obsp.insert(key.clone(), sparse_meta(csr, x_dtype));
            }
        }

        // --- varp ---
        for (key, csr) in &dataset.varp.map {
            if filter.includes(&format!("varp:{key}")) {
                write_sparse(&varp_key_dir(dir, key), csr)?;
                meta.varp.insert(key.clone(), sparse_meta(csr, x_dtype));
            }
        }

        // --- uns ---
        if filter.includes("uns") && !dataset.uns.raw.is_null() {
            write_json(&dir.join("uns.json"), &dataset.uns.raw)?;
            meta.uns = Some(true);
        }

        write_json(&dir.join("meta.json"), &meta)?;
        Ok(())
    }

    /// Stream a dataset from `reader` into an NPY snapshot directory without
    /// materialising X (or layer/obsp matrices) in memory.
    ///
    /// Memory peak is bounded by one X-chunk's nnz plus the (already-small)
    /// obs/var/obsm/varm/indptr buffers — independent of total nnz.
    ///
    /// Single-pass when the reader exposes a full `x_indptr` (h5ad, 10x HDF5,
    /// SCX H5). Falls back to two-pass when it does not (BPCells, dense X):
    /// pass 1 walks `x_stream` to build indptr, pass 2 writes data/indices.
    pub async fn stream(
        dir: &Path,
        reader: &mut dyn DatasetReader,
        filter: &SlotFilter,
        chunk_size: usize,
    ) -> Result<()> {
        fs::create_dir_all(dir)?;
        let (n_obs, n_vars) = reader.shape();
        let x_dtype = reader.dtype();

        let mut meta = Meta {
            scxd_version: "0.1".to_string(),
            n_obs,
            n_vars,
            x: None,
            obs_index: None,
            var_index: None,
            obs: Vec::new(),
            var: Vec::new(),
            obsm: HashMap::new(),
            varm: HashMap::new(),
            layers: HashMap::new(),
            obsp: HashMap::new(),
            varp: HashMap::new(),
            uns: None,
        };

        // --- small slots (loaded fully by the reader; all O(n_obs) or O(n_vars)) ---
        let obs = reader.obs().await?;
        let var = reader.var().await?;
        let obsm = reader.obsm().await?;
        let uns = reader.uns().await?;
        let varm = reader.varm().await?;
        let layer_metas = reader.layer_metas().await?;
        let obsp_metas = reader.obsp_metas().await?;

        // --- obs/var indexes and columns ---
        if filter.includes("obs_index") {
            write_txt(&dir.join("obs_index.txt"), &obs.index)?;
            meta.obs_index = Some(IndexMeta { n: n_obs });
        }
        if filter.includes("var_index") {
            write_txt(&dir.join("var_index.txt"), &var.index)?;
            meta.var_index = Some(IndexMeta { n: n_vars });
        }

        let od = obs_dir(dir);
        for col in &obs.columns {
            if filter.includes(&format!("obs:{}", col.name)) {
                fs::create_dir_all(&od)?;
                write_col(&od, col)?;
                meta.obs.push(ColumnMeta {
                    name: col.name.clone(),
                    kind: col_kind(&col.data).to_string(),
                    shape: [col.data.len()],
                    n_levels: if let ColumnData::Categorical { levels, .. } = &col.data {
                        Some(levels.len())
                    } else {
                        None
                    },
                });
            }
        }

        let vd = var_dir(dir);
        for col in &var.columns {
            if filter.includes(&format!("var:{}", col.name)) {
                fs::create_dir_all(&vd)?;
                write_col(&vd, col)?;
                meta.var.push(ColumnMeta {
                    name: col.name.clone(),
                    kind: col_kind(&col.data).to_string(),
                    shape: [col.data.len()],
                    n_levels: if let ColumnData::Categorical { levels, .. } = &col.data {
                        Some(levels.len())
                    } else {
                        None
                    },
                });
            }
        }

        // --- obsm / varm (dense, small per-cell embeddings) ---
        let om = obsm_dir(dir);
        for (key, m) in &obsm.map {
            if filter.includes(&format!("obsm:{key}")) {
                fs::create_dir_all(&om)?;
                write_2d_f64(&om.join(format!("{key}.npy")), &m.data, m.shape)?;
                meta.obsm.insert(
                    key.clone(),
                    DenseArrayMeta {
                        shape: [m.shape.0, m.shape.1],
                        dtype: "f64".to_string(),
                    },
                );
            }
        }
        let vm = varm_dir(dir);
        for (key, m) in &varm.map {
            if filter.includes(&format!("varm:{key}")) {
                fs::create_dir_all(&vm)?;
                write_2d_f64(&vm.join(format!("{key}.npy")), &m.data, m.shape)?;
                meta.varm.insert(
                    key.clone(),
                    DenseArrayMeta {
                        shape: [m.shape.0, m.shape.1],
                        dtype: "f64".to_string(),
                    },
                );
            }
        }

        // --- X (streamed) ---
        if filter.includes("X") {
            // Snapshot the reader's indptr before calling x_stream (which
            // borrows &mut self).  Empty slice means the reader doesn't
            // expose a precomputed indptr — fall back to two-pass.
            let indptr_hint: Option<Vec<u64>> = {
                let ip = reader.x_indptr();
                if ip.len() == n_obs + 1 {
                    Some(ip.to_vec())
                } else {
                    None
                }
            };

            let xd = x_dir(dir);
            let nnz = stream_csr_x(&xd, n_obs, n_vars, x_dtype, indptr_hint, reader).await?;
            meta.x = Some(SparseArrayMeta {
                shape: [n_obs, n_vars],
                nnz,
                dtype: dtype_str(x_dtype).to_string(),
            });
        }

        // --- layers (streamed; indptr is loaded eagerly by layer_metas) ---
        for lmeta in &layer_metas {
            if !filter.includes(&format!("layers:{}", lmeta.name)) {
                continue;
            }
            if lmeta.indptr.is_empty() {
                // Dense layer — not supported by the streaming path; skip.
                tracing::warn!(
                    "skipping layer '{}': dense layers are not supported by streaming snapshot",
                    lmeta.name
                );
                continue;
            }
            let ldir = layers_key_dir(dir, &lmeta.name);
            let nnz = stream_csr_layer(&ldir, lmeta, x_dtype, reader, chunk_size).await?;
            meta.layers.insert(
                lmeta.name.clone(),
                SparseArrayMeta {
                    shape: [lmeta.shape.0, lmeta.shape.1],
                    nnz,
                    dtype: dtype_str(x_dtype).to_string(),
                },
            );
        }

        // --- obsp (same pattern as layers) ---
        for ometa in &obsp_metas {
            if !filter.includes(&format!("obsp:{}", ometa.name)) {
                continue;
            }
            if ometa.indptr.is_empty() {
                tracing::warn!(
                    "skipping obsp '{}': dense obsp is not supported by streaming snapshot",
                    ometa.name
                );
                continue;
            }
            let odir = obsp_key_dir(dir, &ometa.name);
            let nnz = stream_csr_obsp(&odir, ometa, x_dtype, reader, chunk_size).await?;
            meta.obsp.insert(
                ometa.name.clone(),
                SparseArrayMeta {
                    shape: [ometa.shape.0, ometa.shape.1],
                    nnz,
                    dtype: dtype_str(x_dtype).to_string(),
                },
            );
        }

        // --- uns ---
        if filter.includes("uns") && !uns.raw.is_null() {
            write_json(&dir.join("uns.json"), &uns.raw)?;
            meta.uns = Some(true);
        }

        write_json(&dir.join("meta.json"), &meta)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Streaming sparse-write helpers
// ---------------------------------------------------------------------------

fn dtype_npy_descr(d: DataType) -> &'static str {
    match d {
        DataType::F32 => "<f4",
        DataType::F64 => "<f8",
        DataType::I32 => "<i4",
        DataType::U32 => "<u4",
    }
}

fn write_typed_bytes<W: Write>(w: &mut W, tv: &TypedVec) -> io::Result<()> {
    match tv {
        TypedVec::F32(v) => w.write_all(unsafe { as_bytes(v.as_slice()) }),
        TypedVec::F64(v) => w.write_all(unsafe { as_bytes(v.as_slice()) }),
        TypedVec::I32(v) => w.write_all(unsafe { as_bytes(v.as_slice()) }),
        TypedVec::U32(v) => w.write_all(unsafe { as_bytes(v.as_slice()) }),
    }
}

/// Open data.npy + indices.npy with sized headers for a known nnz.
/// Caller appends raw bytes per chunk, then drops the writers to flush.
fn open_sparse_writers(
    dir: &Path,
    nnz: usize,
    x_dtype: DataType,
) -> Result<(BufWriter<File>, BufWriter<File>)> {
    fs::create_dir_all(dir)?;
    let mut data_w = BufWriter::new(File::create(dir.join("data.npy"))?);
    let mut idx_w = BufWriter::new(File::create(dir.join("indices.npy"))?);
    write_npy_header(&mut data_w, dtype_npy_descr(x_dtype), &[nnz])?;
    write_npy_header(&mut idx_w, "<u4", &[nnz])?;
    Ok((data_w, idx_w))
}

/// Stream the main X matrix to `dir`.  Uses single-pass when `indptr_hint` is
/// Some; otherwise walks x_stream twice (pass 1 builds indptr, pass 2 writes).
/// Returns total nnz.
async fn stream_csr_x(
    dir: &Path,
    n_obs: usize,
    _n_vars: usize,
    x_dtype: DataType,
    indptr_hint: Option<Vec<u64>>,
    reader: &mut dyn DatasetReader,
) -> Result<usize> {
    let indptr = match indptr_hint {
        Some(ip) => ip,
        None => {
            // Pass 1: walk x_stream to accumulate the absolute indptr.
            let mut ip = Vec::with_capacity(n_obs + 1);
            ip.push(0u64);
            let mut s = reader.x_stream();
            let mut rows_seen = 0usize;
            while let Some(chunk) = s.next().await {
                let chunk = chunk?;
                let base = *ip.last().unwrap();
                for &p in &chunk.data.indptr[1..] {
                    ip.push(base + p);
                }
                rows_seen += chunk.nrows;
            }
            drop(s);
            if rows_seen != n_obs {
                return Err(ScxError::InvalidFormat(format!(
                    "X stream length mismatch: expected {n_obs} rows, got {rows_seen}"
                )));
            }
            ip
        }
    };

    let nnz = *indptr.last().unwrap_or(&0) as usize;
    fs::create_dir_all(dir)?;
    write_1d_u64(&dir.join("indptr.npy"), &indptr)?;
    let (mut data_w, mut idx_w) = open_sparse_writers(dir, nnz, x_dtype)?;

    let mut s = reader.x_stream();
    let mut written = 0usize;
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        idx_w.write_all(unsafe { as_bytes(chunk.data.indices.as_slice()) })?;
        write_typed_bytes(&mut data_w, &chunk.data.data)?;
        written += chunk.data.indices.len();
    }
    drop(s);

    if written != nnz {
        return Err(ScxError::InvalidFormat(format!(
            "X stream nnz mismatch: indptr says {nnz}, got {written}"
        )));
    }
    Ok(nnz)
}

async fn stream_csr_layer(
    dir: &Path,
    meta: &SparseMatrixMeta,
    x_dtype: DataType,
    reader: &dyn DatasetReader,
    chunk_size: usize,
) -> Result<usize> {
    let nnz = *meta.indptr.last().unwrap_or(&0) as usize;
    fs::create_dir_all(dir)?;
    write_1d_u64(&dir.join("indptr.npy"), &meta.indptr)?;
    let (mut data_w, mut idx_w) = open_sparse_writers(dir, nnz, x_dtype)?;

    let mut s = reader.layer_stream(meta, chunk_size);
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        idx_w.write_all(unsafe { as_bytes(chunk.data.indices.as_slice()) })?;
        write_typed_bytes(&mut data_w, &chunk.data.data)?;
    }
    Ok(nnz)
}

async fn stream_csr_obsp(
    dir: &Path,
    meta: &SparseMatrixMeta,
    x_dtype: DataType,
    reader: &dyn DatasetReader,
    chunk_size: usize,
) -> Result<usize> {
    let nnz = *meta.indptr.last().unwrap_or(&0) as usize;
    fs::create_dir_all(dir)?;
    write_1d_u64(&dir.join("indptr.npy"), &meta.indptr)?;
    let (mut data_w, mut idx_w) = open_sparse_writers(dir, nnz, x_dtype)?;

    let mut s = reader.obsp_stream(meta, chunk_size);
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        idx_w.write_all(unsafe { as_bytes(chunk.data.indices.as_slice()) })?;
        write_typed_bytes(&mut data_w, &chunk.data.data)?;
    }
    Ok(nnz)
}

// ---------------------------------------------------------------------------
// NpyIrReader
