//! The steps of `scx convert`, for round-trip tests: write a reader into
//! any writer, open the result, and read X back dense.
//!
//! Shared by roundtrip.rs and roundtrip_prop.rs via `#[path]`.
#![allow(dead_code)]

use futures::StreamExt;
use scx_core::bpcells::BpcellsDirWriter;
use scx_core::dtype::DataType;
use scx_core::h5ad::H5AdWriter;
use scx_core::h5bpcells::BpcellsH5Writer;
use scx_core::h5seurat::H5SeuratWriter;
use scx_core::npy::{NpyIrWriter, SlotFilter};
use scx_core::stream::{DatasetReader, DatasetWriter};
use std::path::{Path, PathBuf};

pub async fn open(path: &Path, layer: &str, chunk: usize) -> Box<dyn DatasetReader + Send> {
    let opts = scx_core::OpenOptions {
        layer: Some(layer.to_string()),
        ..scx_core::OpenOptions::new(chunk)
    };
    scx_core::open(path.to_str().unwrap(), &opts)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

pub async fn dense_x(reader: &mut (dyn DatasetReader + Send)) -> Vec<Vec<f64>> {
    let (n_obs, n_vars) = reader.shape();
    let mut out = vec![vec![0.0; n_vars]; n_obs];
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let values = chunk.data.data.to_f64();
        for r in 0..chunk.nrows {
            for k in chunk.data.indptr[r] as usize..chunk.data.indptr[r + 1] as usize {
                out[chunk.row_offset + r][chunk.data.indices[k] as usize] = values[k];
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub enum Out {
    H5ad,
    /// SeuratDisk dgCMatrix layout (`scx convert --dgcmatrix`).
    H5SeuratDgc,
    /// BPCells-backed H5Seurat (`scx convert` default for .h5seurat).
    H5SeuratBpcells,
    BpcellsDir,
    Npy,
}

/// The same steps as `scx convert`: metadata, layers, obsp, then X. For
/// H5Seurat, X goes to the `data` slot when the source has a `counts` layer
/// (the CLI's `--x-slot auto`), and a layer with the X slot's name is skipped.
pub async fn write(
    reader: &mut (dyn DatasetReader + Send),
    out: Out,
    path: &Path,
    chunk: usize,
) -> String {
    if let Out::Npy = out {
        NpyIrWriter::stream(path, &mut *reader, &SlotFilter::all(), chunk)
            .await
            .unwrap();
        return "counts".into();
    }
    let (n, m) = reader.shape();
    let layers = reader.layer_metas().await.unwrap();
    let x_slot = if layers.iter().any(|l| l.name == "counts") {
        "data"
    } else {
        "counts"
    };
    let mut w: Box<dyn DatasetWriter> = match out {
        Out::H5ad => Box::new(H5AdWriter::create(path, n, m, DataType::F32).unwrap()),
        Out::H5SeuratDgc => Box::new(
            H5SeuratWriter::create(path, n, m, DataType::F32, None, Some(x_slot), None, false)
                .unwrap(),
        ),
        Out::H5SeuratBpcells => Box::new(
            BpcellsH5Writer::create(path, n, m, DataType::F32, None, Some(x_slot), None, false)
                .unwrap(),
        ),
        Out::BpcellsDir => Box::new(BpcellsDirWriter::create(path, n, m).unwrap()),
        Out::Npy => unreachable!(),
    };
    let seurat = matches!(out, Out::H5SeuratDgc | Out::H5SeuratBpcells);

    w.write_obs(&reader.obs().await.unwrap()).await.unwrap();
    w.write_var(&reader.var().await.unwrap()).await.unwrap();
    let obsm = reader.obsm().await.unwrap();
    if !obsm.map.is_empty() {
        w.write_obsm(&obsm).await.unwrap();
    }
    w.write_uns(&reader.uns().await.unwrap()).await.unwrap();
    let varm = reader.varm().await.unwrap();
    if !varm.map.is_empty() {
        w.write_varm(&varm).await.unwrap();
    }
    for meta in &layers {
        if seurat && meta.name == x_slot {
            continue;
        }
        w.begin_sparse("layers", &meta.name, meta).await.unwrap();
        let mut s = reader.layer_stream(meta, chunk);
        while let Some(c) = s.next().await {
            w.write_sparse_chunk(&c.unwrap()).await.unwrap();
        }
        w.end_sparse().await.unwrap();
    }
    for meta in &reader.obsp_metas().await.unwrap() {
        w.begin_sparse("obsp", &meta.name, meta).await.unwrap();
        let mut s = reader.obsp_stream(meta, chunk);
        while let Some(c) = s.next().await {
            w.write_sparse_chunk(&c.unwrap()).await.unwrap();
        }
        w.end_sparse().await.unwrap();
    }
    let mut s = reader.x_stream();
    while let Some(c) = s.next().await {
        w.write_x_chunk(&c.unwrap()).await.unwrap();
    }
    w.finalize().await.unwrap();
    if seurat {
        x_slot.into()
    } else {
        "counts".into()
    }
}

pub fn out_path(dir: &Path, out: Out) -> PathBuf {
    dir.join(match out {
        Out::H5ad => "out.h5ad",
        Out::H5SeuratDgc | Out::H5SeuratBpcells => "out.h5seurat",
        Out::BpcellsDir => "out_bpcells",
        Out::Npy => "out_npy",
    })
}
