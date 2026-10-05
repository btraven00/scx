use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use futures::stream::{self};

use crate::sparse::csr_chunk;
use crate::stream::{row_chunks, ChunkStream};
use crate::{
    dtype::{DataType, TypedVec},
    error::Result,
    ir::{
        Embeddings, Layers, ObsTable, Obsp, SingleCellDataset, SparseMatrixCSR, SparseMatrixMeta,
        UnsTable, VarTable, Varm, Varp,
    },
    stream::DatasetReader,
};

use super::format::*;
use super::meta::*;

// ---------------------------------------------------------------------------

pub struct NpyIrReader {
    dataset: SingleCellDataset,
    chunk_size: usize,
}

impl NpyIrReader {
    pub fn open(dir: &Path, chunk_size: usize) -> Result<Self> {
        let meta: Meta = read_json(&dir.join("meta.json"))?;

        let x_dtype = meta
            .x
            .as_ref()
            .map(|m| m.dtype.parse::<DataType>())
            .transpose()?
            .unwrap_or(DataType::F32);

        let (n_obs, n_vars) = (meta.n_obs, meta.n_vars);

        // --- X ---
        let x = if let Some(ref xm) = meta.x {
            let dtype = xm.dtype.parse::<DataType>()?;
            read_sparse(&x_dir(dir), (n_obs, n_vars), dtype)?
        } else {
            SparseMatrixCSR {
                shape: (n_obs, n_vars),
                indptr: vec![0u64; n_obs + 1],
                indices: vec![],
                data: TypedVec::F32(vec![]),
            }
        };

        // --- obs ---
        let obs_index = if meta.obs_index.is_some() {
            read_txt(&dir.join("obs_index.txt"))?
        } else {
            (0..n_obs).map(|i| i.to_string()).collect()
        };
        let od = obs_dir(dir);
        let mut obs_columns = Vec::new();
        for cm in &meta.obs {
            obs_columns.push(read_col(&od, &cm.name, cm)?);
        }

        // --- var ---
        let var_index = if meta.var_index.is_some() {
            read_txt(&dir.join("var_index.txt"))?
        } else {
            (0..n_vars).map(|i| i.to_string()).collect()
        };
        let vd = var_dir(dir);
        let mut var_columns = Vec::new();
        for cm in &meta.var {
            var_columns.push(read_col(&vd, &cm.name, cm)?);
        }

        // --- obsm ---
        let om = obsm_dir(dir);
        let mut obsm_map = HashMap::new();
        for key in meta.obsm.keys() {
            obsm_map.insert(key.clone(), read_2d_f64(&om.join(format!("{key}.npy")))?);
        }

        // --- varm ---
        let vm = varm_dir(dir);
        let mut varm_map = HashMap::new();
        for key in meta.varm.keys() {
            varm_map.insert(key.clone(), read_2d_f64(&vm.join(format!("{key}.npy")))?);
        }

        // --- layers ---
        let mut layers_map = HashMap::new();
        for (key, lm) in &meta.layers {
            let dtype = lm.dtype.parse::<DataType>()?;
            let shape = (lm.shape[0], lm.shape[1]);
            layers_map.insert(
                key.clone(),
                read_sparse(&layers_key_dir(dir, key), shape, dtype)?,
            );
        }

        // --- obsp ---
        let mut obsp_map = HashMap::new();
        for (key, sm) in &meta.obsp {
            let dtype = sm.dtype.parse::<DataType>()?;
            let shape = (sm.shape[0], sm.shape[1]);
            obsp_map.insert(
                key.clone(),
                read_sparse(&obsp_key_dir(dir, key), shape, dtype)?,
            );
        }

        // --- varp ---
        let mut varp_map = HashMap::new();
        for (key, sm) in &meta.varp {
            let dtype = sm.dtype.parse::<DataType>()?;
            let shape = (sm.shape[0], sm.shape[1]);
            varp_map.insert(
                key.clone(),
                read_sparse(&varp_key_dir(dir, key), shape, dtype)?,
            );
        }

        // --- uns ---
        let uns = if meta.uns == Some(true) {
            let raw: serde_json::Value = read_json(&dir.join("uns.json"))?;
            UnsTable { raw }
        } else {
            UnsTable::default()
        };

        let dataset = SingleCellDataset {
            x,
            x_dtype,
            obs: ObsTable {
                index: obs_index,
                columns: obs_columns,
            },
            var: VarTable {
                index: var_index,
                columns: var_columns,
            },
            obsm: Embeddings { map: obsm_map },
            uns,
            layers: Layers { map: layers_map },
            obsp: Obsp { map: obsp_map },
            varp: Varp { map: varp_map },
            varm: Varm { map: varm_map },
        };
        Ok(Self {
            dataset,
            chunk_size,
        })
    }

    pub fn into_dataset(self) -> SingleCellDataset {
        self.dataset
    }
}

// ---------------------------------------------------------------------------
// DatasetReader for NpyIrReader
// ---------------------------------------------------------------------------

/// Rows of a matrix held in memory, as chunks. The npy reader materialises
/// every matrix at open, so X, layers and obsp all stream this way.
fn in_memory_rows<'a>(mat: &'a SparseMatrixCSR, chunk_size: usize) -> ChunkStream<'a> {
    row_chunks(mat.shape.0, chunk_size, move |rows| {
        let nnz = mat.indptr[rows.start] as usize..mat.indptr[rows.end] as usize;
        let indices = mat.indices[nnz.clone()].to_vec();
        let data = mat.data.slice(nnz);
        Ok(csr_chunk(&mat.indptr, rows, mat.shape.1, indices, data))
    })
}

/// A layer or obsp matrix by name; an unknown name is an error.
fn named_rows<'a>(
    map: &'a std::collections::HashMap<String, SparseMatrixCSR>,
    meta: &'a SparseMatrixMeta,
    chunk_size: usize,
) -> ChunkStream<'a> {
    match map.get(&meta.name) {
        Some(mat) => in_memory_rows(mat, chunk_size),
        None => {
            let msg = format!("no matrix named '{}'", meta.name);
            Box::pin(stream::once(async move {
                Err(crate::error::ScxError::InvalidFormat(msg))
            }))
        }
    }
}

#[async_trait]
impl DatasetReader for NpyIrReader {
    fn shape(&self) -> (usize, usize) {
        self.dataset.x.shape
    }
    fn dtype(&self) -> DataType {
        self.dataset.x_dtype
    }
    fn x_indptr(&self) -> &[u64] {
        &self.dataset.x.indptr
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        Ok(self.dataset.obs.clone())
    }
    async fn var(&mut self) -> Result<VarTable> {
        Ok(self.dataset.var.clone())
    }
    async fn obsm(&mut self) -> Result<Embeddings> {
        Ok(self.dataset.obsm.clone())
    }
    async fn uns(&mut self) -> Result<UnsTable> {
        Ok(self.dataset.uns.clone())
    }
    async fn varm(&mut self) -> Result<Varm> {
        Ok(self.dataset.varm.clone())
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(self
            .dataset
            .layers
            .map
            .iter()
            .map(|(name, mat)| SparseMatrixMeta {
                name: name.clone(),
                shape: mat.shape,
                indptr: mat.indptr.clone(),
            })
            .collect())
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(self
            .dataset
            .obsp
            .map
            .iter()
            .map(|(name, mat)| SparseMatrixMeta {
                name: name.clone(),
                shape: mat.shape,
                indptr: mat.indptr.clone(),
            })
            .collect())
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> ChunkStream<'a> {
        named_rows(&self.dataset.layers.map, meta, chunk_size)
    }

    fn obsp_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk_size: usize) -> ChunkStream<'a> {
        named_rows(&self.dataset.obsp.map, meta, chunk_size)
    }

    fn x_stream(&mut self) -> ChunkStream<'_> {
        in_memory_rows(&self.dataset.x, self.chunk_size)
    }
}

// ---------------------------------------------------------------------------
// Tests
