use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self};

use crate::{
    dtype::{DataType, TypedVec},
    error::Result,
    ir::{
        Embeddings, Layers, MatrixChunk, ObsTable, Obsp, SingleCellDataset, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm, Varp,
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
            .map(|m| parse_dtype(&m.dtype))
            .transpose()?
            .unwrap_or(DataType::F32);

        let (n_obs, n_vars) = (meta.n_obs, meta.n_vars);

        // --- X ---
        let x = if let Some(ref xm) = meta.x {
            let dtype = parse_dtype(&xm.dtype)?;
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
            let dtype = parse_dtype(&lm.dtype)?;
            let shape = (lm.shape[0], lm.shape[1]);
            layers_map.insert(
                key.clone(),
                read_sparse(&layers_key_dir(dir, key), shape, dtype)?,
            );
        }

        // --- obsp ---
        let mut obsp_map = HashMap::new();
        for (key, sm) in &meta.obsp {
            let dtype = parse_dtype(&sm.dtype)?;
            let shape = (sm.shape[0], sm.shape[1]);
            obsp_map.insert(
                key.clone(),
                read_sparse(&obsp_key_dir(dir, key), shape, dtype)?,
            );
        }

        // --- varp ---
        let mut varp_map = HashMap::new();
        for (key, sm) in &meta.varp {
            let dtype = parse_dtype(&sm.dtype)?;
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

/// Stream a materialized `SparseMatrixCSR` from a `HashMap` as row-chunks.
/// Used by `NpyIrReader::layer_stream` and `obsp_stream` where the data is
/// already fully in memory.
fn npy_sparse_stream<'a>(
    map: &'a std::collections::HashMap<String, SparseMatrixCSR>,
    meta: &'a SparseMatrixMeta,
    chunk_size: usize,
) -> Pin<Box<dyn stream::Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
    let mat = match map.get(&meta.name) {
        Some(m) => m,
        None => return Box::pin(stream::empty()),
    };
    let n_rows = mat.shape.0;
    let n_cols = mat.shape.1;
    let indptr = Arc::new(mat.indptr.clone());
    let indices = Arc::new(mat.indices.clone());
    let data = Arc::new(mat.data.clone());

    Box::pin(stream::unfold(0usize, move |row_start| {
        let indptr = Arc::clone(&indptr);
        let indices = Arc::clone(&indices);
        let data = Arc::clone(&data);
        async move {
            if row_start >= n_rows {
                return None;
            }
            let row_end = (row_start + chunk_size).min(n_rows);
            let nnz_start = indptr[row_start] as usize;
            let nnz_end = indptr[row_end] as usize;
            let nrows = row_end - row_start;
            let chunk_indptr: Vec<u64> = (row_start..=row_end)
                .map(|i| indptr[i] - indptr[row_start])
                .collect();
            let chunk_indices = indices[nnz_start..nnz_end].to_vec();
            let chunk_data = match data.as_ref() {
                TypedVec::F32(v) => TypedVec::F32(v[nnz_start..nnz_end].to_vec()),
                TypedVec::F64(v) => TypedVec::F64(v[nnz_start..nnz_end].to_vec()),
                TypedVec::I32(v) => TypedVec::I32(v[nnz_start..nnz_end].to_vec()),
                TypedVec::U32(v) => TypedVec::U32(v[nnz_start..nnz_end].to_vec()),
            };
            let chunk = Ok(MatrixChunk {
                row_offset: row_start,
                nrows,
                data: SparseMatrixCSR {
                    shape: (nrows, n_cols),
                    indptr: chunk_indptr,
                    indices: chunk_indices,
                    data: chunk_data,
                },
            });
            Some((chunk, row_end))
        }
    }))
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
    ) -> Pin<Box<dyn stream::Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        npy_sparse_stream(&self.dataset.layers.map, meta, chunk_size)
    }

    fn obsp_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn stream::Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        npy_sparse_stream(&self.dataset.obsp.map, meta, chunk_size)
    }

    fn x_stream(&mut self) -> Pin<Box<dyn stream::Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let n_obs = self.dataset.x.shape.0;
        let n_vars = self.dataset.x.shape.1;
        let chunk_size = self.chunk_size;
        // Move X arrays into Arcs without cloning — avoids a full duplicate
        // of the X data in memory while the stream is live.
        let indptr = Arc::new(std::mem::take(&mut self.dataset.x.indptr));
        let indices = Arc::new(std::mem::take(&mut self.dataset.x.indices));
        let data = Arc::new(std::mem::replace(
            &mut self.dataset.x.data,
            TypedVec::F32(vec![]),
        ));

        Box::pin(stream::unfold(0usize, move |row_start| {
            let indptr = Arc::clone(&indptr);
            let indices = Arc::clone(&indices);
            let data = Arc::clone(&data);
            async move {
                if row_start >= n_obs {
                    return None;
                }
                let row_end = (row_start + chunk_size).min(n_obs);
                let nnz_start = indptr[row_start] as usize;
                let nnz_end = indptr[row_end] as usize;
                let nrows = row_end - row_start;
                let chunk_indptr: Vec<u64> = (row_start..=row_end)
                    .map(|i| indptr[i] - indptr[row_start])
                    .collect();
                let chunk_indices = indices[nnz_start..nnz_end].to_vec();
                let chunk_data = match data.as_ref() {
                    TypedVec::F32(v) => TypedVec::F32(v[nnz_start..nnz_end].to_vec()),
                    TypedVec::F64(v) => TypedVec::F64(v[nnz_start..nnz_end].to_vec()),
                    TypedVec::I32(v) => TypedVec::I32(v[nnz_start..nnz_end].to_vec()),
                    TypedVec::U32(v) => TypedVec::U32(v[nnz_start..nnz_end].to_vec()),
                };
                let chunk = MatrixChunk {
                    row_offset: row_start,
                    nrows,
                    data: SparseMatrixCSR {
                        shape: (nrows, n_vars),
                        indptr: chunk_indptr,
                        indices: chunk_indices,
                        data: chunk_data,
                    },
                };
                Some((Ok(chunk), row_end))
            }
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
