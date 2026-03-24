//! BPCells on-disk directory-format reader.
//!
//! Supports `packed-uint-matrix-v2`, `unpacked-uint-matrix-v2`,
//! `packed-float-matrix-v2`, and `packed-double-matrix-v2` in CSC and CSR.
//!
//! Format spec: https://bnprks.github.io/BPCells/articles/web-only/bitpacking-format.html

use std::path::Path;

// ─── Zigzag codec ────────────────────────────────────────────────────────────

/// Encode a signed integer as a zigzag-encoded unsigned integer.
/// `x ≥ 0 → 2x`; `x < 0 → -2x - 1`.
pub fn zigzag_encode(x: i32) -> u32 {
    ((x << 1) ^ (x >> 31)) as u32
}

/// Decode a zigzag-encoded unsigned integer back to a signed integer.
pub fn zigzag_decode(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

// ─── BP-128 core unpack ──────────────────────────────────────────────────────

/// Unpack 128 u32 values from `4 * b` packed words.
///
/// **Layout** (matches BPCells' SIMD implementation):
/// - 4 SIMD lanes, each holding 32 values at stride-4 positions.
///   Lane `l` holds values at global positions `l, l+4, l+8, ..., l+124`.
/// - Output words are interleaved across lanes:
///   `packed[w*4 + l]` is word `w` of lane `l` (for `w` in `0..b`, `l` in `0..4`).
///
/// When `b == 0`, all 128 output values are 0 (packed must be empty).
pub fn bp128_unpack(b: u8, packed: &[u32]) -> [u32; 128] {
    let mut out = [0u32; 128];
    if b == 0 {
        debug_assert!(packed.is_empty());
        return out;
    }
    let b = b as usize;
    debug_assert_eq!(packed.len(), 4 * b);
    let mask = if b == 32 { u32::MAX } else { (1u32 << b) - 1 };
    for lane in 0..4usize {
        for j in 0..32usize {
            let bit_start = j * b;
            let word = bit_start / 32;
            let bit_off = bit_start % 32;
            let lane_word = |w: usize| packed[w * 4 + lane];
            let val = if bit_off + b <= 32 {
                (lane_word(word) >> bit_off) & mask
            } else {
                let lo = lane_word(word) >> bit_off;
                let excess = bit_off + b - 32;
                let hi = lane_word(word + 1) & ((1u32 << excess) - 1);
                lo | (hi << (b - excess))
            };
            out[lane + 4 * j] = val;
        }
    }
    out
}

// ─── Stream decoders ─────────────────────────────────────────────────────────

/// Decode `count` values from a **BP-128-D1Z** stream (used for row indices).
///
/// D1Z = BP-128 applied to delta-zigzag-encoded differences.
///
/// - `data`: the raw packed word stream (`index_data` file, body only).
/// - `chunk_offsets`: word boundary per chunk in `data`, length `n_chunks + 1`
///   (`index_idx` file, body only).
/// - `starts`: the "previous" value before each chunk (`index_starts` file).
pub fn decode_d1z(
    data: &[u32],
    chunk_offsets: &[u32],
    starts: &[u32],
    count: usize,
) -> Vec<u32> {
    let n_chunks = chunk_offsets.len().saturating_sub(1);
    let mut out = Vec::with_capacity(count);
    let mut remaining = count;
    for k in 0..n_chunks {
        if remaining == 0 {
            break;
        }
        let ws = chunk_offsets[k] as usize;
        let we = chunk_offsets[k + 1] as usize;
        let b = ((we - ws) / 4) as u8;
        let raw = bp128_unpack(b, &data[ws..we]);
        let take = remaining.min(128);
        let mut prev = starts[k];
        for i in 0..take {
            let delta = zigzag_decode(raw[i]);
            prev = (prev as i64 + delta as i64) as u32;
            out.push(prev);
        }
        remaining -= take;
    }
    out
}

/// Decode `count` values from a **BP-128-FOR** stream (used for uint32 values).
///
/// FOR = Frame Of Reference with offset 1: packed values are `v - 1`.
///
/// - `data`: the raw packed word stream (`val_data` file, body only).
/// - `chunk_offsets`: word boundary per chunk (`val_idx` file, body only).
pub fn decode_for(data: &[u32], chunk_offsets: &[u32], count: usize) -> Vec<u32> {
    let n_chunks = chunk_offsets.len().saturating_sub(1);
    let mut out = Vec::with_capacity(count);
    let mut remaining = count;
    for k in 0..n_chunks {
        if remaining == 0 {
            break;
        }
        let ws = chunk_offsets[k] as usize;
        let we = chunk_offsets[k + 1] as usize;
        let b = ((we - ws) / 4) as u8;
        let raw = bp128_unpack(b, &data[ws..we]);
        let take = remaining.min(128);
        for i in 0..take {
            out.push(raw[i].wrapping_add(1));
        }
        remaining -= take;
    }
    out
}

// ─── File I/O helpers ────────────────────────────────────────────────────────

fn read_u32s_file(path: &Path) -> std::io::Result<Vec<u32>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 4 == 0, "{path:?}");
    Ok(data[8..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_u64s_file(path: &Path) -> std::io::Result<Vec<u64>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 8 == 0, "{path:?}");
    Ok(data[8..]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_f32s_file(path: &Path) -> std::io::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 4 == 0, "{path:?}");
    Ok(data[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_f64s_file(path: &Path) -> std::io::Result<Vec<f64>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 8 == 0, "{path:?}");
    Ok(data[8..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Read a plain-text names file (one name per line, no binary header).
/// Returns an empty vec if the file is missing or empty.
fn read_names_file(path: &Path) -> std::io::Result<Vec<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(s.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e),
    }
}

// ─── BpcellsDirReader ─────────────────────────────────────────────────────────

/// Storage order of a BPCells matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageOrder {
    Col,
    Row,
}

/// Value storage for the three supported scalar types.
#[derive(Debug)]
pub enum ValStore {
    Uint32(Vec<u32>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

/// In-memory representation of a BPCells directory-format matrix.
pub struct BpcellsDirReader {
    pub nrow: usize,
    pub ncol: usize,
    pub storage_order: StorageOrder,
    /// Names for the row dimension (length = nrow, or empty).
    pub row_names: Vec<String>,
    /// Names for the column dimension (length = ncol, or empty).
    pub col_names: Vec<String>,
    /// Outer-dimension pointer array (length = outer_dim + 1).
    pub idxptr: Vec<u64>,
    /// Inner-dimension indices (row for CSC, col for CSR), length = nnz.
    pub index: Vec<u32>,
    /// Values, one per nnz.
    pub values: ValStore,
}

impl BpcellsDirReader {
    /// Open and fully decode a BPCells directory-format matrix.
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        let version = std::fs::read_to_string(dir.join("version"))?;
        let version = version.trim().to_owned();

        let order_str = std::fs::read_to_string(dir.join("storage_order"))?;
        let storage_order = match order_str.trim() {
            "col" => StorageOrder::Col,
            "row" => StorageOrder::Row,
            s => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown storage_order: {s}"),
                ))
            }
        };

        let shape = read_u32s_file(&dir.join("shape"))?;
        let (nrow, ncol) = (shape[0] as usize, shape[1] as usize);

        let idxptr = read_u64s_file(&dir.join("idxptr"))?;
        let total_nnz = *idxptr.last().unwrap_or(&0) as usize;

        let (index, values) = match version.as_str() {
            "packed-uint-matrix-v2" => {
                let idx_data = read_u32s_file(&dir.join("index_data"))?;
                let idx_idx = read_u32s_file(&dir.join("index_idx"))?;
                let idx_starts = read_u32s_file(&dir.join("index_starts"))?;
                let val_data = read_u32s_file(&dir.join("val_data"))?;
                let val_idx = read_u32s_file(&dir.join("val_idx"))?;
                let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
                let vals = decode_for(&val_data, &val_idx, total_nnz);
                (index, ValStore::Uint32(vals))
            }
            "unpacked-uint-matrix-v2" => {
                let index = read_u32s_file(&dir.join("index"))?;
                let vals = read_u32s_file(&dir.join("val"))?;
                (index, ValStore::Uint32(vals))
            }
            "packed-float-matrix-v2" => {
                let idx_data = read_u32s_file(&dir.join("index_data"))?;
                let idx_idx = read_u32s_file(&dir.join("index_idx"))?;
                let idx_starts = read_u32s_file(&dir.join("index_starts"))?;
                let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
                let vals = read_f32s_file(&dir.join("val"))?;
                (index, ValStore::Float32(vals))
            }
            "packed-double-matrix-v2" => {
                let idx_data = read_u32s_file(&dir.join("index_data"))?;
                let idx_idx = read_u32s_file(&dir.join("index_idx"))?;
                let idx_starts = read_u32s_file(&dir.join("index_starts"))?;
                let index = decode_d1z(&idx_data, &idx_idx, &idx_starts, total_nnz);
                let vals = read_f64s_file(&dir.join("val"))?;
                (index, ValStore::Float64(vals))
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unsupported BPCells version: {version}"),
                ))
            }
        };

        let row_names = read_names_file(&dir.join("row_names")).unwrap_or_default();
        let col_names = read_names_file(&dir.join("col_names")).unwrap_or_default();

        Ok(BpcellsDirReader {
            nrow,
            ncol,
            storage_order,
            row_names,
            col_names,
            idxptr,
            index,
            values,
        })
    }

    /// Reconstruct the logical matrix as a flat row-major f64 array.
    ///
    /// Element `[row, col]` is at index `row * ncol + col`.
    /// CSR matrices are transparently handled (outer = row, inner = col).
    pub fn to_dense_f64(&self) -> Vec<f64> {
        let mut dense = vec![0.0f64; self.nrow * self.ncol];
        let outer_len = self.idxptr.len().saturating_sub(1);
        for outer in 0..outer_len {
            let start = self.idxptr[outer] as usize;
            let end = self.idxptr[outer + 1] as usize;
            for ptr in start..end {
                let inner = self.index[ptr] as usize;
                let v = match &self.values {
                    ValStore::Uint32(vs) => vs[ptr] as f64,
                    ValStore::Float32(vs) => vs[ptr] as f64,
                    ValStore::Float64(vs) => vs[ptr],
                };
                let (row, col) = match self.storage_order {
                    StorageOrder::Col => (inner, outer),
                    StorageOrder::Row => (outer, inner),
                };
                dense[row * self.ncol + col] = v;
            }
        }
        dense
    }

    /// Reconstruct the logical matrix as a flat row-major u32 array.
    pub fn to_dense_u32(&self) -> Vec<u32> {
        self.to_dense_f64().into_iter().map(|v| v as u32).collect()
    }
}

// ─── BpcellsDatasetReader ─────────────────────────────────────────────────────

use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use futures::{stream, Stream};

use crate::dtype::{DataType, TypedVec};
use crate::error::Result;
use crate::ir::{
    Embeddings, Layers, MatrixChunk, Obsp, ObsTable, SparseMatrixCSR,
    UnsTable, VarTable, Varm, Varp,
};
use crate::stream::DatasetReader;

/// BPCells directory matrix exposed as a streaming `DatasetReader`.
///
/// Convention (matches Seurat v5 BPCells default):
/// - CSC (`storage_order = "col"`): outer dim = obs (cells), inner dim = vars (genes).
///   Shape `[n_vars, n_obs]` → `n_obs = ncol`, `n_vars = nrow`.
/// - CSR (`storage_order = "row"`): outer dim = obs, inner dim = vars.
///   Shape `[n_obs, n_vars]` → `n_obs = nrow`, `n_vars = ncol`.
pub struct BpcellsDatasetReader {
    pub n_obs: usize,
    pub n_vars: usize,
    chunk_size: usize,
    obs_names: Vec<String>,
    var_names: Vec<String>,
    /// obs-major pointer array, length n_obs + 1.
    idxptr: Arc<Vec<u64>>,
    /// var indices per nnz.
    index: Arc<Vec<u32>>,
    values: Arc<ValStore>,
    dtype: DataType,
}

impl BpcellsDatasetReader {
    /// Construct from already-decoded parts (used by the HDF5 backend).
    pub fn from_parts(
        n_obs: usize,
        n_vars: usize,
        chunk_size: usize,
        obs_names: Vec<String>,
        var_names: Vec<String>,
        idxptr: Vec<u64>,
        index: Vec<u32>,
        values: ValStore,
        dtype: DataType,
    ) -> Self {
        Self {
            n_obs, n_vars, chunk_size, obs_names, var_names,
            idxptr: Arc::new(idxptr),
            index: Arc::new(index),
            values: Arc::new(values),
            dtype,
        }
    }

    pub fn open(dir: &std::path::Path, chunk_size: usize) -> std::io::Result<Self> {
        let r = BpcellsDirReader::open(dir)?;

        let (n_obs, n_vars, obs_names, var_names) = match r.storage_order {
            // CSC: idxptr indexed by column = obs; index = row indices = var indices.
            StorageOrder::Col => (r.ncol, r.nrow, r.col_names, r.row_names),
            // CSR: idxptr indexed by row = obs; index = column indices = var indices.
            StorageOrder::Row => (r.nrow, r.ncol, r.row_names, r.col_names),
        };

        let dtype = match &r.values {
            ValStore::Uint32(_) => DataType::U32,
            ValStore::Float32(_) => DataType::F32,
            ValStore::Float64(_) => DataType::F64,
        };

        Ok(Self {
            n_obs,
            n_vars,
            chunk_size,
            obs_names,
            var_names,
            idxptr: Arc::new(r.idxptr),
            index: Arc::new(r.index),
            values: Arc::new(r.values),
            dtype,
        })
    }
}

#[async_trait]
impl DatasetReader for BpcellsDatasetReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        Ok(ObsTable { index: self.obs_names.clone(), columns: vec![] })
    }

    async fn var(&mut self) -> Result<VarTable> {
        Ok(VarTable { index: self.var_names.clone(), columns: vec![] })
    }

    async fn obsm(&mut self) -> Result<Embeddings> { Ok(Embeddings::default()) }
    async fn uns(&mut self) -> Result<UnsTable>    { Ok(UnsTable::default()) }
    async fn layers(&mut self) -> Result<Layers>   { Ok(Layers::default()) }
    async fn obsp(&mut self) -> Result<Obsp>       { Ok(Obsp::default()) }
    async fn varp(&mut self) -> Result<Varp>       { Ok(Varp::default()) }
    async fn varm(&mut self) -> Result<Varm>       { Ok(Varm::default()) }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let n_obs      = self.n_obs;
        let n_vars     = self.n_vars;
        let chunk_size = self.chunk_size;
        let idxptr     = Arc::clone(&self.idxptr);
        let index      = Arc::clone(&self.index);
        let values     = Arc::clone(&self.values);

        Box::pin(stream::unfold(0usize, move |obs_start| {
            let idxptr = Arc::clone(&idxptr);
            let index  = Arc::clone(&index);
            let values = Arc::clone(&values);
            async move {
                if obs_start >= n_obs { return None; }
                let obs_end  = (obs_start + chunk_size).min(n_obs);
                let n_chunk  = obs_end - obs_start;
                let nnz_start = idxptr[obs_start] as usize;
                let nnz_end   = idxptr[obs_end]   as usize;

                // Normalised CSR indptr for this chunk.
                let indptr: Vec<u64> = idxptr[obs_start..=obs_end]
                    .iter()
                    .map(|&p| p - idxptr[obs_start])
                    .collect();

                let indices: Vec<u32> = index[nnz_start..nnz_end].to_vec();

                let data = match values.as_ref() {
                    ValStore::Uint32(vs) => TypedVec::U32(vs[nnz_start..nnz_end].to_vec()),
                    ValStore::Float32(vs) => TypedVec::F32(vs[nnz_start..nnz_end].to_vec()),
                    ValStore::Float64(vs) => TypedVec::F64(vs[nnz_start..nnz_end].to_vec()),
                };

                let chunk = Ok(MatrixChunk {
                    row_offset: obs_start,
                    nrows: n_chunk,
                    data: SparseMatrixCSR {
                        shape: (n_chunk, n_vars),
                        indptr,
                        indices,
                        data,
                    },
                });
                Some((chunk, obs_end))
            }
        }))
    }
}
