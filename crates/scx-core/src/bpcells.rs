//! BPCells on-disk directory-format reader.
//!
//! Supports `packed-uint-matrix-v2`, `unpacked-uint-matrix-v2`,
//! `packed-float-matrix-v2`, and `packed-double-matrix-v2` in CSC and CSR.
//!
//! Format spec: https://bnprks.github.io/BPCells/articles/web-only/bitpacking-format.html

use rayon::prelude::*;
use std::path::Path;

/// Minimum number of BP-128 chunks before switching to parallel decode.
/// Below ~256 chunks (32 768 values) the rayon thread-pool overhead dominates.
const RAYON_CHUNK_THRESHOLD: usize = 256;

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

/// Return the minimum number of bits needed to represent `max_val`.
pub fn bits_needed(max_val: u32) -> u8 {
    if max_val == 0 {
        0
    } else {
        (32 - max_val.leading_zeros()) as u8
    }
}

// ─── BP-128 core pack / unpack ───────────────────────────────────────────────

/// Pack 128 u32 values into `4 * b` words using the BP-128 lane-interleaved
/// layout used by BPCells.
///
/// **Layout** (matches BPCells' SIMD implementation):
/// - 4 SIMD lanes, each holding 32 values at stride-4 positions.
///   Lane `l` holds values at global positions `l, l+4, l+8, ..., l+124`.
/// - Output words are interleaved across lanes:
///   `packed[w*4 + l]` is word `w` of lane `l` (for `w` in `0..b`, `l` in `0..4`).
///
/// When `b == 0`, the packed stream is empty.
pub fn bp128_pack(b: u8, values: &[u32; 128]) -> Vec<u32> {
    if b == 0 {
        return Vec::new();
    }
    let b = b as usize;
    let mask = if b == 32 { u32::MAX } else { (1u32 << b) - 1 };
    let mut out = vec![0u32; 4 * b];

    for lane in 0..4usize {
        for j in 0..32usize {
            let v = values[lane + 4 * j] & mask;
            let bit_start = j * b;
            let word = bit_start / 32;
            let bit_off = bit_start % 32;

            out[word * 4 + lane] |= v << bit_off;

            if bit_off + b > 32 {
                let excess = bit_off + b - 32;
                out[(word + 1) * 4 + lane] |= v >> (b - excess);
            }
        }
    }

    out
}

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

/// Encode a uint32 value stream using BP-128-FOR (`v - 1` before packing).
///
/// Returns `(val_data, val_idx)`, where:
/// - `val_data` is the concatenated packed words for all chunks
/// - `val_idx[k]` is the starting word offset of chunk `k`
/// - `val_idx.last()` is the total word count
pub fn encode_for(values: &[u32]) -> (Vec<u32>, Vec<u64>) {
    let n_chunks = values.len().div_ceil(128);
    let mut data = Vec::new();
    let mut idx = Vec::with_capacity(n_chunks + 1);
    idx.push(0);

    for chunk in values.chunks(128) {
        // BPCells pads a partial chunk by repeating its last value.
        let mut buf = [chunk[chunk.len() - 1].wrapping_sub(1); 128];
        for (i, &v) in chunk.iter().enumerate() {
            buf[i] = v.wrapping_sub(1);
        }
        let max_val = buf[..chunk.len()].iter().copied().max().unwrap_or(0);
        let b = bits_needed(max_val);
        let packed = bp128_pack(b, &buf);
        data.extend_from_slice(&packed);
        idx.push(data.len() as u64);
    }

    (data, idx)
}

/// Encode a sorted uint32 index stream using BP-128-D1Z.
///
/// Returns `(index_data, index_idx, index_starts)`, where:
/// - `index_data` is the concatenated packed words for all chunks
/// - `index_idx[k]` is the starting word offset of chunk `k`
/// - `index_starts[k]` is the first value of chunk `k`
pub fn encode_d1z(values: &[u32]) -> crate::error::Result<(Vec<u32>, Vec<u64>, Vec<u32>)> {
    let n_chunks = values.len().div_ceil(128);
    let mut data = Vec::new();
    let mut idx = Vec::with_capacity(n_chunks + 1);
    let mut starts = Vec::with_capacity(n_chunks);
    idx.push(0);

    for chunk in values.chunks(128) {
        // BPCells: starts[k] is the value at index 128k, so the first delta is 0.
        let mut prev = chunk[0];
        starts.push(prev);

        let mut buf = [0u32; 128];
        for (i, &value) in chunk.iter().enumerate() {
            let delta = value as i64 - prev as i64;
            let delta_i32 = i32::try_from(delta).map_err(|_| {
                crate::error::ScxError::InvalidFormat(format!(
                    "BPCells D1Z delta out of i32 range ({delta}); row indices must be \
                     sorted and within i32"
                ))
            })?;
            buf[i] = zigzag_encode(delta_i32);
            prev = value;
        }

        let max_val = buf[..chunk.len()].iter().copied().max().unwrap_or(0);
        let b = bits_needed(max_val);
        let packed = bp128_pack(b, &buf);
        data.extend_from_slice(&packed);
        idx.push(data.len() as u64);
    }

    Ok((data, idx, starts))
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
pub fn decode_d1z(data: &[u32], chunk_offsets: &[u32], starts: &[u32], count: usize) -> Vec<u32> {
    let n_chunks = chunk_offsets.len().saturating_sub(1);
    let mut out = vec![0u32; count.min(n_chunks * 128)];
    // `starts[k]` is the running-prefix value before chunk k, so each chunk
    // decodes independently into its own 128-value slice of `out`.
    let decode = |(k, dst): (usize, &mut [u32])| {
        let (ws, we) = (chunk_offsets[k] as usize, chunk_offsets[k + 1] as usize);
        let raw = bp128_unpack(((we - ws) / 4) as u8, &data[ws..we]);
        let mut prev = starts[k];
        for (d, &encoded) in dst.iter_mut().zip(raw.iter()) {
            prev = (prev as i64 + zigzag_decode(encoded) as i64) as u32;
            *d = prev;
        }
    };
    if n_chunks >= RAYON_CHUNK_THRESHOLD {
        out.par_chunks_mut(128).enumerate().for_each(decode);
    } else {
        out.chunks_mut(128).enumerate().for_each(decode);
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
    let mut out = vec![0u32; count.min(n_chunks * 128)];
    let decode = |(k, dst): (usize, &mut [u32])| {
        let (ws, we) = (chunk_offsets[k] as usize, chunk_offsets[k + 1] as usize);
        let raw = bp128_unpack(((we - ws) / 4) as u8, &data[ws..we]);
        for (d, &packed) in dst.iter_mut().zip(raw.iter()) {
            *d = packed.wrapping_add(1);
        }
    };
    if n_chunks >= RAYON_CHUNK_THRESHOLD {
        out.par_chunks_mut(128).enumerate().for_each(decode);
    } else {
        out.chunks_mut(128).enumerate().for_each(decode);
    }
    out
}

// ─── File I/O helpers ────────────────────────────────────────────────────────

fn read_u32s_file(path: &Path) -> std::io::Result<Vec<u32>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 4 == 0, "{path:?}");
    Ok(data[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect())
}

fn read_u64s_file(path: &Path) -> std::io::Result<Vec<u64>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 8 == 0, "{path:?}");
    Ok(data[8..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| u64::from_le_bytes(c))
        .collect())
}

fn read_f32s_file(path: &Path) -> std::io::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 4 == 0, "{path:?}");
    Ok(data[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect())
}

fn read_f64s_file(path: &Path) -> std::io::Result<Vec<f64>> {
    let data = std::fs::read(path)?;
    assert!(data.len() >= 8 && (data.len() - 8) % 8 == 0, "{path:?}");
    Ok(data[8..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| f64::from_le_bytes(c))
        .collect())
}

/// Read a plain-text names file (one name per line, no binary header).
/// Returns an empty vec if the file is missing or empty.
fn read_names_file(path: &Path) -> std::io::Result<Vec<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(s
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect()),
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

use async_trait::async_trait;
use futures::{stream, Stream};
use std::pin::Pin;
use std::sync::Arc;

use crate::dtype::{DataType, TypedVec};
use crate::error::Result;
use crate::ir::{
    Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta, UnsTable, VarTable, Varm,
};
use crate::stream::DatasetReader;

/// BPCells directory matrix exposed as a streaming `DatasetReader`.
///
/// Convention (matches Seurat v5 BPCells default):
/// - CSC (`storage_order = "col"`): outer dim = obs (cells), inner dim = vars (genes).
///   Shape `[n_vars, n_obs]` → `n_obs = ncol`, `n_vars = nrow`.
/// - CSR (`storage_order = "row"`): outer dim = obs, inner dim = vars.
///   Shape `[n_obs, n_vars]` → `n_obs = nrow`, `n_vars = ncol`.
///
/// When opened with `open_metadata_only`, matrix fields are empty and
/// `x_stream` / `read_chunk` will return an error.
#[derive(Clone)]
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
    /// True when opened via `open_metadata_only` — matrix fields are unpopulated.
    metadata_only: bool,
}

impl BpcellsDatasetReader {
    /// Construct from already-decoded parts (used by the HDF5 backend).
    #[allow(clippy::too_many_arguments)]
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
            n_obs,
            n_vars,
            chunk_size,
            obs_names,
            var_names,
            idxptr: Arc::new(idxptr),
            index: Arc::new(index),
            values: Arc::new(values),
            dtype,
            metadata_only: false,
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
            metadata_only: false,
        })
    }

    /// Open only shape, names, and dtype — no matrix decoding.
    ///
    /// Reads `version`, `storage_order`, `shape`, `row_names`, `col_names`.
    /// Calling `x_stream()` or `read_chunk()` on this reader returns an error.
    pub fn open_metadata_only(dir: &Path) -> std::io::Result<Self> {
        let version = std::fs::read_to_string(dir.join("version")).map(|s| s.trim().to_owned())?;

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

        let (n_obs, n_vars, obs_file, var_file) = match storage_order {
            StorageOrder::Col => (ncol, nrow, "col_names", "row_names"),
            StorageOrder::Row => (nrow, ncol, "row_names", "col_names"),
        };

        let obs_names = read_names_file(&dir.join(obs_file)).unwrap_or_default();
        let var_names = read_names_file(&dir.join(var_file)).unwrap_or_default();

        let dtype = match version.as_str() {
            "packed-uint-matrix-v2" | "unpacked-uint-matrix-v2" => DataType::U32,
            "packed-float-matrix-v2" => DataType::F32,
            "packed-double-matrix-v2" => DataType::F64,
            _ => DataType::F32,
        };

        Ok(Self {
            n_obs,
            n_vars,
            chunk_size: 0,
            obs_names,
            var_names,
            idxptr: Arc::new(Vec::new()),
            index: Arc::new(Vec::new()),
            values: Arc::new(ValStore::Uint32(Vec::new())),
            dtype,
            metadata_only: true,
        })
    }

    /// Read a single obs-major chunk directly from decoded BPCells storage.
    ///
    /// This is the low-level helper used by backend adapters that want direct
    /// chunk access without going through `x_stream()`.
    pub fn read_chunk(&self, obs_start: usize, obs_end: usize) -> Result<MatrixChunk> {
        if self.metadata_only {
            return Err(crate::error::ScxError::InvalidFormat(
                "BPCells reader opened in metadata-only mode; call open() to stream matrix data"
                    .into(),
            ));
        }
        if obs_start > obs_end || obs_end > self.n_obs {
            return Err(crate::error::ScxError::InvalidFormat(format!(
                "BPCells chunk out of bounds: {obs_start}..{obs_end} for n_obs={}",
                self.n_obs
            )));
        }

        let n_chunk = obs_end - obs_start;
        let nnz_start = self.idxptr[obs_start] as usize;
        let nnz_end = self.idxptr[obs_end] as usize;

        let indptr: Vec<u64> = self.idxptr[obs_start..=obs_end]
            .iter()
            .map(|&p| p - self.idxptr[obs_start])
            .collect();

        let indices: Vec<u32> = self.index[nnz_start..nnz_end].to_vec();

        let data = match self.values.as_ref() {
            ValStore::Uint32(vs) => TypedVec::U32(vs[nnz_start..nnz_end].to_vec()),
            ValStore::Float32(vs) => TypedVec::F32(vs[nnz_start..nnz_end].to_vec()),
            ValStore::Float64(vs) => TypedVec::F64(vs[nnz_start..nnz_end].to_vec()),
        };

        Ok(MatrixChunk {
            row_offset: obs_start,
            nrows: n_chunk,
            data: SparseMatrixCSR {
                shape: (n_chunk, self.n_vars),
                indptr,
                indices,
                data,
            },
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
        Ok(ObsTable {
            index: self.obs_names.clone(),
            columns: vec![],
        })
    }

    async fn var(&mut self) -> Result<VarTable> {
        Ok(VarTable {
            index: self.var_names.clone(),
            columns: vec![],
        })
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        Ok(Embeddings::default())
    }
    async fn uns(&mut self) -> Result<UnsTable> {
        Ok(UnsTable::default())
    }
    async fn varm(&mut self) -> Result<Varm> {
        Ok(Varm::default())
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }
    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    fn layer_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        Box::pin(stream::empty())
    }

    fn obsp_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        Box::pin(stream::empty())
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        if self.metadata_only {
            return Box::pin(stream::once(async {
                Err(crate::error::ScxError::InvalidFormat(
                    "BPCells reader opened in metadata-only mode; call open() to stream matrix data".into(),
                ))
            }));
        }
        let n_obs = self.n_obs;
        let chunk_size = self.chunk_size;
        let reader = Self {
            n_obs: self.n_obs,
            n_vars: self.n_vars,
            chunk_size: self.chunk_size,
            obs_names: self.obs_names.clone(),
            var_names: self.var_names.clone(),
            idxptr: Arc::clone(&self.idxptr),
            index: Arc::clone(&self.index),
            values: Arc::clone(&self.values),
            dtype: self.dtype,
            metadata_only: self.metadata_only,
        };

        Box::pin(stream::unfold(0usize, move |obs_start| {
            let reader = Self {
                n_obs: reader.n_obs,
                n_vars: reader.n_vars,
                chunk_size: reader.chunk_size,
                obs_names: reader.obs_names.clone(),
                var_names: reader.var_names.clone(),
                idxptr: Arc::clone(&reader.idxptr),
                index: Arc::clone(&reader.index),
                values: Arc::clone(&reader.values),
                dtype: reader.dtype,
                metadata_only: reader.metadata_only,
            };
            async move {
                if obs_start >= n_obs {
                    return None;
                }
                let obs_end = (obs_start + chunk_size).min(n_obs);
                let chunk = reader.read_chunk(obs_start, obs_end);
                Some((chunk, obs_end))
            }
        }))
    }
}

// ─── Directory-format writer ─────────────────────────────────────────────────

use crate::error::ScxError;
use crate::h5bpcells::{Arr, BpSink, BpcellsStreamEncoder};
use crate::stream::DatasetWriter;
use std::collections::HashMap;
use std::io::{BufWriter, Write};

/// BPCells directory: one file per array, each an 8-byte type magic followed
/// by little-endian values; names/version/storage_order are plain text.
pub(crate) struct DirSink {
    dir: std::path::PathBuf,
    streams: HashMap<String, BufWriter<std::fs::File>>,
}

fn arr_bytes(data: &Arr) -> (&'static [u8; 8], Vec<u8>) {
    match data {
        Arr::U32(v) => (
            b"UINT32v1",
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        Arr::U64(v) => (
            b"UINT64v1",
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        Arr::F32(v) => (
            b"FLOATSv1",
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        Arr::F64(v) => (
            b"DOUBLEv1",
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
    }
}

impl BpSink for DirSink {
    fn append(&mut self, name: &str, data: Arr) -> Result<()> {
        let (magic, bytes) = arr_bytes(&data);
        if !self.streams.contains_key(name) {
            let mut f = BufWriter::new(std::fs::File::create(self.dir.join(name))?);
            f.write_all(magic)?;
            self.streams.insert(name.to_string(), f);
        }
        self.streams.get_mut(name).unwrap().write_all(&bytes)?;
        Ok(())
    }

    fn put(&mut self, name: &str, data: Arr) -> Result<()> {
        let (magic, bytes) = arr_bytes(&data);
        std::fs::write(self.dir.join(name), [magic.as_slice(), &bytes].concat())?;
        Ok(())
    }

    fn put_strings(&mut self, name: &str, values: &[String]) -> Result<()> {
        let mut s = String::new();
        for v in values {
            s.push_str(v);
            s.push('\n');
        }
        std::fs::write(self.dir.join(name), s)?;
        Ok(())
    }

    fn put_version(&mut self, version: &str) -> Result<()> {
        std::fs::write(self.dir.join("version"), format!("{version}\n"))?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        for f in self.streams.values_mut() {
            f.flush()?;
        }
        Ok(())
    }
}

/// Writes X as a BPCells matrix directory (genes × cells, col-major — what
/// `BPCells::open_matrix_dir` + Seurat v5 expect). Bounded memory: X streams
/// through the shared [`BpcellsStreamEncoder`].
///
/// A matrix dir holds one matrix plus dimnames, so obs/var columns, obsm,
/// varm, uns, layers and obsp are dropped (layers/obsp with a warning).
pub struct BpcellsDirWriter {
    enc: Option<BpcellsStreamEncoder<DirSink>>,
    obs_names: Vec<String>,
    var_names: Vec<String>,
    skipping_sparse: bool,
}

impl BpcellsDirWriter {
    pub fn create(dir: &Path, n_obs: usize, n_vars: usize) -> Result<Self> {
        if dir.exists() {
            return Err(ScxError::InvalidFormat(format!(
                "BPCells dir output {} already exists",
                dir.display()
            )));
        }
        std::fs::create_dir_all(dir)?;
        let sink = DirSink {
            dir: dir.to_path_buf(),
            streams: HashMap::new(),
        };
        Ok(Self {
            enc: Some(BpcellsStreamEncoder::new(
                sink,
                StorageOrder::Col,
                n_vars,
                n_obs,
            )?),
            obs_names: Vec::new(),
            var_names: Vec::new(),
            skipping_sparse: false,
        })
    }
}

#[async_trait]
impl DatasetWriter for BpcellsDirWriter {
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()> {
        self.obs_names = obs.index.clone();
        Ok(())
    }
    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        self.var_names = var.index.clone();
        Ok(())
    }
    async fn write_obsm(&mut self, _: &Embeddings) -> Result<()> {
        Ok(())
    }
    async fn write_uns(&mut self, _: &UnsTable) -> Result<()> {
        Ok(())
    }
    async fn write_varm(&mut self, _: &Varm) -> Result<()> {
        Ok(())
    }
    async fn begin_sparse(&mut self, prefix: &str, name: &str, _: &SparseMatrixMeta) -> Result<()> {
        tracing::warn!("BPCells dir holds X only; dropping {prefix}/{name}");
        self.skipping_sparse = true;
        Ok(())
    }
    async fn write_sparse_chunk(&mut self, _: &MatrixChunk) -> Result<()> {
        Ok(())
    }
    async fn end_sparse(&mut self) -> Result<()> {
        self.skipping_sparse = false;
        Ok(())
    }
    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        self.enc
            .as_mut()
            .expect("write after finalize")
            .push_chunk(chunk)
    }
    async fn finalize(&mut self) -> Result<()> {
        match self.enc.take() {
            Some(enc) => enc.finalize(&self.var_names, &self.obs_names),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod dir_writer_tests {
    use super::*;

    /// Stream a small CSR matrix (in two chunks, crossing a 128-cell run
    /// boundary) through the dir writer and decode it with the dir reader.
    #[tokio::test]
    async fn dir_writer_roundtrips_through_reader() {
        let (n_obs, n_vars) = (200usize, 7usize);
        let dense = |c: usize, g: usize| ((c * 31 + g * 7) % 5) as u32; // ~20% zeros
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("m");

        let mut w = BpcellsDirWriter::create(&out, n_obs, n_vars).unwrap();
        w.write_obs(&ObsTable {
            index: (0..n_obs).map(|i| format!("c{i}")).collect(),
            ..Default::default()
        })
        .await
        .unwrap();
        w.write_var(&VarTable {
            index: (0..n_vars).map(|i| format!("g{i}")).collect(),
            ..Default::default()
        })
        .await
        .unwrap();
        for (lo, hi) in [(0, 150), (150, n_obs)] {
            let (mut indptr, mut idx, mut val) = (vec![0u64], vec![], vec![]);
            for c in lo..hi {
                for g in 0..n_vars {
                    if dense(c, g) != 0 {
                        idx.push(g as u32);
                        val.push(dense(c, g));
                    }
                }
                indptr.push(idx.len() as u64);
            }
            let chunk = MatrixChunk {
                row_offset: lo,
                nrows: hi - lo,
                data: SparseMatrixCSR {
                    shape: (hi - lo, n_vars),
                    indptr,
                    indices: idx,
                    data: TypedVec::U32(val),
                },
            };
            w.write_x_chunk(&chunk).await.unwrap();
        }
        w.finalize().await.unwrap();

        assert_eq!(
            std::fs::read_to_string(out.join("version")).unwrap(),
            "packed-uint-matrix-v2\n"
        );
        let r = BpcellsDirReader::open(&out).unwrap();
        assert_eq!(
            (r.nrow, r.ncol, r.storage_order),
            (n_vars, n_obs, StorageOrder::Col)
        );
        assert_eq!(r.col_names[199], "c199");
        assert_eq!(r.row_names[6], "g6");
        let got = r.to_dense_u32(); // row-major genes × cells
        for c in 0..n_obs {
            for g in 0..n_vars {
                assert_eq!(got[g * n_obs + c], dense(c, g), "cell {c} gene {g}");
            }
        }
    }
}
