//! NPY-backed IR snapshots.
//!
//! A snapshot is a directory of raw binary `.npy` files plus a `meta.json`
//! manifest.  The format is intentionally minimal — no compression, no
//! schema negotiation — so it can be read by any language with an NPY parser
//! and eliminates HDF5 overhead for benchmarking and debugging.
//!
//! ## File layout
//!
//! ```text
//! snapshot.scxd/
//!   meta.json               # full manifest: shapes, dtypes, slot keys
//!   obs_index.txt           # n_obs lines (cell barcodes)
//!   var_index.txt           # n_vars lines (gene names)
//!   uns.json                # unstructured metadata (absent if empty)
//!   X/
//!     data.npy              # (nnz,)      f32|f64|i32|u32
//!     indices.npy           # (nnz,)      u32
//!     indptr.npy            # (n_obs+1,)  u64
//!   obs/
//!     {col}.npy             # numeric / bool column
//!     {col}_strings.txt     # string column, one value per line
//!     {col}_codes.npy       # categorical codes (u32)
//!     {col}_levels.txt      # categorical levels, one per line
//!   var/                    # same layout as obs/
//!   obsm/
//!     {key}.npy             # (n_obs, k) f64 dense, C-contiguous
//!   varm/
//!     {key}.npy             # (n_vars, k) f64 dense, C-contiguous
//!   layers/{name}/
//!     data.npy
//!     indices.npy
//!     indptr.npy
//!   obsp/{name}/            # same layout as layers/{name}/
//!   varp/{name}/
//! ```

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self};
use serde::{Deserialize, Serialize};

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, Layers, MatrixChunk, ObsTable, Obsp,
        SingleCellDataset, SparseMatrixCSR, SparseMatrixMeta, UnsTable, VarTable, Varm, Varp,
    },
    stream::DatasetReader,
};

// ---------------------------------------------------------------------------
// Slot filter
// ---------------------------------------------------------------------------

/// Controls which slots are written by [`NpyIrWriter`].
///
/// Targets understood by [`SlotFilter::includes`]:
/// - `"X"`, `"obs_index"`, `"var_index"`, `"uns"`
/// - `"obs"` (all obs columns) or `"obs:col_name"` (one column)
/// - `"var"` / `"var:col_name"`
/// - `"obsm"` / `"obsm:key"`, `"varm"` / `"varm:key"`
/// - `"layers"` / `"layers:key"`, `"obsp"` / `"obsp:key"`, `"varp"` / `"varp:key"`
#[derive(Debug, Default)]
pub struct SlotFilter {
    /// If `Some`, only listed specifiers are included.
    pub only: Option<Vec<String>>,
    /// Specifiers to exclude (applied after `only`).
    pub exclude: Vec<String>,
}

impl SlotFilter {
    pub fn all() -> Self {
        Self {
            only: None,
            exclude: vec![],
        }
    }

    pub fn from_only(s: &str) -> Self {
        Self {
            only: Some(s.split(',').map(|x| x.trim().to_string()).collect()),
            exclude: vec![],
        }
    }

    pub fn from_exclude(s: &str) -> Self {
        Self {
            only: None,
            exclude: s.split(',').map(|x| x.trim().to_string()).collect(),
        }
    }

    pub fn includes(&self, target: &str) -> bool {
        if self.exclude.iter().any(|s| slot_matches(s, target)) {
            return false;
        }
        if let Some(only) = &self.only {
            return only.iter().any(|s| slot_matches(s, target));
        }
        true
    }
}

fn slot_matches(filter: &str, target: &str) -> bool {
    filter == target || target.starts_with(&format!("{filter}:"))
}

// ---------------------------------------------------------------------------
// Rich meta.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct SparseArrayMeta {
    shape: [usize; 2],
    nnz: usize,
    dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DenseArrayMeta {
    shape: [usize; 2],
    dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexMeta {
    n: usize,
}

/// Column metadata entry — stored as an array in meta.json to preserve order.
#[derive(Debug, Serialize, Deserialize)]
struct ColumnMeta {
    name: String,
    kind: String,
    shape: [usize; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    n_levels: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Meta {
    scxd_version: String,
    n_obs: usize,
    n_vars: usize,
    #[serde(rename = "X", skip_serializing_if = "Option::is_none")]
    x: Option<SparseArrayMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    obs_index: Option<IndexMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    var_index: Option<IndexMeta>,
    /// Ordered list — preserves IR column order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    obs: Vec<ColumnMeta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    var: Vec<ColumnMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    obsm: HashMap<String, DenseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    varm: HashMap<String, DenseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    layers: HashMap<String, SparseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    obsp: HashMap<String, SparseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    varp: HashMap<String, SparseArrayMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    uns: Option<bool>,
}

fn dtype_str(dt: DataType) -> &'static str {
    match dt {
        DataType::F32 => "f32",
        DataType::F64 => "f64",
        DataType::I32 => "i32",
        DataType::U32 => "u32",
    }
}

fn parse_dtype(s: &str) -> Result<DataType> {
    match s {
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "i32" => Ok(DataType::I32),
        "u32" => Ok(DataType::U32),
        other => Err(ScxError::InvalidFormat(format!(
            "unknown dtype in meta.json: {other}"
        ))),
    }
}

fn col_kind(data: &ColumnData) -> &'static str {
    match data {
        ColumnData::Int(_) => "int",
        ColumnData::Float(_) => "float",
        ColumnData::Bool(_) => "bool",
        ColumnData::String(_) => "string",
        ColumnData::Categorical { .. } => "categorical",
    }
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

fn x_dir(root: &Path) -> PathBuf {
    root.join("X")
}
fn obs_dir(root: &Path) -> PathBuf {
    root.join("obs")
}
fn var_dir(root: &Path) -> PathBuf {
    root.join("var")
}
fn obsm_dir(root: &Path) -> PathBuf {
    root.join("obsm")
}
fn varm_dir(root: &Path) -> PathBuf {
    root.join("varm")
}
fn layers_key_dir(root: &Path, k: &str) -> PathBuf {
    root.join("layers").join(k)
}
fn obsp_key_dir(root: &Path, k: &str) -> PathBuf {
    root.join("obsp").join(k)
}
fn varp_key_dir(root: &Path, k: &str) -> PathBuf {
    root.join("varp").join(k)
}

// ---------------------------------------------------------------------------
// NPY v1.0 header write
// ---------------------------------------------------------------------------

fn write_npy_header<W: Write>(w: &mut W, descr: &str, shape: &[usize]) -> io::Result<()> {
    let shape_str = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<_> = shape.iter().map(|x| x.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    let needed = 10 + dict.len() + 1;
    let padded = (needed + 63) / 64 * 64;
    let header_len = padded - 10;
    let n_spaces = header_len - dict.len() - 1;

    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header_len as u16).to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    for _ in 0..n_spaces {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")
}

// ---------------------------------------------------------------------------
// NPY v1.0 header read
// ---------------------------------------------------------------------------

fn read_npy_raw(path: &Path) -> Result<(String, Vec<usize>, Mmap, usize)> {
    let file = File::open(path)?;
    // SAFETY: the file is read-only; no other process modifies it during this
    // call, and we never write to the mapping.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| ScxError::InvalidFormat(format!("mmap {}: {e}", path.display())))?;

    if mmap.len() < 10 || &mmap[..6] != b"\x93NUMPY" {
        return Err(ScxError::InvalidFormat(format!(
            "not an NPY file: {}",
            path.display()
        )));
    }
    let major = mmap[6];
    if major != 1 {
        return Err(ScxError::UnsupportedVersion(format!("NPY v{major}.x")));
    }
    let header_len = u16::from_le_bytes([mmap[8], mmap[9]]) as usize;
    let body_offset = 10 + header_len;
    if mmap.len() < body_offset {
        return Err(ScxError::InvalidFormat(format!(
            "NPY header truncated: {}",
            path.display()
        )));
    }
    let header_str = std::str::from_utf8(&mmap[10..body_offset])
        .map_err(|_| ScxError::InvalidFormat("NPY header not UTF-8".into()))?
        .trim_end();

    let descr = extract_header_str(header_str, "descr")?;
    let fortran = extract_header_bool(header_str, "fortran_order")?;
    let shape = extract_header_shape(header_str)?;

    if fortran {
        return Err(ScxError::InvalidFormat(format!(
            "Fortran-order NPY not supported: {}",
            path.display()
        )));
    }
    Ok((descr, shape, mmap, body_offset))
}

fn extract_header_str(header: &str, key: &str) -> Result<String> {
    let needle = format!("'{key}'");
    let pos = header
        .find(&needle)
        .ok_or_else(|| ScxError::MissingField(format!("NPY header missing '{key}'")))?;
    let rest = header[pos + needle.len()..].trim_start();
    let rest = rest
        .strip_prefix(':')
        .ok_or_else(|| ScxError::InvalidFormat(format!("NPY header malformed at '{key}'")))?
        .trim_start();
    let q = if rest.starts_with('\'') { '\'' } else { '"' };
    let inner = rest
        .strip_prefix(q)
        .ok_or_else(|| ScxError::InvalidFormat("NPY header: missing opening quote".into()))?;
    let end = inner
        .find(q)
        .ok_or_else(|| ScxError::InvalidFormat("NPY header: unclosed string".into()))?;
    Ok(inner[..end].to_string())
}

fn extract_header_bool(header: &str, key: &str) -> Result<bool> {
    let needle = format!("'{key}'");
    let pos = header
        .find(&needle)
        .ok_or_else(|| ScxError::MissingField(format!("NPY header missing '{key}'")))?;
    let rest = header[pos + needle.len()..].trim_start();
    let rest = rest
        .strip_prefix(':')
        .ok_or_else(|| ScxError::InvalidFormat(format!("NPY header malformed at '{key}'")))?
        .trim_start();
    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        Err(ScxError::InvalidFormat(format!(
            "NPY header bad bool for '{key}'"
        )))
    }
}

fn extract_header_shape(header: &str) -> Result<Vec<usize>> {
    let pos = header
        .find("'shape'")
        .ok_or_else(|| ScxError::MissingField("NPY header missing 'shape'".into()))?;
    let rest = header[pos + 7..].trim_start();
    let rest = rest
        .strip_prefix(':')
        .ok_or_else(|| ScxError::InvalidFormat("NPY header malformed at 'shape'".into()))?
        .trim_start();
    let rest = rest
        .strip_prefix('(')
        .ok_or_else(|| ScxError::InvalidFormat("NPY shape missing '('".into()))?;
    let end = rest
        .find(')')
        .ok_or_else(|| ScxError::InvalidFormat("NPY shape missing ')'".into()))?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| ScxError::InvalidFormat(format!("NPY shape non-integer: '{s}'")))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Byte-level helpers
// ---------------------------------------------------------------------------

unsafe fn as_bytes<T>(v: &[T]) -> &[u8] {
    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>())
}

unsafe fn bytes_to_vec<T: Copy>(body: &[u8], n: usize) -> Result<Vec<T>> {
    let elem = std::mem::size_of::<T>();
    if body.len() != n * elem {
        return Err(ScxError::InvalidFormat(format!(
            "NPY body size mismatch: expected {} bytes for {n} elements, got {}",
            n * elem,
            body.len()
        )));
    }
    let mut v = vec![std::mem::zeroed::<T>(); n];
    std::ptr::copy_nonoverlapping(body.as_ptr(), v.as_mut_ptr() as *mut u8, body.len());
    Ok(v)
}

// ---------------------------------------------------------------------------
// Low-level NPY write helpers
// ---------------------------------------------------------------------------

fn npy_descr(tv: &TypedVec) -> &'static str {
    match tv {
        TypedVec::F32(_) => "<f4",
        TypedVec::F64(_) => "<f8",
        TypedVec::I32(_) => "<i4",
        TypedVec::U32(_) => "<u4",
    }
}

fn write_1d_typed(path: &Path, tv: &TypedVec) -> Result<()> {
    let n = tv.len();
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, npy_descr(tv), &[n])?;
    match tv {
        TypedVec::F32(v) => w.write_all(unsafe { as_bytes(v.as_slice()) })?,
        TypedVec::F64(v) => w.write_all(unsafe { as_bytes(v.as_slice()) })?,
        TypedVec::I32(v) => w.write_all(unsafe { as_bytes(v.as_slice()) })?,
        TypedVec::U32(v) => w.write_all(unsafe { as_bytes(v.as_slice()) })?,
    }
    Ok(())
}

fn write_1d_u32(path: &Path, data: &[u32]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<u4", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

fn write_1d_u64(path: &Path, data: &[u64]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<u8", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

fn write_1d_i32(path: &Path, data: &[i32]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<i4", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

fn write_1d_f64(path: &Path, data: &[f64]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<f8", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

fn write_1d_bool(path: &Path, data: &[bool]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "|b1", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

fn write_2d_f64(path: &Path, data: &[f64], shape: (usize, usize)) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<f8", &[shape.0, shape.1])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

fn write_txt(path: &Path, lines: &[String]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for line in lines {
        writeln!(w, "{line}")?;
    }
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut w, value)
        .map_err(|e| ScxError::InvalidFormat(format!("JSON serialization error: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Low-level NPY read helpers
// ---------------------------------------------------------------------------

fn read_1d_typed(path: &Path, dtype: DataType) -> Result<TypedVec> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    if shape.len() != 1 {
        return Err(ScxError::InvalidFormat(format!(
            "expected 1D NPY, got {}D: {}",
            shape.len(),
            path.display()
        )));
    }
    let n = shape[0];
    let body = &mmap[off..];
    match dtype {
        DataType::F32 => {
            check_descr(&descr, "<f4", path)?;
            Ok(TypedVec::F32(unsafe { bytes_to_vec::<f32>(body, n) }?))
        }
        DataType::F64 => {
            check_descr(&descr, "<f8", path)?;
            Ok(TypedVec::F64(unsafe { bytes_to_vec::<f64>(body, n) }?))
        }
        DataType::I32 => {
            check_descr(&descr, "<i4", path)?;
            Ok(TypedVec::I32(unsafe { bytes_to_vec::<i32>(body, n) }?))
        }
        DataType::U32 => {
            check_descr(&descr, "<u4", path)?;
            Ok(TypedVec::U32(unsafe { bytes_to_vec::<u32>(body, n) }?))
        }
    }
}

fn read_1d_u32(path: &Path) -> Result<Vec<u32>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<u4", path)?;
    unsafe { bytes_to_vec::<u32>(&mmap[off..], shape[0]) }
}

fn read_1d_u64(path: &Path) -> Result<Vec<u64>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<u8", path)?;
    unsafe { bytes_to_vec::<u64>(&mmap[off..], shape[0]) }
}

fn read_1d_i32(path: &Path) -> Result<Vec<i32>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<i4", path)?;
    unsafe { bytes_to_vec::<i32>(&mmap[off..], shape[0]) }
}

fn read_1d_f64(path: &Path) -> Result<Vec<f64>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<f8", path)?;
    unsafe { bytes_to_vec::<f64>(&mmap[off..], shape[0]) }
}

fn read_1d_bool(path: &Path) -> Result<Vec<bool>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "|b1", path)?;
    let n = shape[0];
    let body = &mmap[off..];
    if body.len() != n {
        return Err(ScxError::InvalidFormat(format!(
            "bool NPY body size mismatch: {}",
            path.display()
        )));
    }
    Ok(body.iter().map(|&b| b != 0).collect())
}

fn read_2d_f64(path: &Path) -> Result<DenseMatrix> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    if shape.len() != 2 {
        return Err(ScxError::InvalidFormat(format!(
            "expected 2D NPY, got {}D: {}",
            shape.len(),
            path.display()
        )));
    }
    check_descr(&descr, "<f8", path)?;
    let (nrows, ncols) = (shape[0], shape[1]);
    let data = unsafe { bytes_to_vec::<f64>(&mmap[off..], nrows * ncols) }?;
    Ok(DenseMatrix {
        shape: (nrows, ncols),
        data,
    })
}

fn read_txt(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().map(|l| l.to_string()).collect())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let content = fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(|e| {
        ScxError::InvalidFormat(format!("JSON parse error in {}: {e}", path.display()))
    })
}

fn check_1d(shape: &[usize], path: &Path) -> Result<()> {
    if shape.len() != 1 {
        return Err(ScxError::InvalidFormat(format!(
            "expected 1D NPY, got {}D: {}",
            shape.len(),
            path.display()
        )));
    }
    Ok(())
}

fn check_descr(got: &str, expected: &str, _path: &Path) -> Result<()> {
    if got != expected {
        return Err(ScxError::DtypeMismatch {
            expected: expected.to_string(),
            got: got.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Column helpers
// ---------------------------------------------------------------------------

fn write_col(col_dir: &Path, col: &Column) -> Result<()> {
    match &col.data {
        ColumnData::Int(v) => write_1d_i32(&col_dir.join(format!("{}.npy", col.name)), v)?,
        ColumnData::Float(v) => write_1d_f64(&col_dir.join(format!("{}.npy", col.name)), v)?,
        ColumnData::Bool(v) => write_1d_bool(&col_dir.join(format!("{}.npy", col.name)), v)?,
        ColumnData::String(v) => write_txt(&col_dir.join(format!("{}_strings.txt", col.name)), v)?,
        ColumnData::Categorical { codes, levels } => {
            write_1d_u32(&col_dir.join(format!("{}_codes.npy", col.name)), codes)?;
            write_txt(&col_dir.join(format!("{}_levels.txt", col.name)), levels)?;
        }
    }
    Ok(())
}

fn read_col(col_dir: &Path, name: &str, cm: &ColumnMeta) -> Result<Column> {
    let data = match cm.kind.as_str() {
        "int" => ColumnData::Int(read_1d_i32(&col_dir.join(format!("{name}.npy")))?),
        "float" => ColumnData::Float(read_1d_f64(&col_dir.join(format!("{name}.npy")))?),
        "bool" => ColumnData::Bool(read_1d_bool(&col_dir.join(format!("{name}.npy")))?),
        "string" => ColumnData::String(read_txt(&col_dir.join(format!("{name}_strings.txt")))?),
        "categorical" => {
            let codes = read_1d_u32(&col_dir.join(format!("{name}_codes.npy")))?;
            let levels = read_txt(&col_dir.join(format!("{name}_levels.txt")))?;
            ColumnData::Categorical { codes, levels }
        }
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unknown column kind '{other}' for '{name}'"
            )))
        }
    };
    Ok(Column {
        name: name.to_string(),
        data,
    })
}

fn write_sparse(dir: &Path, csr: &SparseMatrixCSR) -> Result<()> {
    fs::create_dir_all(dir)?;
    write_1d_typed(&dir.join("data.npy"), &csr.data)?;
    write_1d_u32(&dir.join("indices.npy"), &csr.indices)?;
    write_1d_u64(&dir.join("indptr.npy"), &csr.indptr)?;
    Ok(())
}

fn read_sparse(dir: &Path, shape: (usize, usize), dtype: DataType) -> Result<SparseMatrixCSR> {
    let data = read_1d_typed(&dir.join("data.npy"), dtype)?;
    let indices = read_1d_u32(&dir.join("indices.npy"))?;
    let indptr = read_1d_u64(&dir.join("indptr.npy"))?;
    Ok(SparseMatrixCSR {
        shape,
        data,
        indices,
        indptr,
    })
}

fn sparse_meta(csr: &SparseMatrixCSR, dtype: DataType) -> SparseArrayMeta {
    SparseArrayMeta {
        shape: [csr.shape.0, csr.shape.1],
        nnz: csr.indices.len(),
        dtype: dtype_str(dtype).to_string(),
    }
}

// ---------------------------------------------------------------------------
// NpyIrWriter
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
}

// ---------------------------------------------------------------------------
// NpyIrReader
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
        for (key, _dm) in &meta.obsm {
            obsm_map.insert(key.clone(), read_2d_f64(&om.join(format!("{key}.npy")))?);
        }

        // --- varm ---
        let vm = varm_dir(dir);
        let mut varm_map = HashMap::new();
        for (key, _dm) in &meta.varm {
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
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::*;
    use crate::ir::*;

    fn synthetic_dataset() -> SingleCellDataset {
        let x = SparseMatrixCSR {
            shape: (3, 4),
            indptr: vec![0, 2, 3, 5],
            indices: vec![0, 2, 1, 0, 3],
            data: TypedVec::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let obs = ObsTable {
            index: vec!["cell1".into(), "cell2".into(), "cell3".into()],
            columns: vec![
                Column {
                    name: "count".into(),
                    data: ColumnData::Int(vec![10, 20, 30]),
                },
                Column {
                    name: "score".into(),
                    data: ColumnData::Float(vec![1.1, 2.2, 3.3]),
                },
                Column {
                    name: "active".into(),
                    data: ColumnData::Bool(vec![true, false, true]),
                },
                Column {
                    name: "label".into(),
                    data: ColumnData::Categorical {
                        codes: vec![0, 1, 0],
                        levels: vec!["A".into(), "B".into()],
                    },
                },
                Column {
                    name: "notes".into(),
                    data: ColumnData::String(vec!["x".into(), "y".into(), "z".into()]),
                },
            ],
        };
        let var = VarTable {
            index: vec!["g1".into(), "g2".into(), "g3".into(), "g4".into()],
            columns: vec![Column {
                name: "highly_variable".into(),
                data: ColumnData::Bool(vec![true, false, true, false]),
            }],
        };
        let obsm = Embeddings {
            map: [(
                "X_pca".to_string(),
                DenseMatrix {
                    shape: (3, 2),
                    data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                },
            )]
            .into_iter()
            .collect(),
        };
        let varm = Varm {
            map: [(
                "PCs".to_string(),
                DenseMatrix {
                    shape: (4, 2),
                    data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                },
            )]
            .into_iter()
            .collect(),
        };
        let layers = Layers {
            map: [(
                "spliced".to_string(),
                SparseMatrixCSR {
                    shape: (3, 4),
                    indptr: vec![0, 1, 2, 3],
                    indices: vec![1, 2, 3],
                    data: TypedVec::F32(vec![9.0, 8.0, 7.0]),
                },
            )]
            .into_iter()
            .collect(),
        };
        SingleCellDataset {
            x,
            x_dtype: DataType::F32,
            obs,
            var,
            obsm,
            uns: UnsTable::default(),
            layers,
            obsp: Obsp::default(),
            varp: Varp::default(),
            varm,
        }
    }

    #[test]
    fn test_npy_header_alignment() {
        for &n in &[1usize, 100, 1_000_000, 5_000_000_000] {
            let mut buf = Vec::new();
            write_npy_header(&mut buf, "<f4", &[n]).unwrap();
            assert_eq!(buf.len() % 64, 0, "header not multiple of 64 for n={n}");
        }
        let mut buf = Vec::new();
        write_npy_header(&mut buf, "<f8", &[2638, 50]).unwrap();
        assert_eq!(buf.len() % 64, 0);
    }

    #[test]
    fn test_full_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let ds = synthetic_dataset();
        NpyIrWriter::write(dir.path(), &ds, &SlotFilter::all()).unwrap();

        // Verify nested layout
        assert!(dir.path().join("X/data.npy").exists());
        assert!(dir.path().join("X/indices.npy").exists());
        assert!(dir.path().join("X/indptr.npy").exists());
        assert!(dir.path().join("obs/count.npy").exists());
        assert!(dir.path().join("obs/label_codes.npy").exists());
        assert!(dir.path().join("obs/label_levels.txt").exists());
        assert!(dir.path().join("obs/notes_strings.txt").exists());
        assert!(dir.path().join("obsm/X_pca.npy").exists());
        assert!(dir.path().join("varm/PCs.npy").exists());
        assert!(dir.path().join("layers/spliced/data.npy").exists());

        let reader = NpyIrReader::open(dir.path(), 1000).unwrap();
        let got = reader.into_dataset();

        assert_eq!(got.x.shape, ds.x.shape);
        match (&got.x.data, &ds.x.data) {
            (TypedVec::F32(a), TypedVec::F32(b)) => assert_eq!(a, b),
            _ => panic!("dtype mismatch"),
        }
        assert_eq!(got.x.indices, ds.x.indices);
        assert_eq!(got.x.indptr, ds.x.indptr);
        assert_eq!(got.obs.index, ds.obs.index);
        assert_eq!(got.var.index, ds.var.index);
        assert_eq!(got.obs.columns.len(), ds.obs.columns.len());

        for (g, e) in got.obs.columns.iter().zip(ds.obs.columns.iter()) {
            assert_eq!(g.name, e.name);
            match (&g.data, &e.data) {
                (ColumnData::Int(a), ColumnData::Int(b)) => assert_eq!(a, b),
                (ColumnData::Float(a), ColumnData::Float(b)) => {
                    for (x, y) in a.iter().zip(b.iter()) {
                        assert!((x - y).abs() < 1e-10);
                    }
                }
                (ColumnData::Bool(a), ColumnData::Bool(b)) => assert_eq!(a, b),
                (ColumnData::String(a), ColumnData::String(b)) => assert_eq!(a, b),
                (
                    ColumnData::Categorical {
                        codes: ca,
                        levels: la,
                    },
                    ColumnData::Categorical {
                        codes: cb,
                        levels: lb,
                    },
                ) => {
                    assert_eq!(ca, cb);
                    assert_eq!(la, lb);
                }
                _ => panic!("column kind mismatch for '{}'", g.name),
            }
        }
        assert_eq!(got.obsm.map["X_pca"].data, ds.obsm.map["X_pca"].data);
        assert_eq!(got.varm.map["PCs"].data, ds.varm.map["PCs"].data);
        assert_eq!(
            got.layers.map["spliced"].indices,
            ds.layers.map["spliced"].indices
        );
    }

    #[test]
    fn test_meta_json_is_rich() {
        let dir = tempfile::tempdir().unwrap();
        let ds = synthetic_dataset();
        NpyIrWriter::write(dir.path(), &ds, &SlotFilter::all()).unwrap();
        let meta: Meta = read_json(&dir.path().join("meta.json")).unwrap();

        // X carries shape + nnz + dtype
        let xm = meta.x.as_ref().unwrap();
        assert_eq!(xm.shape, [3, 4]);
        assert_eq!(xm.nnz, 5);
        assert_eq!(xm.dtype, "f32");

        // obs columns carry kind + shape
        let count_meta = meta.obs.iter().find(|c| c.name == "count").unwrap();
        assert_eq!(count_meta.kind, "int");
        assert_eq!(count_meta.shape, [3]);

        // categorical has n_levels
        let label_meta = meta.obs.iter().find(|c| c.name == "label").unwrap();
        assert_eq!(label_meta.kind, "categorical");
        assert_eq!(label_meta.n_levels, Some(2));

        // obsm carries shape + dtype
        let pca_meta = meta.obsm.get("X_pca").unwrap();
        assert_eq!(pca_meta.shape, [3, 2]);
        assert_eq!(pca_meta.dtype, "f64");

        // layers carry shape + nnz + dtype
        let spliced_meta = meta.layers.get("spliced").unwrap();
        assert_eq!(spliced_meta.shape, [3, 4]);
        assert_eq!(spliced_meta.nnz, 3);
    }

    #[test]
    fn test_selective_only_x() {
        let dir = tempfile::tempdir().unwrap();
        let ds = synthetic_dataset();
        NpyIrWriter::write(dir.path(), &ds, &SlotFilter::from_only("X,obs_index")).unwrap();

        assert!(dir.path().join("X/data.npy").exists());
        assert!(dir.path().join("obs_index.txt").exists());
        assert!(!dir.path().join("var_index.txt").exists());
        assert!(!dir.path().join("obsm/X_pca.npy").exists());
        assert!(!dir.path().join("obs").exists());

        let reader = NpyIrReader::open(dir.path(), 5000).unwrap();
        let got = reader.into_dataset();
        assert_eq!(got.obs.index, ds.obs.index);
        assert!(got.obs.columns.is_empty());
        assert!(got.obsm.map.is_empty());
    }

    #[test]
    fn test_selective_exclude() {
        let dir = tempfile::tempdir().unwrap();
        let ds = synthetic_dataset();
        NpyIrWriter::write(
            dir.path(),
            &ds,
            &SlotFilter::from_exclude("layers,obsp,varp"),
        )
        .unwrap();

        assert!(dir.path().join("X/data.npy").exists());
        assert!(dir.path().join("obsm/X_pca.npy").exists());
        assert!(!dir.path().join("layers").exists());
    }

    #[tokio::test]
    async fn test_dataset_reader_stream() {
        use futures::StreamExt;
        let dir = tempfile::tempdir().unwrap();
        let ds = synthetic_dataset();
        NpyIrWriter::write(dir.path(), &ds, &SlotFilter::all()).unwrap();

        let mut reader = NpyIrReader::open(dir.path(), 2).unwrap();
        let mut chunks = Vec::new();
        let mut stream = reader.x_stream();
        while let Some(c) = stream.next().await {
            chunks.push(c.unwrap());
        }

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].row_offset, 0);
        assert_eq!(chunks[0].nrows, 2);
        assert_eq!(chunks[1].row_offset, 2);
        assert_eq!(chunks[1].nrows, 1);
        let total_nnz: usize = chunks.iter().map(|c| c.data.indices.len()).sum();
        assert_eq!(total_nnz, ds.x.indices.len());
    }
}
