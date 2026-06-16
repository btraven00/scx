use memmap2::Mmap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{Column, ColumnData, DenseMatrix, SparseMatrixCSR},
};

use super::meta::*;

// ---------------------------------------------------------------------------
// NPY v1.0 header write
// ---------------------------------------------------------------------------

pub(super) fn write_npy_header<W: Write>(
    w: &mut W,
    descr: &str,
    shape: &[usize],
) -> io::Result<()> {
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
    let padded = needed.div_ceil(64) * 64;
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

pub(super) fn read_npy_raw(path: &Path) -> Result<(String, Vec<usize>, Mmap, usize)> {
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

pub(super) fn extract_header_str(header: &str, key: &str) -> Result<String> {
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

pub(super) fn extract_header_bool(header: &str, key: &str) -> Result<bool> {
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

pub(super) fn extract_header_shape(header: &str) -> Result<Vec<usize>> {
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

pub(super) unsafe fn as_bytes<T>(v: &[T]) -> &[u8] {
    std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
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

pub(super) fn npy_descr(tv: &TypedVec) -> &'static str {
    match tv {
        TypedVec::F32(_) => "<f4",
        TypedVec::F64(_) => "<f8",
        TypedVec::I32(_) => "<i4",
        TypedVec::U32(_) => "<u4",
    }
}

pub(super) fn write_1d_typed(path: &Path, tv: &TypedVec) -> Result<()> {
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

pub(super) fn write_1d_u32(path: &Path, data: &[u32]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<u4", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

pub(super) fn write_1d_u64(path: &Path, data: &[u64]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<u8", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

pub(super) fn write_1d_i32(path: &Path, data: &[i32]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<i4", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

pub(super) fn write_1d_f64(path: &Path, data: &[f64]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<f8", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

pub(super) fn write_1d_bool(path: &Path, data: &[bool]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "|b1", &[data.len()])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

pub(super) fn write_2d_f64(path: &Path, data: &[f64], shape: (usize, usize)) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_npy_header(&mut w, "<f8", &[shape.0, shape.1])?;
    w.write_all(unsafe { as_bytes(data) })?;
    Ok(())
}

pub(super) fn write_txt(path: &Path, lines: &[String]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for line in lines {
        writeln!(w, "{line}")?;
    }
    Ok(())
}

pub(super) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut w, value)
        .map_err(|e| ScxError::InvalidFormat(format!("JSON serialization error: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Low-level NPY read helpers
// ---------------------------------------------------------------------------

pub(super) fn read_1d_typed(path: &Path, dtype: DataType) -> Result<TypedVec> {
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

pub(super) fn read_1d_u32(path: &Path) -> Result<Vec<u32>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<u4", path)?;
    unsafe { bytes_to_vec::<u32>(&mmap[off..], shape[0]) }
}

pub(super) fn read_1d_u64(path: &Path) -> Result<Vec<u64>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<u8", path)?;
    unsafe { bytes_to_vec::<u64>(&mmap[off..], shape[0]) }
}

pub(super) fn read_1d_i32(path: &Path) -> Result<Vec<i32>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<i4", path)?;
    unsafe { bytes_to_vec::<i32>(&mmap[off..], shape[0]) }
}

pub(super) fn read_1d_f64(path: &Path) -> Result<Vec<f64>> {
    let (descr, shape, mmap, off) = read_npy_raw(path)?;
    check_1d(&shape, path)?;
    check_descr(&descr, "<f8", path)?;
    unsafe { bytes_to_vec::<f64>(&mmap[off..], shape[0]) }
}

pub(super) fn read_1d_bool(path: &Path) -> Result<Vec<bool>> {
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

pub(super) fn read_2d_f64(path: &Path) -> Result<DenseMatrix> {
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

pub(super) fn read_txt(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().map(|l| l.to_string()).collect())
}

pub(super) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let content = fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(|e| {
        ScxError::InvalidFormat(format!("JSON parse error in {}: {e}", path.display()))
    })
}

pub(super) fn check_1d(shape: &[usize], path: &Path) -> Result<()> {
    if shape.len() != 1 {
        return Err(ScxError::InvalidFormat(format!(
            "expected 1D NPY, got {}D: {}",
            shape.len(),
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn check_descr(got: &str, expected: &str, _path: &Path) -> Result<()> {
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

pub(super) fn write_col(col_dir: &Path, col: &Column) -> Result<()> {
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

pub(super) fn read_col(col_dir: &Path, name: &str, cm: &ColumnMeta) -> Result<Column> {
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

pub(super) fn write_sparse(dir: &Path, csr: &SparseMatrixCSR) -> Result<()> {
    fs::create_dir_all(dir)?;
    write_1d_typed(&dir.join("data.npy"), &csr.data)?;
    write_1d_u32(&dir.join("indices.npy"), &csr.indices)?;
    write_1d_u64(&dir.join("indptr.npy"), &csr.indptr)?;
    Ok(())
}

pub(super) fn read_sparse(
    dir: &Path,
    shape: (usize, usize),
    dtype: DataType,
) -> Result<SparseMatrixCSR> {
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

pub(super) fn sparse_meta(csr: &SparseMatrixCSR, dtype: DataType) -> SparseArrayMeta {
    SparseArrayMeta {
        shape: [csr.shape.0, csr.shape.1],
        nnz: csr.indices.len(),
        dtype: dtype_str(dtype).to_string(),
    }
}

// ---------------------------------------------------------------------------
// NpyIrWriter
