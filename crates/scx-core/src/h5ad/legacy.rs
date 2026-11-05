//! The h5ad layout of anndata before 0.7, which anndata still reads (scanpy's
//! `pbmc3k_processed` dataset is served in it):
//!
//! - `obs`, `var`, `raw.var` are compound datasets, one field per column plus
//!   an `index` field, strings as fixed-length bytes;
//! - a categorical column is its integer codes, with the levels in
//!   `uns/<column>_categories`;
//! - `obsm` and `varm` are compound datasets of `(n, k)` subarray fields;
//! - sparse matrices are groups tagged `h5sparse_format` / `h5sparse_shape`
//!   (handled in `reader::read_shape` / `is_csc`);
//! - neighbor graphs are `uns/neighbors/{connectivities,distances}`, which
//!   anndata >= 0.7 moves to `obsp` on read; so does this reader.
//!
//! `raw.X` / `raw.var` at the root are not read: a warning says so.

use std::collections::HashMap;

use hdf5::types::{CompoundField, CompoundType, FloatSize, IntSize, TypeDescriptor};
use hdf5::{Dataset, File};

use crate::error::{Result, ScxError};
use crate::h5_str::{read_str_attr, read_strings};
use crate::ir::{Column, ColumnData, DenseMatrix};

/// Where pre-0.7 files keep the graphs that are obsp today.
pub(super) const NEIGHBORS: &str = "uns/neighbors";
const GRAPHS: [&str; 2] = ["connectivities", "distances"];

/// The compound dataset at `path`, if `path` is one (a pre-0.7 dataframe or
/// obsm/varm).
pub(super) fn compound(file: &File, path: &str) -> Option<(Dataset, CompoundType)> {
    let ds = file.dataset(path).ok()?;
    match ds.dtype().ok()?.to_descriptor().ok()? {
        TypeDescriptor::Compound(c) => Some((ds, c)),
        _ => None,
    }
}

/// Whether `file` is in the pre-0.7 layout: obs and var as compound datasets.
pub(super) fn is_legacy(file: &File) -> bool {
    compound(file, "obs").is_some() && compound(file, "var").is_some()
}

/// One field of every element of the compound dataset `ds`, as `ty` in
/// native layout: HDF5 converts from the stored type (endianness, widths).
fn read_field(ds: &Dataset, field: &str, ty: TypeDescriptor) -> Result<Vec<u8>> {
    let size = ty.size();
    let mem = hdf5::Datatype::from_descriptor(&TypeDescriptor::Compound(CompoundType {
        fields: vec![CompoundField {
            name: field.to_string(),
            ty,
            offset: 0,
            index: 0,
        }],
        size,
    }))?;
    let mut buf = vec![0u8; ds.size() * size];
    if buf.is_empty() {
        return Ok(buf);
    }
    // Raw hdf5-sys FFI: hold the process-global HDF5 lock (see h5_chunk.rs).
    let rc = hdf5::sync::sync(|| unsafe {
        hdf5_sys::h5d::H5Dread(
            ds.id(),
            mem.id(),
            hdf5_sys::h5s::H5S_ALL,
            hdf5_sys::h5s::H5S_ALL,
            hdf5_sys::h5p::H5P_DEFAULT,
            buf.as_mut_ptr().cast(),
        )
    });
    if rc < 0 {
        return Err(ScxError::InvalidFormat(format!(
            "{}: reading field '{field}' failed",
            ds.name()
        )));
    }
    Ok(buf)
}

fn strings(ds: &Dataset, field: &str, len: usize) -> Result<Vec<String>> {
    let raw = read_field(ds, field, TypeDescriptor::FixedAscii(len))?;
    Ok(raw
        .chunks(len.max(1))
        .map(|b| {
            let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
            String::from_utf8_lossy(&b[..end]).into_owned()
        })
        .collect())
}

fn ne<const N: usize>(raw: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
    raw.as_chunks::<N>().0.iter().copied()
}

/// A pre-0.7 dataframe: its `index` field and the other fields as columns.
pub(super) fn read_dataframe(file: &File, path: &str) -> Result<(Vec<String>, Vec<Column>)> {
    let (ds, ty) = compound(file, path)
        .ok_or_else(|| ScxError::InvalidFormat(format!("{path} is not a compound dataset")))?;
    let mut index = None;
    let mut columns = Vec::new();
    for f in &ty.fields {
        let name = f.name.as_str();
        let data = match &f.ty {
            TypeDescriptor::FixedAscii(n) | TypeDescriptor::FixedUnicode(n) => {
                let v = strings(&ds, name, *n)?;
                if name == "index" {
                    index = Some(v);
                    continue;
                }
                ColumnData::String(v)
            }
            TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {
                let cats = format!("uns/{name}_categories");
                if file.dataset(&cats).is_ok() {
                    let codes: Vec<i64> = ne::<8>(&read_field(
                        &ds,
                        name,
                        TypeDescriptor::Integer(IntSize::U8),
                    )?)
                    .map(i64::from_ne_bytes)
                    .collect();
                    let levels = read_strings(file, &cats)?;
                    let missing: Vec<bool> = codes.iter().map(|&c| c < 0).collect();
                    if let Some(&c) = codes.iter().find(|&&c| c >= levels.len() as i64) {
                        return Err(ScxError::InvalidFormat(format!(
                            "{path}/{name}: code {c} but {cats} has {} levels",
                            levels.len()
                        )));
                    }
                    let codes = codes.iter().map(|&c| c.max(0) as u32).collect();
                    columns.push(Column::with_missing(
                        name,
                        ColumnData::Categorical {
                            codes,
                            levels,
                            ordered: false,
                        },
                        missing,
                    ));
                    continue;
                }
                let wide = matches!(
                    f.ty,
                    TypeDescriptor::Integer(IntSize::U8)
                        | TypeDescriptor::Unsigned(IntSize::U4 | IntSize::U8)
                );
                if wide {
                    let raw = read_field(&ds, name, TypeDescriptor::Integer(IntSize::U8))?;
                    ColumnData::Int64(ne::<8>(&raw).map(i64::from_ne_bytes).collect())
                } else {
                    let raw = read_field(&ds, name, TypeDescriptor::Integer(IntSize::U4))?;
                    ColumnData::Int(ne::<4>(&raw).map(i32::from_ne_bytes).collect())
                }
            }
            TypeDescriptor::Float(_) => {
                let raw = read_field(&ds, name, TypeDescriptor::Float(FloatSize::U8))?;
                ColumnData::Float(ne::<8>(&raw).map(f64::from_ne_bytes).collect())
            }
            TypeDescriptor::Boolean => {
                let raw = read_field(&ds, name, TypeDescriptor::Boolean)?;
                ColumnData::Bool(raw.iter().map(|&b| b != 0).collect())
            }
            other => {
                tracing::warn!("{path}/{name}: legacy column of type {other} is not read");
                continue;
            }
        };
        columns.push(Column::with_missing(name, data, Vec::new()));
    }
    let index =
        index.ok_or_else(|| ScxError::InvalidFormat(format!("{path} has no 'index' field")))?;
    Ok((index, columns))
}

/// A pre-0.7 obsm / varm: each `(n, k)` subarray field as an (n, k) matrix.
pub(super) fn read_dense_dict(
    file: &File,
    path: &str,
) -> Result<Option<HashMap<String, DenseMatrix>>> {
    let Some((ds, ty)) = compound(file, path) else {
        return Ok(None);
    };
    let n = ds.size();
    let mut map = HashMap::new();
    for f in &ty.fields {
        let k = match &f.ty {
            TypeDescriptor::FixedArray(inner, k)
                if matches!(
                    **inner,
                    TypeDescriptor::Float(_)
                        | TypeDescriptor::Integer(_)
                        | TypeDescriptor::Unsigned(_)
                ) =>
            {
                *k
            }
            TypeDescriptor::Float(_) => 1,
            other => {
                tracing::warn!(
                    "{path}/{}: legacy field of type {other} is not read",
                    f.name
                );
                continue;
            }
        };
        let ty = TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Float(FloatSize::U8)), k);
        let ty = if k == 1 {
            TypeDescriptor::Float(FloatSize::U8)
        } else {
            ty
        };
        let raw = read_field(&ds, &f.name, ty)?;
        let data = ne::<8>(&raw).map(f64::from_ne_bytes).collect();
        map.insert(
            f.name.clone(),
            DenseMatrix {
                shape: (n, k),
                data,
            },
        );
    }
    Ok(Some(map))
}

/// The neighbor graphs a pre-0.7 file keeps in `uns/neighbors`, which are
/// obsp today.
pub(super) fn obsp_names(file: &File) -> Vec<String> {
    GRAPHS
        .iter()
        .filter(|g| {
            file.group(&format!("{NEIGHBORS}/{g}"))
                .is_ok_and(|grp| read_str_attr(&grp, "h5sparse_format").is_ok())
        })
        .map(|g| g.to_string())
        .collect()
}

/// Drop from a pre-0.7 `uns` what the reader moved elsewhere: the
/// `<column>_categories` levels (now in the categorical columns) and the
/// neighbor graphs (now obsp).
pub(super) fn trim_uns(file: &File, uns: &mut serde_json::Value) {
    let Some(obj) = uns.as_object_mut() else {
        return;
    };
    let mut cols: Vec<String> = Vec::new();
    for df in ["obs", "var"] {
        if let Some((_, ty)) = compound(file, df) {
            cols.extend(ty.fields.into_iter().map(|f| f.name));
        }
    }
    for c in cols {
        obj.remove(&format!("{c}_categories"));
    }
    if let Some(n) = obj.get_mut("neighbors").and_then(|n| n.as_object_mut()) {
        for g in GRAPHS {
            n.remove(g);
        }
    }
    if file.group("raw.X").is_ok() || file.dataset("raw.X").is_ok() {
        tracing::warn!("pre-0.7 raw.X / raw.var at the file root are not read");
    }
}
