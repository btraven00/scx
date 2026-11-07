//! Unstructured metadata (AnnData `uns`, Seurat `misc`) between JSON and HDF5.
//!
//! Objects are groups; scalars and 1-D arrays of one type are datasets.
//! Writing for AnnData also sets the `encoding-type` attributes anndata
//! expects; Seurat's `misc` has none.

use hdf5::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{Group, H5Type, Location};
use serde_json::Value;
use std::str::FromStr;

use crate::error::{Result, ScxError};

/// Which attribute convention to write.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Encoding {
    AnnData,
    None,
}

fn vlen(s: &str) -> Result<VarLenUnicode> {
    VarLenUnicode::from_str(s)
        .map_err(|e| ScxError::InvalidFormat(format!("invalid string {s:?}: {e}")))
}

fn encode(loc: &Location, enc: Encoding, kind: &str, version: &str) -> Result<()> {
    if enc == Encoding::AnnData {
        crate::h5_str::write_str_attr(loc, "encoding-type", kind)?;
        crate::h5_str::write_str_attr(loc, "encoding-version", version)?;
    }
    Ok(())
}

fn scalar<T: H5Type>(parent: &Group, name: &str, v: T, enc: Encoding, kind: &str) -> Result<()> {
    let ds = parent.new_dataset::<T>().shape(()).create(name)?;
    ds.write_scalar(&v)?;
    encode(&ds, enc, kind, "0.2.0")
}

fn array<T: H5Type>(
    parent: &Group,
    name: &str,
    v: Vec<T>,
    enc: Encoding,
    kind: &str,
) -> Result<()> {
    array_nd(parent, name, &[v.len()], v, enc, kind)
}

/// `v` (row-major) as a dataset of `shape`.
fn array_nd<T: H5Type>(
    parent: &Group,
    name: &str,
    shape: &[usize],
    v: Vec<T>,
    enc: Encoding,
    kind: &str,
) -> Result<()> {
    let ds = parent.new_dataset::<T>().shape(shape).create(name)?;
    ds.write_raw(&v)?;
    encode(&ds, enc, kind, "0.2.0")
}

/// Values of a numeric JSON array: integers, or floats once any value is
/// fractional, large or NaN.
pub(crate) enum Num {
    I64(Vec<i64>),
    F64(Vec<f64>),
}

/// A JSON array as an N-d numeric array: its shape and row-major values. A
/// nested array is a dataset read row by row (see [`nest`]); `null` is NaN,
/// since JSON has none. `None` unless every leaf is a number or `null` and
/// every row has the same length.
pub(crate) fn numeric_array(a: &[Value]) -> Option<(Vec<usize>, Num)> {
    fn leaves<'a>(a: &'a [Value], shape: &[usize], out: &mut Vec<&'a Value>) -> bool {
        if a.len() != shape[0] {
            return false;
        }
        if shape.len() == 1 {
            out.extend(a);
            return a.iter().all(|v| v.is_number() || v.is_null());
        }
        a.iter()
            .all(|v| matches!(v, Value::Array(row) if leaves(row, &shape[1..], out)))
    }
    let mut shape = vec![a.len()];
    let mut first = a.first();
    while let Some(Value::Array(row)) = first {
        shape.push(row.len());
        first = row.first();
    }
    let mut out = Vec::new();
    if !leaves(a, &shape, &mut out) {
        return None;
    }
    let num = if !out.is_empty() && out.iter().all(|v| v.as_i64().is_some()) {
        Num::I64(out.iter().filter_map(|v| v.as_i64()).collect())
    } else {
        Num::F64(out.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
    };
    Some((shape, num))
}

/// Flat row-major `values` of an N-d dataset of `shape` as nested rows, so
/// a matrix keeps its shape through JSON.
pub(crate) fn nest(values: Vec<Value>, shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::Array(values);
    }
    let row: usize = shape[1..].iter().product();
    if row == 0 {
        return Value::Array(vec![nest(Vec::new(), &shape[1..]); shape[0]]);
    }
    let mut it = values.into_iter();
    Value::Array(
        (0..shape[0])
            .map(|_| nest(it.by_ref().take(row).collect(), &shape[1..]))
            .collect(),
    )
}

/// Write `value` as `parent/name`. `null` has no HDF5 form and is skipped.
/// An array that isn't all-bool, all-number or all-string (mixed, or holding
/// objects) is written as a group keyed "0", "1", ... — the closest AnnData
/// has to a list — and so reads back as an object.
pub(crate) fn write_json(parent: &Group, name: &str, value: &Value, enc: Encoding) -> Result<()> {
    if let Value::Array(a) = value {
        if let Some((shape, num)) = numeric_array(a) {
            return match num {
                Num::I64(v) => array_nd(parent, name, &shape, v, enc, "array"),
                Num::F64(v) => array_nd(parent, name, &shape, v, enc, "array"),
            };
        }
    }
    match value {
        // The readers skip what they can't carry, so null is a NaN.
        Value::Null => scalar(parent, name, f64::NAN, enc, "numeric-scalar"),
        Value::Bool(b) => scalar(parent, name, *b, enc, "numeric-scalar"),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => scalar(parent, name, i, enc, "numeric-scalar"),
            (None, Some(u)) => scalar(parent, name, u, enc, "numeric-scalar"),
            _ => scalar(
                parent,
                name,
                n.as_f64().unwrap_or(f64::NAN),
                enc,
                "numeric-scalar",
            ),
        },
        Value::String(s) => scalar(parent, name, vlen(s)?, enc, "string"),
        Value::Array(a) if a.iter().all(Value::is_boolean) && !a.is_empty() => array(
            parent,
            name,
            a.iter().map(|v| v.as_bool() == Some(true)).collect(),
            enc,
            "array",
        ),
        Value::Array(a) if a.iter().all(Value::is_string) => {
            let v = a
                .iter()
                .map(|s| vlen(s.as_str().unwrap_or_default()))
                .collect::<Result<_>>()?;
            array::<VarLenUnicode>(parent, name, v, enc, "string-array")
        }
        Value::Array(a) => {
            let grp = crate::h5::create_group(parent, name)?;
            encode(&grp, enc, "dict", "0.1.0")?;
            for (i, v) in a.iter().enumerate() {
                write_json(&grp, &i.to_string(), v, enc)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            let grp = crate::h5::create_group(parent, name)?;
            encode(&grp, enc, "dict", "0.1.0")?;
            for (k, v) in map {
                write_json(&grp, k, v, enc)?;
            }
            Ok(())
        }
    }
}

/// Read the group at `path` as a JSON object. A dataset of a type with no
/// JSON form (opaque, ...) is skipped with a warning; a NaN reads as `null`.
pub(crate) fn read_json(file: &Group, path: &str) -> Result<Value> {
    let grp = file.group(path)?;
    let mut map = serde_json::Map::new();
    for name in grp.member_names()? {
        let child = format!("{path}/{name}");
        let value = if file.dataset(&child).is_ok() {
            dataset_json(file, &child)?
        } else {
            Some(read_json(file, &child)?)
        };
        if let Some(value) = value {
            map.insert(name, value);
        }
    }
    Ok(Value::Object(map))
}

fn dataset_json(file: &Group, path: &str) -> Result<Option<Value>> {
    let ds = file.dataset(path)?;
    let scalar = ds.ndim() == 0;
    fn vec_or_scalar<T: serde::Serialize>(scalar: bool, v: Vec<T>) -> Value {
        if scalar {
            v.into_iter()
                .next()
                .map_or(Value::Null, |x| serde_json::json!(x))
        } else {
            serde_json::json!(v)
        }
    }
    let v = match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Boolean => vec_or_scalar(scalar, ds.read_raw::<bool>()?),
        TypeDescriptor::Integer(_) => vec_or_scalar(scalar, ds.read_raw::<i64>()?),
        TypeDescriptor::Unsigned(_) => vec_or_scalar(scalar, ds.read_raw::<u64>()?),
        TypeDescriptor::Float(_) => vec_or_scalar(scalar, ds.read_raw::<f64>()?),
        TypeDescriptor::VarLenUnicode => {
            let v: Vec<String> = ds
                .read_raw::<VarLenUnicode>()?
                .iter()
                .map(|s| s.to_string())
                .collect();
            // A one-element string array is read as a string, as writers that
            // predate scalar strings (SeuratDisk among them) store them so.
            vec_or_scalar(scalar || v.len() == 1, v)
        }
        TypeDescriptor::VarLenAscii => {
            let v: Vec<String> = ds
                .read_raw::<VarLenAscii>()?
                .iter()
                .map(|s| s.to_string())
                .collect();
            vec_or_scalar(scalar || v.len() == 1, v)
        }
        TypeDescriptor::FixedAscii(_) | TypeDescriptor::FixedUnicode(_) => {
            let v = crate::h5_str::read_str_1d(&ds)?;
            vec_or_scalar(scalar || v.len() == 1, v)
        }
        TypeDescriptor::Compound(ct) if ds.ndim() == 1 => rec_array_json(&ds, &ct)?,
        other => {
            let enc = crate::h5_str::read_str_attr(&ds, "encoding-type").unwrap_or_default();
            tracing::warn!("{path}: {enc} of type {other} has no JSON form; it is not carried");
            return Ok(None);
        }
    };
    // A matrix (or higher) keeps its shape as nested rows.
    Ok(Some(match v {
        Value::Array(flat) if ds.ndim() >= 2 => nest(flat, &ds.shape()),
        v => v,
    }))
}

/// A 1-D compound dataset (an anndata rec-array, e.g. scanpy's
/// rank_genes_groups `names`/`scores`) as `{field: [values]}`. It is written
/// back as a dict of arrays: same values, but a dict rather than a rec-array.
// TODO: keep the rec-array layout on h5ad output if scanpy code that reads
// `.dtype.names` (rank_genes_groups plotting) needs it.
fn rec_array_json(ds: &hdf5::Dataset, ct: &hdf5::types::CompoundType) -> Result<Value> {
    use crate::h5::{read_field, read_vlen_str_field};
    use hdf5::types::{FloatSize, IntSize};
    let mut map = serde_json::Map::new();
    for f in &ct.fields {
        let name = f.name.as_str();
        let v = match &f.ty {
            TypeDescriptor::Float(_) => {
                let raw = read_field(ds, name, TypeDescriptor::Float(FloatSize::U8))?;
                serde_json::json!(raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_ne_bytes(*b))
                    .collect::<Vec<_>>())
            }
            TypeDescriptor::Boolean => {
                let raw = read_field(ds, name, TypeDescriptor::Boolean)?;
                serde_json::json!(raw.iter().map(|&b| b != 0).collect::<Vec<_>>())
            }
            TypeDescriptor::Integer(_) => {
                let raw = read_field(ds, name, TypeDescriptor::Integer(IntSize::U8))?;
                serde_json::json!(raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_ne_bytes(*b))
                    .collect::<Vec<_>>())
            }
            TypeDescriptor::Unsigned(_) => {
                let raw = read_field(ds, name, TypeDescriptor::Unsigned(IntSize::U8))?;
                serde_json::json!(raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_ne_bytes(*b))
                    .collect::<Vec<_>>())
            }
            TypeDescriptor::VarLenUnicode => {
                serde_json::json!(read_vlen_str_field(ds, name, false)?)
            }
            TypeDescriptor::VarLenAscii => serde_json::json!(read_vlen_str_field(ds, name, true)?),
            TypeDescriptor::FixedAscii(n) | TypeDescriptor::FixedUnicode(n) => {
                let raw = read_field(ds, name, TypeDescriptor::FixedAscii(*n))?;
                let v: Vec<String> = raw
                    .chunks((*n).max(1))
                    .map(|b| {
                        let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
                        String::from_utf8_lossy(&b[..end]).into_owned()
                    })
                    .collect();
                serde_json::json!(v)
            }
            other => {
                tracing::warn!(
                    "{}: rec-array field '{name}' of type {other} is not carried",
                    ds.name()
                );
                continue;
            }
        };
        map.insert(name.to_string(), v);
    }
    Ok(Value::Object(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::VarLenUnicode;

    #[derive(hdf5::H5Type, Clone)]
    #[repr(C)]
    struct Rec {
        a: f32,
        b: i16,
        flag: bool,
        name: VarLenUnicode,
    }

    /// Regression (CELLxGENE 5e787e2e): uns rec-arrays (scanpy's
    /// rank_genes_groups) read as null and were dropped on write. They now
    /// read as one array per field.
    #[test]
    fn rec_array_reads_as_dict_of_fields() {
        let tmp = tempfile::NamedTempFile::with_suffix(".h5").unwrap();
        let f = hdf5::File::create(tmp.path()).unwrap();
        let uns = f.create_group("uns").unwrap();
        let rec = |a, b, flag, s: &str| Rec {
            a,
            b,
            flag,
            name: s.parse().unwrap(),
        };
        let rows = [rec(1.5, -2, true, "CD3E"), rec(0.25, 7, false, "MS4A1")];
        uns.new_dataset_builder()
            .with_data(&rows)
            .create("names")
            .unwrap();
        let v = read_json(&f, "uns").unwrap();
        assert_eq!(
            v,
            serde_json::json!({"names": {
                "a": [1.5, 0.25], "b": [-2, 7], "flag": [true, false], "name": ["CD3E", "MS4A1"]
            }})
        );
    }

    /// Regression (anndataR uns *_with_nas): JSON has no NaN, so a NaN
    /// scalar was dropped, a vector with NaN came back as a dict of its other
    /// entries, and every matrix was flattened. Now NaN survives as null ->
    /// NaN and N-d arrays keep their shape.
    #[test]
    fn uns_nan_and_matrices_round_trip() {
        let tmp = tempfile::NamedTempFile::with_suffix(".h5").unwrap();
        let f = hdf5::File::create(tmp.path()).unwrap();
        let uns = f.create_group("uns").unwrap();
        let nan = f64::NAN;
        uns.new_dataset_builder()
            .with_data(&ndarray::arr0(nan))
            .create("scalar")
            .unwrap();
        uns.new_dataset_builder()
            .with_data(&[nan, 0.5, nan])
            .create("vec")
            .unwrap();
        uns.new_dataset_builder()
            .with_data(&ndarray::array![[1.0, nan, 3.0], [4.0, 5.0, 6.0]])
            .create("mat")
            .unwrap();
        uns.new_dataset_builder()
            .with_data(&ndarray::array![[1i64, 2], [3, 4], [5, 6]])
            .create("imat")
            .unwrap();

        let v = read_json(&f, "uns").unwrap();
        let want = serde_json::json!({
            "scalar": null,
            "vec": [null, 0.5, null],
            "mat": [[1.0, null, 3.0], [4.0, 5.0, 6.0]],
            "imat": [[1, 2], [3, 4], [5, 6]],
        });
        assert_eq!(v, want);

        let out = f.create_group("out").unwrap();
        for (k, x) in v.as_object().unwrap() {
            write_json(&out, k, x, Encoding::AnnData).unwrap();
        }
        assert_eq!(out.dataset("mat").unwrap().shape(), [2, 3]);
        assert_eq!(out.dataset("imat").unwrap().shape(), [3, 2]);
        assert!(out
            .dataset("scalar")
            .unwrap()
            .read_scalar::<f64>()
            .unwrap()
            .is_nan());
        assert_eq!(read_json(&f, "out").unwrap(), want);
    }
}
