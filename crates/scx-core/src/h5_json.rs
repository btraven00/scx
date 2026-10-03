//! Unstructured metadata (AnnData `uns`, Seurat `misc`) between JSON and HDF5.
//!
//! Objects are groups; scalars and 1-D arrays of one type are datasets.
//! Writing for AnnData also sets the `encoding-type` attributes anndata
//! expects; Seurat's `misc` has none.

use hdf5::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{File, Group, H5Type, Location};
use ndarray::Array1;
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

pub(crate) fn str_attr(loc: &Location, name: &str, value: &str) -> Result<()> {
    loc.new_attr::<VarLenUnicode>()
        .create(name)?
        .write_scalar(&vlen(value)?)?;
    Ok(())
}

fn encode(loc: &Location, enc: Encoding, kind: &str, version: &str) -> Result<()> {
    if enc == Encoding::AnnData {
        str_attr(loc, "encoding-type", kind)?;
        str_attr(loc, "encoding-version", version)?;
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
    let ds = parent.new_dataset::<T>().shape(v.len()).create(name)?;
    ds.write(&Array1::from_vec(v))?;
    encode(&ds, enc, kind, "0.2.0")
}

/// Write `value` as `parent/name`. `null` has no HDF5 form and is skipped.
/// An array that isn't all-bool, all-number or all-string (mixed, or holding
/// objects) is written as a group keyed "0", "1", ... — the closest AnnData
/// has to a list — and so reads back as an object.
pub(crate) fn write_json(parent: &Group, name: &str, value: &Value, enc: Encoding) -> Result<()> {
    match value {
        Value::Null => Ok(()),
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
        Value::Array(a) if a.iter().all(|v| v.as_i64().is_some()) && !a.is_empty() => array(
            parent,
            name,
            a.iter().filter_map(Value::as_i64).collect(),
            enc,
            "array",
        ),
        Value::Array(a) if a.iter().all(Value::is_number) => array(
            parent,
            name,
            a.iter().filter_map(Value::as_f64).collect(),
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
            let grp = parent.create_group(name)?;
            encode(&grp, enc, "dict", "0.1.0")?;
            for (i, v) in a.iter().enumerate() {
                write_json(&grp, &i.to_string(), v, enc)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            let grp = parent.create_group(name)?;
            encode(&grp, enc, "dict", "0.1.0")?;
            for (k, v) in map {
                write_json(&grp, k, v, enc)?;
            }
            Ok(())
        }
    }
}

/// Read the group at `path` as a JSON object. A dataset of a type with no
/// JSON form (compound, opaque, ...) becomes `null`.
pub(crate) fn read_json(file: &File, path: &str) -> Result<Value> {
    let grp = file.group(path)?;
    let mut map = serde_json::Map::new();
    for name in grp.member_names()? {
        let child = format!("{path}/{name}");
        let value = if file.dataset(&child).is_ok() {
            dataset_json(file, &child)?
        } else {
            read_json(file, &child)?
        };
        map.insert(name, value);
    }
    Ok(Value::Object(map))
}

fn dataset_json(file: &File, path: &str) -> Result<Value> {
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
    Ok(match ds.dtype()?.to_descriptor()? {
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
        _ => Value::Null,
    })
}
