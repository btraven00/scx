//! AnnData Zarr store reader (`.zarr` directories, Zarr v2 and v3).
//!
//! AnnData's Zarr layout is its h5ad layout on a different container: the same
//! `encoding-type` attributes, CSR groups of `data`/`indices`/`indptr`,
//! dataframes with `_index` and `column-order`, categoricals as
//! `codes`/`categories`, nullables as `values`/`mask`. So this mirrors
//! [`crate::h5ad::H5AdReader`] rule for rule, and its tests assert it returns
//! exactly what `H5AdReader` returns for the same AnnData written as h5ad.
//!
//! Spec: <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>
//!
//! Local filesystem stores only. zarrs decodes the chunks of a requested range
//! concurrently, so a row-chunk read is parallel across Zarr chunks without
//! anything like h5ad's hand-rolled parallel inflate. anndata >= 0.13 writes v3
//! arrays sharded; zarrs reads shards natively.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use serde_json::{Map, Value};
use zarrs::array::{data_type as zdt, Array, ElementOwned};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
    },
    sparse::sort_csr_indices,
    stream::DatasetReader,
};

type Store = Arc<FilesystemStore>;
type ZArray = Array<FilesystemStore>;

fn zerr(path: &str, e: impl std::fmt::Display) -> ScxError {
    ScxError::Zarr(format!("{path}: {e}"))
}

/// Streaming reader for AnnData Zarr stores. See the module docs.
pub struct ZarrAdReader {
    store: Store,
    n_obs: usize,
    n_vars: usize,
    /// CSR row pointer array (n_obs + 1 entries). None when X is dense.
    indptr: Option<Vec<u64>>,
    chunk_size: usize,
    dtype: DataType,
    /// Node the matrix is read from: `"X"`, or `"layers/<name>"` when X is
    /// absent (or a layer was requested explicitly).
    x_path: String,
}

impl ZarrAdReader {
    /// Open with the default matrix source: `X`, falling back to a layer when X
    /// is absent (a store written with `adata.X = None`).
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        Self::open_layer(path, chunk_size, None)
    }

    /// Open reading the matrix from `layers/<layer>` instead of `X`.
    pub fn open_layer<P: AsRef<Path>>(
        path: P,
        chunk_size: usize,
        layer: Option<&str>,
    ) -> Result<Self> {
        let root: PathBuf = path.as_ref().to_path_buf();
        let store: Store = Arc::new(
            FilesystemStore::new(&root).map_err(|e| zerr(&root.display().to_string(), e))?,
        );

        // Optional root encoding check, tolerating stores without it (as h5ad).
        if let Some(Node::Group(attrs)) = node(&store, "") {
            if let Some(enc) = attrs.get("encoding-type").and_then(Value::as_str) {
                if !enc.is_empty() && enc != "anndata" {
                    return Err(ScxError::InvalidFormat(format!(
                        "not an AnnData store: root encoding-type = '{enc}'"
                    )));
                }
            }
        }

        let base = resolve_matrix_path(&store, layer)?;
        let (n_obs, n_vars, indptr, dtype) = match node(&store, &base) {
            Some(Node::Array(arr)) => {
                let sh = arr.shape();
                if sh.len() != 2 {
                    return Err(ScxError::InvalidFormat(format!("dense {base} must be 2-D")));
                }
                (sh[0] as usize, sh[1] as usize, None, detect_dtype(&arr))
            }
            Some(Node::Group(attrs)) => {
                if attrs.get("encoding-type").and_then(Value::as_str) == Some("csc_matrix") {
                    return Err(ScxError::InvalidFormat(format!(
                        "{base} is stored as CSC. Convert to CSR first: \
                         adata.X = adata.X.tocsr(); adata.write_zarr(path)"
                    )));
                }
                let (n_obs, n_vars) = shape_attr(&attrs, &base)?;
                let indptr = read_u64(&open_array(&store, &format!("{base}/indptr"))?, None)?;
                if indptr.len() != n_obs + 1 {
                    return Err(ScxError::InvalidFormat(format!(
                        "{base}/indptr length {} != n_obs+1 {}",
                        indptr.len(),
                        n_obs + 1
                    )));
                }
                let dtype = detect_dtype(&open_array(&store, &format!("{base}/data"))?);
                (n_obs, n_vars, Some(indptr), dtype)
            }
            None => return Err(ScxError::InvalidFormat(format!("missing {base}"))),
        };

        Ok(Self {
            store,
            n_obs,
            n_vars,
            indptr,
            chunk_size,
            dtype,
            x_path: base,
        })
    }

    /// The node the matrix is read from — `"X"` or `"layers/<name>"`.
    pub fn x_source(&self) -> &str {
        &self.x_path
    }
}

// ---------------------------------------------------------------------------
// Store access
// ---------------------------------------------------------------------------

/// A node in the store: an array, or a group with its attributes.
enum Node {
    Array(Box<ZArray>),
    Group(Map<String, Value>),
}

/// zarrs paths are absolute ("/X/data"); scx uses h5ad-style relative ones.
fn zpath(path: &str) -> String {
    format!("/{}", path.trim_start_matches('/'))
}

fn node(store: &Store, path: &str) -> Option<Node> {
    let p = zpath(path);
    if let Ok(arr) = Array::open(store.clone(), &p) {
        return Some(Node::Array(Box::new(arr)));
    }
    Group::open(store.clone(), &p)
        .ok()
        .map(|g| Node::Group(g.attributes().clone()))
}

fn open_array(store: &Store, path: &str) -> Result<ZArray> {
    Array::open(store.clone(), &zpath(path)).map_err(|e| zerr(path, e))
}

fn is_group(store: &Store, path: &str) -> bool {
    matches!(node(store, path), Some(Node::Group(_)))
}

/// Member names of a group, sorted (HDF5 lists links by name; match it).
fn children(store: &Store, path: &str) -> Vec<String> {
    let Ok(g) = Group::open(store.clone(), &zpath(path)) else {
        return Vec::new();
    };
    let mut names: Vec<String> = g
        .child_paths()
        .unwrap_or_default()
        .iter()
        .filter_map(|p| p.as_str().rsplit('/').next().map(str::to_string))
        .collect();
    names.sort();
    names
}

fn shape_attr(attrs: &Map<String, Value>, path: &str) -> Result<(usize, usize)> {
    let s: Vec<u64> = attrs
        .get("shape")
        .and_then(Value::as_array)
        .map(|v| v.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    match s.as_slice() {
        [r, c] => Ok((*r as usize, *c as usize)),
        _ => Err(ScxError::InvalidFormat(format!(
            "missing or bad {path} shape attribute"
        ))),
    }
}

/// Whole array, or the given range of a 1-D array.
fn subset(arr: &ZArray, range: Option<Range<usize>>) -> Vec<Range<u64>> {
    match range {
        // One range: a 1-D subset (not `vec![a; b]`).
        Some(r) => std::iter::once(r.start as u64..r.end as u64).collect(),
        None => arr.shape().iter().map(|&n| 0..n).collect(),
    }
}

fn get<T: ElementOwned>(arr: &ZArray, sub: Vec<Range<u64>>) -> Result<Vec<T>> {
    arr.retrieve_array_subset::<Vec<T>>(&sub)
        .map_err(|e| zerr(arr.path().as_str(), e))
}

/// Read a numeric (or bool) array as `f64`, converting from its stored type.
macro_rules! read_numeric {
    ($name:ident, $out:ty) => {
        fn $name(arr: &ZArray, range: Option<Range<usize>>) -> Result<Vec<$out>> {
            let dt = arr.data_type();
            let s = subset(arr, range);
            Ok(if *dt == zdt::float32() {
                get::<f32>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::float64() {
                get::<f64>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::int8() {
                get::<i8>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::int16() {
                get::<i16>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::int32() {
                get::<i32>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::int64() {
                get::<i64>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::uint8() {
                get::<u8>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::uint16() {
                get::<u16>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::uint32() {
                get::<u32>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::uint64() {
                get::<u64>(arr, s)?.into_iter().map(|x| x as $out).collect()
            } else if *dt == zdt::bool() {
                get::<bool>(arr, s)?
                    .into_iter()
                    .map(|x| x as u8 as $out)
                    .collect()
            } else {
                return Err(zerr(
                    arr.path().as_str(),
                    format!("unsupported data type {dt:?}"),
                ));
            })
        }
    };
}
read_numeric!(read_f64, f64);
read_numeric!(read_i64, i64);
read_numeric!(read_u64, u64);

fn read_strings(arr: &ZArray) -> Result<Vec<String>> {
    get::<String>(arr, subset(arr, None))
}

fn read_bools(arr: &ZArray) -> Result<Vec<bool>> {
    if *arr.data_type() == zdt::bool() {
        get::<bool>(arr, subset(arr, None))
    } else {
        Ok(read_i64(arr, None)?.into_iter().map(|x| x != 0).collect())
    }
}

fn is_string(arr: &ZArray) -> bool {
    *arr.data_type() == zdt::string()
}

fn is_float(arr: &ZArray) -> bool {
    *arr.data_type() == zdt::float32() || *arr.data_type() == zdt::float64()
}

/// Same mapping as h5ad's `ad_detect_dtype`.
fn detect_dtype(arr: &ZArray) -> DataType {
    let dt = arr.data_type();
    if *dt == zdt::float32() {
        DataType::F32
    } else if *dt == zdt::float64() {
        DataType::F64
    } else if *dt == zdt::int32() || *dt == zdt::int64() {
        DataType::I32 // i64 → i32 (counts fit)
    } else if *dt == zdt::uint32() {
        DataType::U32
    } else {
        DataType::F32
    }
}

/// Read `data[range]` as `dtype`, converting from the stored type.
fn read_typed(arr: &ZArray, range: Range<usize>, dtype: DataType) -> Result<TypedVec> {
    let dt = arr.data_type();
    let s = subset(arr, Some(range.clone()));
    Ok(match dtype {
        DataType::F32 if *dt == zdt::float32() => TypedVec::F32(get::<f32>(arr, s)?),
        DataType::F64 if *dt == zdt::float64() => TypedVec::F64(get::<f64>(arr, s)?),
        DataType::I32 if *dt == zdt::int32() => TypedVec::I32(get::<i32>(arr, s)?),
        DataType::U32 if *dt == zdt::uint32() => TypedVec::U32(get::<u32>(arr, s)?),
        DataType::F32 => TypedVec::F32(
            read_f64(arr, Some(range))?
                .into_iter()
                .map(|x| x as f32)
                .collect(),
        ),
        DataType::F64 => TypedVec::F64(read_f64(arr, Some(range))?),
        DataType::I32 => TypedVec::I32(
            read_i64(arr, Some(range))?
                .into_iter()
                .map(|x| x as i32)
                .collect(),
        ),
        DataType::U32 => TypedVec::U32(
            read_u64(arr, Some(range))?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
        ),
    })
}

/// Same rules as h5ad's `resolve_matrix_path`.
fn resolve_matrix_path(store: &Store, layer: Option<&str>) -> Result<String> {
    if let Some(name) = layer {
        let p = format!("layers/{name}");
        if node(store, &p).is_none() {
            return Err(ScxError::InvalidFormat(format!(
                "layer '{name}' not found (no layers/{name})"
            )));
        }
        return Ok(p);
    }
    if node(store, "X").is_some() {
        return Ok("X".to_string());
    }
    if !is_group(store, "layers") {
        return Err(ScxError::InvalidFormat(
            "missing X and no layers to fall back to".into(),
        ));
    }
    let names = children(store, "layers");
    match names.as_slice() {
        [] => Err(ScxError::InvalidFormat(
            "missing X and layers is empty".into(),
        )),
        [only] => Ok(format!("layers/{only}")),
        many => match many.iter().find(|n| *n == "counts" || *n == "X") {
            Some(n) => Ok(format!("layers/{n}")),
            None => Err(ScxError::InvalidFormat(format!(
                "missing X; multiple layers present ({}). Pick one with --layer <name>",
                many.join(", ")
            ))),
        },
    }
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

fn read_csr_chunk(
    store: &Store,
    base: &str,
    indptr: &[u64],
    rows: Range<usize>,
    n_cols: usize,
    dtype: Option<DataType>,
) -> Result<MatrixChunk> {
    let (a, b) = (indptr[rows.start] as usize, indptr[rows.end] as usize);
    let data_arr = open_array(store, &format!("{base}/data"))?;
    let dtype = dtype.unwrap_or_else(|| detect_dtype(&data_arr));
    let (indices, data) = if b > a {
        let idx = open_array(store, &format!("{base}/indices"))?;
        let indices = read_u64(&idx, Some(a..b))?
            .into_iter()
            .map(|x| x as u32)
            .collect();
        (indices, read_typed(&data_arr, a..b, dtype)?)
    } else {
        (Vec::new(), TypedVec::F32(Vec::new()))
    };
    let chunk_indptr = indptr[rows.start..=rows.end]
        .iter()
        .map(|&p| p - indptr[rows.start])
        .collect();
    let nrows = rows.end - rows.start;
    // Unsorted column indices within a row are valid CSR; consumers
    // (dgCMatrix, CSC writers) need them sorted, as from H5AdReader.
    let mut csr = SparseMatrixCSR {
        shape: (nrows, n_cols),
        indptr: chunk_indptr,
        indices,
        data,
    };
    sort_csr_indices(&mut csr);
    Ok(MatrixChunk {
        row_offset: rows.start,
        nrows,
        data: csr,
    })
}

/// Read rows of a dense 2-D array and convert them to a CSR chunk, skipping
/// exact zeros (as h5ad's dense path does).
fn read_dense_chunk(
    store: &Store,
    path: &str,
    rows: Range<usize>,
    n_cols: usize,
    dtype: Option<DataType>,
) -> Result<MatrixChunk> {
    let arr = open_array(store, path)?;
    let dtype = dtype.unwrap_or_else(|| detect_dtype(&arr));
    let sub = vec![rows.start as u64..rows.end as u64, 0..n_cols as u64];
    let vals: Vec<f64> = read_f64_subset(&arr, sub)?;
    let nrows = rows.end - rows.start;
    let mut indptr = Vec::with_capacity(nrows + 1);
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    indptr.push(0u64);
    for row in vals.chunks(n_cols.max(1)) {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                indices.push(j as u32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    let data = match dtype {
        DataType::F32 => TypedVec::F32(data.iter().map(|&x| x as f32).collect()),
        DataType::F64 => TypedVec::F64(data),
        DataType::I32 => TypedVec::I32(data.iter().map(|&x| x as i32).collect()),
        DataType::U32 => TypedVec::U32(data.iter().map(|&x| x as u32).collect()),
    };
    Ok(MatrixChunk {
        row_offset: rows.start,
        nrows,
        data: SparseMatrixCSR {
            shape: (nrows, n_cols),
            indptr,
            indices,
            data,
        },
    })
}

/// A 2-D subset as row-major f64 (read_f64 with an explicit multi-dim subset).
fn read_f64_subset(arr: &ZArray, sub: Vec<Range<u64>>) -> Result<Vec<f64>> {
    let dt = arr.data_type();
    Ok(if *dt == zdt::float64() {
        get::<f64>(arr, sub)?
    } else if *dt == zdt::float32() {
        get::<f32>(arr, sub)?.into_iter().map(f64::from).collect()
    } else if *dt == zdt::int32() {
        get::<i32>(arr, sub)?.into_iter().map(f64::from).collect()
    } else if *dt == zdt::int64() {
        get::<i64>(arr, sub)?
            .into_iter()
            .map(|x| x as f64)
            .collect()
    } else {
        return Err(zerr(
            arr.path().as_str(),
            format!("unsupported dense dtype {dt:?}"),
        ));
    })
}

fn read_dense_2d(store: &Store, path: &str) -> Result<DenseMatrix> {
    let arr = open_array(store, path)?;
    let sh = arr.shape().to_vec();
    if sh.len() != 2 {
        return Err(zerr(path, "not 2-D"));
    }
    let data = read_f64_subset(&arr, subset(&arr, None))?;
    Ok(DenseMatrix {
        shape: (sh[0] as usize, sh[1] as usize),
        data,
    })
}

fn sparse_meta(store: &Store, name: &str, path: &str) -> Result<SparseMatrixMeta> {
    let Some(Node::Group(attrs)) = node(store, path) else {
        return Err(zerr(path, "not a sparse matrix group"));
    };
    let shape = shape_attr(&attrs, path)?;
    let indptr = read_u64(&open_array(store, &format!("{path}/indptr"))?, None)?;
    Ok(SparseMatrixMeta {
        name: name.to_string(),
        shape,
        indptr,
    })
}

// ---------------------------------------------------------------------------
// Dataframes (same rules as h5ad's ad_read_dataframe)
// ---------------------------------------------------------------------------

fn read_dataframe(store: &Store, path: &str) -> Result<(Vec<String>, Vec<Column>)> {
    let Some(Node::Group(attrs)) = node(store, path) else {
        return Err(ScxError::InvalidFormat(format!("missing dataframe {path}")));
    };
    let index_name = attrs
        .get("_index")
        .and_then(Value::as_str)
        .unwrap_or("index");
    let index = read_index(store, &format!("{path}/{index_name}"))?;
    let col_names: Vec<String> = attrs
        .get("column-order")
        .and_then(Value::as_array)
        .map(|v| {
            v.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let mut columns = Vec::new();
    for name in col_names {
        let col_path = format!("{path}/{name}");
        let data = match node(store, &col_path) {
            Some(Node::Group(_)) => {
                if node(store, &format!("{col_path}/codes")).is_some() {
                    read_categorical(store, &col_path)
                } else if node(store, &format!("{col_path}/values")).is_some() {
                    read_nullable(store, &col_path)
                } else {
                    Err(ScxError::InvalidFormat(format!(
                        "unknown group encoding at '{col_path}'"
                    )))
                }
            }
            Some(Node::Array(arr)) => read_column(&arr).map(|d| (d, Vec::new())),
            None => Err(ScxError::InvalidFormat(format!(
                "missing column '{col_path}'"
            ))),
        };
        match data {
            Ok((data, missing)) => columns.push(Column::with_missing(name, data, missing)),
            Err(e) => tracing::warn!("skipping column '{name}': {e}"),
        }
    }
    Ok((index, columns))
}

/// A plain string array, or (anndata >= 0.13) a nullable-string-array group
/// whose mask is ignored: an index has no meaningful missing value.
fn read_index(store: &Store, path: &str) -> Result<Vec<String>> {
    match node(store, path) {
        Some(Node::Array(arr)) => read_strings(&arr),
        Some(Node::Group(_)) => read_strings(&open_array(store, &format!("{path}/values"))?),
        None => Err(ScxError::InvalidFormat(format!("missing index '{path}'"))),
    }
}

fn read_column(arr: &ZArray) -> Result<ColumnData> {
    let dt = arr.data_type();
    if is_string(arr) {
        Ok(ColumnData::String(read_strings(arr)?))
    } else if is_float(arr) {
        Ok(ColumnData::Float(read_f64(arr, None)?))
    } else if *dt == zdt::bool() || *dt == zdt::uint8() || *dt == zdt::int8() {
        // 1-byte ints encode bools, as in h5ad.
        Ok(ColumnData::Bool(read_bools(arr)?))
    } else {
        Ok(ColumnData::Int(
            read_i64(arr, None)?.into_iter().map(|x| x as i32).collect(),
        ))
    }
}

/// Code -1 is NA: it becomes code 0 with `missing[i] == true`.
fn read_categorical(store: &Store, path: &str) -> Result<(ColumnData, Vec<bool>)> {
    let raw = read_i64(&open_array(store, &format!("{path}/codes"))?, None)?;
    let missing = raw.iter().map(|&c| c < 0).collect();
    let codes = raw.iter().map(|&c| c.max(0) as u32).collect();
    let cats = open_array(store, &format!("{path}/categories"))?;
    let levels = if is_string(&cats) {
        read_strings(&cats)?
    } else if is_float(&cats) {
        read_f64(&cats, None)?
            .iter()
            .map(|v| v.to_string())
            .collect()
    } else if *cats.data_type() == zdt::bool() {
        read_bools(&cats)?.iter().map(|v| v.to_string()).collect()
    } else {
        read_i64(&cats, None)?
            .iter()
            .map(|v| v.to_string())
            .collect()
    };
    Ok((ColumnData::Categorical { codes, levels }, missing))
}

/// `values` + `mask` (mask true = NA), as for pandas' nullable dtypes.
fn read_nullable(store: &Store, path: &str) -> Result<(ColumnData, Vec<bool>)> {
    let values = open_array(store, &format!("{path}/values"))?;
    let missing = match open_array(store, &format!("{path}/mask")) {
        Ok(m) => read_bools(&m)?,
        Err(_) => Vec::new(),
    };
    let data = if is_string(&values) {
        ColumnData::String(read_strings(&values)?)
    } else if *values.data_type() == zdt::bool() {
        ColumnData::Bool(read_bools(&values)?)
    } else if is_float(&values) {
        ColumnData::Float(read_f64(&values, None)?)
    } else {
        ColumnData::Int(
            read_i64(&values, None)?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
        )
    };
    Ok((data, missing))
}

// ---------------------------------------------------------------------------
// uns (same JSON mapping as h5ad's ad_walk_group)
// ---------------------------------------------------------------------------

fn walk_group(store: &Store, path: &str) -> Value {
    let mut map = Map::new();
    for name in children(store, path) {
        let child = format!("{path}/{name}");
        let value = match node(store, &child) {
            Some(Node::Group(_)) => walk_group(store, &child),
            Some(Node::Array(arr)) => array_to_json(&arr).unwrap_or(Value::Null),
            None => Value::Null,
        };
        map.insert(name, value);
    }
    Value::Object(map)
}

fn array_to_json(arr: &ZArray) -> Result<Value> {
    let scalar = arr.shape().is_empty();
    if is_string(arr) {
        let mut v = read_strings(arr)?;
        return Ok(if scalar || v.len() == 1 {
            Value::String(v.pop().unwrap_or_default())
        } else {
            Value::from(v)
        });
    }
    if arr.shape().len() > 1 || *arr.data_type() == zdt::bool() {
        return Ok(Value::Null); // as h5ad: only scalars and 1-D numbers/strings
    }
    Ok(if is_float(arr) {
        let v = read_f64(arr, None)?;
        if scalar {
            Value::from(v[0])
        } else {
            Value::from(v)
        }
    } else {
        let v = read_i64(arr, None)?;
        if scalar {
            Value::from(v[0])
        } else {
            Value::from(v)
        }
    })
}

// ---------------------------------------------------------------------------
// DatasetReader impl
// ---------------------------------------------------------------------------

fn row_chunks<'a, F>(
    n_rows: usize,
    chunk_size: usize,
    read: F,
) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>>
where
    F: Fn(Range<usize>) -> Result<MatrixChunk> + Send + 'a,
{
    Box::pin(stream::iter(
        (0..n_rows)
            .step_by(chunk_size.max(1))
            .map(move |start| read(start..(start + chunk_size).min(n_rows))),
    ))
}

#[async_trait]
impl DatasetReader for ZarrAdReader {
    fn x_indptr(&self) -> &[u64] {
        self.indptr.as_deref().unwrap_or(&[])
    }

    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        let (index, columns) = read_dataframe(&self.store, "obs")?;
        Ok(ObsTable { index, columns })
    }

    async fn var(&mut self) -> Result<VarTable> {
        let (index, columns) = read_dataframe(&self.store, "var")?;
        Ok(VarTable { index, columns })
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        let mut map = HashMap::new();
        for name in children(&self.store, "obsm") {
            match read_dense_2d(&self.store, &format!("obsm/{name}")) {
                Ok(m) => {
                    // Guard against transposed storage, as h5ad does.
                    let m = if m.shape.0 != self.n_obs && m.shape.1 == self.n_obs {
                        transpose(m)
                    } else {
                        m
                    };
                    map.insert(name, m);
                }
                Err(e) => tracing::warn!("skipping obsm['{name}']: {e}"),
            }
        }
        Ok(Embeddings { map })
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        if !is_group(&self.store, "uns") {
            return Ok(UnsTable::default());
        }
        let mut raw = walk_group(&self.store, "uns");
        // scx_provenance is stored as a JSON string; parse it back (as h5ad).
        if let Some(obj) = raw.as_object_mut() {
            if let Some(Value::String(s)) = obj.get("scx_provenance") {
                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                    obj.insert("scx_provenance".to_string(), parsed);
                }
            }
        }
        Ok(UnsTable { raw })
    }

    async fn varm(&mut self) -> Result<Varm> {
        let mut map = HashMap::new();
        for name in children(&self.store, "varm") {
            match read_dense_2d(&self.store, &format!("varm/{name}")) {
                Ok(m) => {
                    map.insert(name, m);
                }
                Err(e) => tracing::warn!("skipping varm['{name}']: {e}"),
            }
        }
        Ok(Varm { map })
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let mut metas = Vec::new();
        for name in children(&self.store, "layers") {
            let path = format!("layers/{name}");
            // The layer already serving as X (no X, or open_layer) is not also a
            // layer, or every consumer carries the matrix twice (h5ad: PR #34).
            if path == self.x_path {
                continue;
            }
            match node(&self.store, &path) {
                // Dense layer: shape only; indptr stays empty.
                Some(Node::Array(arr)) if arr.shape().len() == 2 => metas.push(SparseMatrixMeta {
                    name,
                    shape: (arr.shape()[0] as usize, arr.shape()[1] as usize),
                    indptr: Vec::new(),
                }),
                Some(Node::Array(arr)) => tracing::warn!(
                    "skipping dense layer '{name}': unexpected rank {}",
                    arr.shape().len()
                ),
                _ => match sparse_meta(&self.store, &name, &path) {
                    Ok(m) => metas.push(m),
                    Err(e) => tracing::warn!("skipping layers['{name}']: {e}"),
                },
            }
        }
        Ok(metas)
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let mut metas = Vec::new();
        for name in children(&self.store, "obsp") {
            match sparse_meta(&self.store, &name, &format!("obsp/{name}")) {
                Ok(m) => metas.push(m),
                Err(e) => tracing::warn!("skipping obsp['{name}']: {e}"),
            }
        }
        Ok(metas)
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        let store = self.store.clone();
        let path = format!("layers/{}", meta.name);
        let (n_rows, n_cols) = meta.shape;
        row_chunks(n_rows, chunk_size, move |rows| {
            if meta.indptr.is_empty() {
                read_dense_chunk(&store, &path, rows, n_cols, None)
            } else {
                read_csr_chunk(&store, &path, &meta.indptr, rows, n_cols, None)
            }
        })
    }

    fn obsp_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        let store = self.store.clone();
        let path = format!("obsp/{}", meta.name);
        let (n_rows, n_cols) = meta.shape;
        row_chunks(n_rows, chunk_size, move |rows| {
            read_csr_chunk(&store, &path, &meta.indptr, rows, n_cols, None)
        })
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let store = self.store.clone();
        let base = self.x_path.clone();
        let (n_obs, n_vars, dtype) = (self.n_obs, self.n_vars, self.dtype);
        let indptr = self.indptr.clone();
        row_chunks(n_obs, self.chunk_size, move |rows| match &indptr {
            Some(indptr) => read_csr_chunk(&store, &base, indptr, rows, n_vars, Some(dtype)),
            None => read_dense_chunk(&store, &base, rows, n_vars, Some(dtype)),
        })
    }
}

fn transpose(m: DenseMatrix) -> DenseMatrix {
    let (r, c) = m.shape;
    let mut data = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            data[j * r + i] = m.data[i * c + j];
        }
    }
    DenseMatrix {
        shape: (c, r),
        data,
    }
}

#[cfg(test)]
mod tests;
