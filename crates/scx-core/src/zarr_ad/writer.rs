//! AnnData Zarr v3 store writer: the layout `anndata.write_zarr` produces,
//! as [`ZarrAdReader`](super::ZarrAdReader) reads it and [`H5AdWriter`]
//! writes it to HDF5 (same encodings, same dataframe/categorical/nullable
//! rules; the two writers differ only in the container).
//!
//! Zarr v3 only. Arrays are zstd-compressed, regular-chunked, unsharded;
//! anndata >= 0.13 shards by default, but reads unsharded stores the same.
//! X, layers and obsp stream: `data`/`indices` grow by `set_shape` per chunk
//! and their metadata is stored once at the end.
//!
//! Call order is the one every [`DatasetWriter`] has: metadata in any order,
//! then `write_x_chunk` in row order, then `finalize`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::{
    data_type as zdt, Array, ArrayBuilder, DataType as ZDataType, Element, FillValue,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

use crate::h5ad::writer::cast;
use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
        SparseMatrixMeta, UnsTable, VarTable, Varm,
    },
    stream::DatasetWriter,
};

type Store = Arc<FilesystemStore>;
type ZArray = Array<FilesystemStore>;

/// Elements per chunk of the growable `data`/`indices` arrays (4 MiB of f32).
// ponytail: unsharded; one file per chunk. Add the sharding codec when a store
// has to live on object storage, where file count matters.
const CHUNK_ELEMS: u64 = 1 << 20;
/// zstd level 0 is the library default (3), which is what anndata writes.
const ZSTD_LEVEL: i32 = 0;

fn zerr(path: &str, e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> ScxError {
    ScxError::Zarr {
        path: path.to_string(),
        source: e.into(),
    }
}

fn zpath(path: &str) -> String {
    format!("/{}", path.trim_start_matches('/'))
}

fn encoding(kind: &str, version: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("encoding-type".into(), json!(kind));
    m.insert("encoding-version".into(), json!(version));
    m
}

fn array_enc() -> Map<String, Value> {
    encoding("array", "0.2.0")
}

fn create_group(store: &Store, path: &str, attrs: Map<String, Value>) -> Result<()> {
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), &zpath(path))
        .map_err(|e| zerr(path, e))?
        .store_metadata()
        .map_err(|e| zerr(path, e))
}

/// A regular-chunked, zstd-compressed array; metadata stored.
fn create_array(
    store: &Store,
    path: &str,
    data_type: ZDataType,
    fill: FillValue,
    shape: Vec<u64>,
    chunk: Vec<u64>,
    attrs: Map<String, Value>,
) -> Result<ZArray> {
    let arr = ArrayBuilder::new(shape, chunk, data_type, fill)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(ZSTD_LEVEL, false))])
        .attributes(attrs)
        .build(store.clone(), &zpath(path))
        .map_err(|e| zerr(path, e))?;
    arr.store_metadata().map_err(|e| zerr(path, e))?;
    Ok(arr)
}

/// A whole 1-D array in one go, chunked at most `CHUNK_ELEMS`.
fn write_1d<T: Element>(
    store: &Store,
    path: &str,
    data_type: ZDataType,
    fill: FillValue,
    values: &[T],
    attrs: Map<String, Value>,
) -> Result<()> {
    let n = values.len() as u64;
    let arr = create_array(
        store,
        path,
        data_type,
        fill,
        vec![n],
        vec![n.clamp(1, CHUNK_ELEMS)],
        attrs,
    )?;
    if n > 0 {
        arr.store_array_subset(&vec![0..n], values)
            .map_err(|e| zerr(path, e))?;
    }
    Ok(())
}

fn write_strings(store: &Store, path: &str, v: &[String], attrs: Map<String, Value>) -> Result<()> {
    write_1d(store, path, zdt::string(), FillValue::from(""), v, attrs)
}

fn write_bools(store: &Store, path: &str, v: &[bool], attrs: Map<String, Value>) -> Result<()> {
    write_1d(store, path, zdt::bool(), FillValue::from(false), v, attrs)
}

fn write_i32s(store: &Store, path: &str, v: &[i32], attrs: Map<String, Value>) -> Result<()> {
    write_1d(store, path, zdt::int32(), FillValue::from(0i32), v, attrs)
}

fn write_i64s(store: &Store, path: &str, v: &[i64], attrs: Map<String, Value>) -> Result<()> {
    write_1d(store, path, zdt::int64(), FillValue::from(0i64), v, attrs)
}

fn write_f64s(store: &Store, path: &str, v: &[f64], attrs: Map<String, Value>) -> Result<()> {
    write_1d(store, path, zdt::float64(), FillValue::from(0f64), v, attrs)
}

/// A 0-d array holding one value, as anndata stores uns scalars.
fn write_scalar<T: Element>(
    store: &Store,
    path: &str,
    data_type: ZDataType,
    fill: FillValue,
    value: T,
    attrs: Map<String, Value>,
) -> Result<()> {
    let arr = create_array(store, path, data_type, fill, vec![], vec![], attrs)?;
    arr.store_chunk(&[], &[value][..])
        .map_err(|e| zerr(path, e))
}

/// `indptr` as int32, or int64 when it doesn't fit (as h5ad).
fn write_indptr(store: &Store, path: &str, indptr: &[u64]) -> Result<()> {
    if indptr.last().is_some_and(|&n| n > i32::MAX as u64) {
        let v: Vec<i64> = indptr.iter().map(|&x| x as i64).collect();
        write_1d(
            store,
            path,
            zdt::int64(),
            FillValue::from(0i64),
            &v,
            array_enc(),
        )
    } else {
        let v: Vec<i32> = indptr.iter().map(|&x| x as i32).collect();
        write_i32s(store, path, &v, array_enc())
    }
}

fn write_2d_f64(store: &Store, path: &str, mat: &DenseMatrix) -> Result<()> {
    let (r, c) = (mat.shape.0 as u64, mat.shape.1 as u64);
    if mat.data.len() as u64 != r * c {
        return Err(ScxError::InvalidFormat(format!(
            "{path}: {} values for shape {:?}",
            mat.data.len(),
            mat.shape
        )));
    }
    let rows_per_chunk = (CHUNK_ELEMS / c.max(1)).clamp(1, r.max(1));
    let arr = create_array(
        store,
        path,
        zdt::float64(),
        FillValue::from(0f64),
        vec![r, c],
        vec![rows_per_chunk, c.max(1)],
        array_enc(),
    )?;
    if r > 0 && c > 0 {
        arr.store_array_subset(&vec![0..r, 0..c], &mat.data[..])
            .map_err(|e| zerr(path, e))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataframes (same rules as h5ad's write_dataframe / write_column)
// ---------------------------------------------------------------------------

fn write_dataframe(store: &Store, path: &str, index: &[String], columns: &[Column]) -> Result<()> {
    // A column named "_index" collides with the reserved index array (h5ad
    // drops it for the same reason).
    let columns: Vec<&Column> = columns
        .iter()
        .filter(|c| {
            if c.name == "_index" {
                tracing::warn!(group = path, "dropping column '_index': reserved");
            }
            c.name != "_index"
        })
        .collect();
    let mut attrs = encoding("dataframe", "0.2.0");
    attrs.insert("_index".into(), json!("_index"));
    attrs.insert(
        "column-order".into(),
        Value::from(columns.iter().map(|c| c.name.clone()).collect::<Vec<_>>()),
    );
    create_group(store, path, attrs)?;
    write_strings(
        store,
        &format!("{path}/_index"),
        index,
        encoding("string-array", "0.2.0"),
    )?;
    for col in columns {
        write_column(store, &format!("{path}/{}", col.name), col)?;
    }
    Ok(())
}

/// `values` + `mask` (mask true = NA, the inverse of `Column::mask`).
fn write_nullable(
    store: &Store,
    path: &str,
    kind: &str,
    present: &[bool],
    values: impl FnOnce(&str) -> Result<()>,
) -> Result<()> {
    create_group(store, path, encoding(kind, "0.1.0"))?;
    values(&format!("{path}/values"))?;
    let missing: Vec<bool> = present.iter().map(|&p| !p).collect();
    write_bools(store, &format!("{path}/mask"), &missing, array_enc())
}

fn write_column(store: &Store, path: &str, col: &Column) -> Result<()> {
    match (&col.data, &col.mask) {
        (ColumnData::Float(v), mask) => {
            // Float NA is NaN, so no mask group is needed.
            let vals: Vec<f64> = match mask {
                None => v.clone(),
                Some(m) => v
                    .iter()
                    .zip(m)
                    .map(|(&x, &p)| if p { x } else { f64::NAN })
                    .collect(),
            };
            write_f64s(store, path, &vals, array_enc())
        }
        (ColumnData::Int(v), Some(m)) => write_nullable(store, path, "nullable-integer", m, |p| {
            write_i32s(store, p, v, array_enc())
        }),
        (ColumnData::Int64(v), Some(m)) => {
            write_nullable(store, path, "nullable-integer", m, |p| {
                write_i64s(store, p, v, array_enc())
            })
        }
        (ColumnData::Bool(v), Some(m)) => write_nullable(store, path, "nullable-boolean", m, |p| {
            write_bools(store, p, v, array_enc())
        }),
        (ColumnData::String(v), Some(m)) => {
            write_nullable(store, path, "nullable-string-array", m, |p| {
                write_strings(store, p, v, encoding("string-array", "0.2.0"))
            })
        }
        (ColumnData::Int(v), None) => write_i32s(store, path, v, array_enc()),
        (ColumnData::Int64(v), None) => write_i64s(store, path, v, array_enc()),
        (ColumnData::Bool(v), None) => write_bools(store, path, v, array_enc()),
        (ColumnData::String(v), None) => {
            write_strings(store, path, v, encoding("string-array", "0.2.0"))
        }
        (ColumnData::Categorical { codes, levels }, _) => {
            let mut attrs = encoding("categorical", "0.2.0");
            attrs.insert("ordered".into(), json!(false));
            create_group(store, path, attrs)?;
            // NA is code -1; codes are the narrowest signed int holding every
            // level index (as h5ad).
            let codes: Vec<i32> = codes
                .iter()
                .enumerate()
                .map(|(i, &c)| if col.is_na(i) { -1 } else { c as i32 })
                .collect();
            let cp = format!("{path}/codes");
            if levels.len() <= i8::MAX as usize {
                let c: Vec<i8> = codes.iter().map(|&x| x as i8).collect();
                write_1d(
                    store,
                    &cp,
                    zdt::int8(),
                    FillValue::from(0i8),
                    &c,
                    array_enc(),
                )?;
            } else if levels.len() <= i16::MAX as usize {
                let c: Vec<i16> = codes.iter().map(|&x| x as i16).collect();
                write_1d(
                    store,
                    &cp,
                    zdt::int16(),
                    FillValue::from(0i16),
                    &c,
                    array_enc(),
                )?;
            } else {
                write_i32s(store, &cp, &codes, array_enc())?;
            }
            write_strings(
                store,
                &format!("{path}/categories"),
                levels,
                encoding("string-array", "0.2.0"),
            )
        }
    }
}

fn write_dense_dict(store: &Store, path: &str, map: &HashMap<String, DenseMatrix>) -> Result<()> {
    create_group(store, path, encoding("dict", "0.1.0"))?;
    let mut names: Vec<&String> = map.keys().collect();
    names.sort();
    for name in names {
        write_2d_f64(store, &format!("{path}/{name}"), &map[name])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// uns (same mapping as h5_json::write_json for AnnData)
// ---------------------------------------------------------------------------

fn write_json(store: &Store, path: &str, value: &Value) -> Result<()> {
    let scalar_enc = encoding("numeric-scalar", "0.2.0");
    match value {
        Value::Null => Ok(()),
        Value::Bool(b) => write_scalar(
            store,
            path,
            zdt::bool(),
            FillValue::from(false),
            *b,
            scalar_enc,
        ),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => write_scalar(
                store,
                path,
                zdt::int64(),
                FillValue::from(0i64),
                i,
                scalar_enc,
            ),
            (None, Some(u)) => write_scalar(
                store,
                path,
                zdt::uint64(),
                FillValue::from(0u64),
                u,
                scalar_enc,
            ),
            _ => write_scalar(
                store,
                path,
                zdt::float64(),
                FillValue::from(0f64),
                n.as_f64().unwrap_or(f64::NAN),
                scalar_enc,
            ),
        },
        Value::String(s) => write_scalar(
            store,
            path,
            zdt::string(),
            FillValue::from(""),
            s.clone(),
            encoding("string", "0.2.0"),
        ),
        Value::Array(a) if a.iter().all(Value::is_boolean) && !a.is_empty() => {
            let v: Vec<bool> = a.iter().map(|v| v.as_bool() == Some(true)).collect();
            write_bools(store, path, &v, array_enc())
        }
        Value::Array(a) if a.iter().all(|v| v.as_i64().is_some()) && !a.is_empty() => {
            let v: Vec<i64> = a.iter().filter_map(Value::as_i64).collect();
            write_1d(
                store,
                path,
                zdt::int64(),
                FillValue::from(0i64),
                &v,
                array_enc(),
            )
        }
        Value::Array(a) if a.iter().all(Value::is_number) => {
            let v: Vec<f64> = a.iter().filter_map(Value::as_f64).collect();
            write_f64s(store, path, &v, array_enc())
        }
        Value::Array(a) if a.iter().all(Value::is_string) => {
            let v: Vec<String> = a
                .iter()
                .map(|s| s.as_str().unwrap_or_default().to_string())
                .collect();
            write_strings(store, path, &v, encoding("string-array", "0.2.0"))
        }
        Value::Array(a) => {
            create_group(store, path, encoding("dict", "0.1.0"))?;
            for (i, v) in a.iter().enumerate() {
                write_json(store, &format!("{path}/{i}"), v)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            create_group(store, path, encoding("dict", "0.1.0"))?;
            for (k, v) in map {
                write_json(store, &format!("{path}/{k}"), v)?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming CSR groups (X, layers, obsp)
// ---------------------------------------------------------------------------

/// A 1-D array appended to chunk by chunk; its metadata is stored on `finish`.
struct Growable {
    arr: ZArray,
    len: u64,
}

impl Growable {
    fn create(store: &Store, path: &str, data_type: ZDataType, fill: FillValue) -> Result<Self> {
        let arr = create_array(
            store,
            path,
            data_type,
            fill,
            vec![0],
            vec![CHUNK_ELEMS],
            array_enc(),
        )?;
        Ok(Self { arr, len: 0 })
    }

    fn append<T: Element>(&mut self, v: &[T]) -> Result<()> {
        let path = self.arr.path().as_str().to_string();
        let end = self.len + v.len() as u64;
        self.arr.set_shape(vec![end]).map_err(|e| zerr(&path, e))?;
        self.arr
            .store_array_subset(&vec![self.len..end], v)
            .map_err(|e| zerr(&path, e))?;
        self.len = end;
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        self.arr
            .store_metadata()
            .map_err(|e| zerr(self.arr.path().as_str(), e))
    }
}

/// One CSR group being streamed.
struct CsrState {
    path: String,
    data: Growable,
    indices: Growable,
    dtype: DataType,
    indptr: Vec<u64>,
}

impl CsrState {
    fn begin(store: &Store, path: &str, shape: (usize, usize), dtype: DataType) -> Result<Self> {
        let mut attrs = encoding("csr_matrix", "0.1.0");
        attrs.insert("shape".into(), json!([shape.0, shape.1]));
        create_group(store, path, attrs)?;
        let data = match dtype {
            DataType::F32 => Growable::create(
                store,
                &format!("{path}/data"),
                zdt::float32(),
                FillValue::from(0f32),
            )?,
            DataType::F64 => Growable::create(
                store,
                &format!("{path}/data"),
                zdt::float64(),
                FillValue::from(0f64),
            )?,
            DataType::I32 => Growable::create(
                store,
                &format!("{path}/data"),
                zdt::int32(),
                FillValue::from(0i32),
            )?,
            DataType::U32 => Growable::create(
                store,
                &format!("{path}/data"),
                zdt::uint32(),
                FillValue::from(0u32),
            )?,
        };
        let indices = Growable::create(
            store,
            &format!("{path}/indices"),
            zdt::int32(),
            FillValue::from(0i32),
        )?;
        Ok(Self {
            path: path.to_string(),
            data,
            indices,
            dtype,
            indptr: vec![0],
        })
    }

    fn append(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let csr: &SparseMatrixCSR = &chunk.data;
        if !csr.indices.is_empty() {
            match (&csr.data, self.dtype) {
                (TypedVec::F32(v), DataType::F32) => self.data.append(v)?,
                (TypedVec::F64(v), DataType::F64) => self.data.append(v)?,
                (TypedVec::I32(v), DataType::I32) => self.data.append(v)?,
                (TypedVec::U32(v), DataType::U32) => self.data.append(v)?,
                (v, DataType::F32) => self.data.append(&cast(v, |x| x as f32))?,
                (v, DataType::F64) => self.data.append(&cast(v, |x| x))?,
                (v, DataType::I32) => self.data.append(&cast(v, |x| x as i32))?,
                (v, DataType::U32) => self.data.append(&cast(v, |x| x as u32))?,
            }
            let idx: Vec<i32> = csr.indices.iter().map(|&i| i as i32).collect();
            self.indices.append(&idx)?;
        }
        let base = *self.indptr.last().unwrap();
        self.indptr
            .extend(csr.indptr[1..=chunk.nrows].iter().map(|&p| base + p));
        Ok(())
    }

    fn end(self, store: &Store) -> Result<u64> {
        self.data.finish()?;
        self.indices.finish()?;
        write_indptr(store, &format!("{}/indptr", self.path), &self.indptr)?;
        Ok(*self.indptr.last().unwrap())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Streaming writer for AnnData Zarr v3 stores. See the module docs.
pub struct ZarrAdWriter {
    store: Store,
    n_obs: usize,
    n_vars: usize,
    x: Option<CsrState>,
    sparse: Option<CsrState>,
}

impl ZarrAdWriter {
    /// Create the store at `path` (a directory, created if absent) with the
    /// matrix going to `X`.
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
    ) -> Result<Self> {
        Self::create_at(path, n_obs, n_vars, dtype, "X")
    }

    /// As [`create`](Self::create), with the matrix going to `x_path`: `"X"`
    /// or `"layers/<name>"`.
    pub fn create_at<P: AsRef<Path>>(
        path: P,
        n_obs: usize,
        n_vars: usize,
        dtype: DataType,
        x_path: &str,
    ) -> Result<Self> {
        let root = path.as_ref();
        std::fs::create_dir_all(root)?;
        let store: Store =
            Arc::new(FilesystemStore::new(root).map_err(|e| zerr(&root.display().to_string(), e))?);
        create_group(&store, "", encoding("anndata", "0.1.0"))?;
        if let Some(name) = x_path.strip_prefix("layers/") {
            if name.is_empty() {
                return Err(ScxError::InvalidFormat("empty layer name".into()));
            }
            create_group(&store, "layers", encoding("dict", "0.1.0"))?;
        } else if x_path != "X" {
            return Err(ScxError::InvalidFormat(format!(
                "x_path must be \"X\" or \"layers/<name>\", got {x_path:?}"
            )));
        }
        let x = CsrState::begin(&store, x_path, (n_obs, n_vars), dtype)?;
        Ok(Self {
            store,
            n_obs,
            n_vars,
            x: Some(x),
            sparse: None,
        })
    }

    /// The dict group `name` (layers, obsp, ...), created if absent.
    fn dict_group(&self, name: &str) -> Result<()> {
        if zarrs::group::Group::open(self.store.clone(), &zpath(name)).is_ok() {
            return Ok(());
        }
        create_group(&self.store, name, encoding("dict", "0.1.0"))
    }
}

#[async_trait]
impl DatasetWriter for ZarrAdWriter {
    async fn write_obs(&mut self, obs: &ObsTable) -> Result<()> {
        write_dataframe(&self.store, "obs", &obs.index, &obs.columns)
    }

    async fn write_var(&mut self, var: &VarTable) -> Result<()> {
        write_dataframe(&self.store, "var", &var.index, &var.columns)
    }

    async fn write_obsm(&mut self, obsm: &Embeddings) -> Result<()> {
        write_dense_dict(&self.store, "obsm", &obsm.map)
    }

    async fn write_varm(&mut self, varm: &Varm) -> Result<()> {
        write_dense_dict(&self.store, "varm", &varm.map)
    }

    async fn write_uns(&mut self, uns: &UnsTable) -> Result<()> {
        create_group(&self.store, "uns", encoding("dict", "0.1.0"))?;
        if let Some(obj) = uns.raw.as_object() {
            for (k, v) in obj {
                write_json(&self.store, &format!("uns/{k}"), v)?;
            }
        }
        Ok(())
    }

    async fn begin_sparse(
        &mut self,
        group_prefix: &str,
        name: &str,
        meta: &SparseMatrixMeta,
    ) -> Result<()> {
        self.dict_group(group_prefix)?;
        // The value type is fixed at begin; chunks are cast to it. The X dtype
        // is the one the reader reports, which is what every layer is too in
        // practice (counts next to normalised data are both f32).
        let dtype = self.x.as_ref().map_or(DataType::F32, |x| x.dtype);
        let path = format!("{group_prefix}/{name}");
        self.sparse = Some(CsrState::begin(&self.store, &path, meta.shape, dtype)?);
        Ok(())
    }

    async fn write_sparse_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let state = self.sparse.as_mut().ok_or_else(|| {
            ScxError::InvalidFormat("write_sparse_chunk called without begin_sparse".into())
        })?;
        state.append(chunk)
    }

    async fn end_sparse(&mut self) -> Result<()> {
        let state = self.sparse.take().ok_or_else(|| {
            ScxError::InvalidFormat("end_sparse called without begin_sparse".into())
        })?;
        state.end(&self.store).map(|_| ())
    }

    async fn write_x_chunk(&mut self, chunk: &MatrixChunk) -> Result<()> {
        let x = self
            .x
            .as_mut()
            .ok_or_else(|| ScxError::InvalidFormat("write_x_chunk after finalize".into()))?;
        x.append(chunk)
    }

    async fn finalize(&mut self) -> Result<()> {
        let x = self
            .x
            .take()
            .ok_or_else(|| ScxError::InvalidFormat("finalize called twice".into()))?;
        if x.indptr.len() != self.n_obs + 1 {
            return Err(ScxError::InvalidFormat(format!(
                "X has {} rows, expected {}",
                x.indptr.len() - 1,
                self.n_obs
            )));
        }
        let nnz = x.end(&self.store)?;
        tracing::info!(
            n_obs = self.n_obs,
            n_vars = self.n_vars,
            nnz,
            "zarr finalized"
        );
        Ok(())
    }
}
