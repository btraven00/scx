use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5::{File, Group};
use ndarray::s;

use crate::h5::{read_csr_rows, read_u64, value_dtype};
use crate::h5_str::{read_str_array_attr, read_str_attr, read_strings};
use crate::stream::{row_chunks, ChunkStream};
use crate::{
    dtype::DataType,
    error::{Result, ScxError},
    ir::{
        Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixMeta,
        UnsTable, VarTable, Varm,
    },
    sparse::dense_to_csr,
    stream::DatasetReader,
};

/// Streaming reader for the AnnData `.h5ad` format.
///
/// Spec: <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>
///
/// Supports both sparse (CSR) and dense X storage.
/// Files with CSC X must be converted first:
///   `adata.X = adata.X.tocsr(); adata.write_h5ad(path)`
pub struct H5AdReader {
    path: PathBuf,
    n_obs: usize,
    n_vars: usize,
    /// CSR row pointer array (n_obs + 1 entries). None when X is dense.
    indptr: Option<Vec<u64>>,
    chunk_size: usize,
    dtype: DataType,
    /// Base HDF5 path the matrix is read from: `"X"`, or `"layers/<name>"` when
    /// X is absent (or a layer was requested explicitly).
    x_path: String,
}

impl H5AdReader {
    /// Open with the default matrix source: `/X`, falling back to a layer when
    /// `/X` is absent (common for files written with `adata.X = None`).
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        Self::open_layer(path, chunk_size, None)
    }

    /// Open reading the matrix from `layers/<layer>` instead of `/X`. Passing
    /// `None` uses `/X`, auto-falling-back to a layer when `/X` is missing.
    pub fn open_layer<P: AsRef<Path>>(
        path: P,
        chunk_size: usize,
        layer: Option<&str>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        // Optional root encoding check — tolerate files without it
        if let Ok(root) = file.group("/") {
            if let Ok(enc) = read_str_attr(&root, "encoding-type") {
                if !enc.is_empty() && enc != "anndata" {
                    return Err(ScxError::InvalidFormat(format!(
                        "not an AnnData file: root encoding-type = '{enc}'"
                    )));
                }
            }
        }

        let base = resolve_matrix_path(&file, layer)?;

        // The matrix can be stored as a dense 2-D dataset or a sparse CSR group.
        let is_dense = file.dataset(&base).is_ok() && file.group(&base).is_err();

        let (n_obs, n_vars, indptr, dtype) = if is_dense {
            let ds = file.dataset(&base)?;
            let sh = ds.shape();
            if sh.len() != 2 {
                return Err(ScxError::InvalidFormat(format!("dense {base} must be 2-D")));
            }
            (sh[0], sh[1], None, value_dtype(&ds)?)
        } else {
            let meta = ad_read_sparse_meta(&file, &base, &base)?;
            let dtype = value_dtype(&file.dataset(&format!("{base}/data"))?)?;
            (meta.shape.0, meta.shape.1, Some(meta.indptr), dtype)
        };

        Ok(Self {
            path,
            n_obs,
            n_vars,
            indptr,
            chunk_size,
            dtype,
            x_path: base,
        })
    }

    /// The HDF5 path the matrix is read from — `"X"` or `"layers/<name>"`.
    pub fn x_source(&self) -> &str {
        &self.x_path
    }
}

/// Decide which HDF5 path holds the count matrix.
///
/// - `Some(name)` → `layers/<name>` (errors if that layer is absent).
/// - `None` with `/X` present → `"X"`.
/// - `None` with `/X` absent → fall back to a layer: the sole layer if there's
///   exactly one, else one named `counts`/`X`, else an error listing choices.
fn resolve_matrix_path(file: &File, layer: Option<&str>) -> Result<String> {
    if let Some(name) = layer {
        let p = format!("layers/{name}");
        if file.group(&p).is_err() && file.dataset(&p).is_err() {
            return Err(ScxError::InvalidFormat(format!(
                "layer '{name}' not found (no /layers/{name})"
            )));
        }
        return Ok(p);
    }

    if file.dataset("X").is_ok() || file.group("X").is_ok() {
        return Ok("X".to_string());
    }

    // X absent — try to promote a layer.
    let layers = file
        .group("layers")
        .map_err(|_| ScxError::InvalidFormat("missing /X and no /layers to fall back to".into()))?;
    let names = layers.member_names().unwrap_or_default();
    match names.as_slice() {
        [] => Err(ScxError::InvalidFormat(
            "missing /X and /layers is empty".into(),
        )),
        [only] => Ok(format!("layers/{only}")),
        many => {
            let pick = many.iter().find(|n| *n == "counts" || *n == "X");
            match pick {
                Some(n) => Ok(format!("layers/{n}")),
                None => Err(ScxError::InvalidFormat(format!(
                    "missing /X; multiple layers present ({}). Pick one with --layer <name>",
                    many.join(", ")
                ))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------

/// Rows `rows` of the `n_cols`-column matrix at `matrix` in the file at
/// `path`: a CSR group, or a dense 2-D dataset when `indptr` is empty (dense
/// rows become CSR, dropping zeros).
fn read_rows(
    path: &Path,
    matrix: &str,
    indptr: &[u64],
    rows: Range<usize>,
    n_cols: usize,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    if !indptr.is_empty() {
        let dtype = value_dtype(&file.dataset(&format!("{matrix}/data"))?)?;
        return read_csr_rows(&file, matrix, indptr, rows, n_cols, dtype);
    }
    let ds = file.dataset(matrix)?;
    let dtype = value_dtype(&ds)?;
    let values: Vec<f64> = ds
        .read_slice::<f64, _, ndarray::Ix2>(s![rows.clone(), ..])?
        .iter()
        .copied()
        .collect();
    Ok(MatrixChunk {
        row_offset: rows.start,
        nrows: rows.len(),
        data: dense_to_csr(&values, n_cols, dtype),
    })
}

/// The `[rows, cols]` `shape` attribute of a sparse matrix group.
pub(super) fn read_shape(grp: &Group) -> Result<(usize, usize)> {
    let bad = |what: String| ScxError::InvalidFormat(format!("{}/shape {what}", grp.name()));
    let attr = grp
        .attr("shape")
        .map_err(|_| bad("attribute missing".into()))?;
    let shape = attr.read_raw::<i64>()?;
    let [rows, cols] = shape[..] else {
        return Err(bad(format!("has {} entries, expected 2", shape.len())));
    };
    let dim = |d: i64| usize::try_from(d).map_err(|_| bad(format!("has negative entry {d}")));
    Ok((dim(rows)?, dim(cols)?))
}

/// Read a dataframe group at `group_path` (e.g. "obs" or "var").
/// Returns (index, columns).
fn ad_read_dataframe(file: &File, group_path: &str) -> Result<(Vec<String>, Vec<Column>)> {
    let grp = file.group(group_path)?;

    // Index dataset name from _index attr; fall back to "index"
    let index_name = read_str_attr(&grp, "_index").unwrap_or_else(|_| "index".into());
    let index = ad_read_index(file, &format!("{group_path}/{index_name}"))?;

    let col_names = if grp.attr("column-order").is_ok() {
        read_str_array_attr(&grp, "column-order")?
    } else {
        Vec::new()
    };

    let mut columns = Vec::new();
    for col_name in col_names {
        let col_path = format!("{group_path}/{col_name}");
        // Groups are categorical; datasets are array/string-array
        let is_group = file.group(&col_path).is_ok() && file.dataset(&col_path).is_err();

        let read = if is_group {
            // Distinguish categorical (codes+categories) from nullable (values+mask)
            if file.dataset(&format!("{col_path}/codes")).is_ok() {
                ad_read_categorical(file, &col_path)
            } else if file.dataset(&format!("{col_path}/values")).is_ok() {
                ad_read_nullable(file, &col_path)
            } else {
                Err(ScxError::InvalidFormat(format!(
                    "unknown group encoding at '{col_path}'"
                )))
            }
        } else {
            ad_read_column(file, &col_path).map(|cd| (cd, Vec::new()))
        };
        let (data, missing) = read?;
        columns.push(Column::with_missing(col_name, data, missing));
    }

    Ok((index, columns))
}

/// Read a dataframe index.
///
/// Historically a plain `string-array` dataset. anndata 0.13 began writing
/// string columns — the index included — as `nullable-string-array`, a *group*
/// of `values` + `mask`, so opening the path as a dataset fails outright with
/// "not a dataset" on any file a current anndata wrote.
///
/// The mask is deliberately ignored: an index has no meaningful missing value,
/// and a masked entry is an empty label either way.
fn ad_read_index(file: &File, path: &str) -> Result<Vec<String>> {
    if file.dataset(path).is_ok() {
        return read_strings(file, path);
    }
    let values = format!("{path}/values");
    if file.dataset(&values).is_ok() {
        return read_strings(file, &values);
    }
    Err(ScxError::InvalidFormat(format!(
        "index at '{path}' is neither a string dataset nor a nullable-string-array group"
    )))
}

/// Read a single array or string-array dataset as ColumnData.
fn ad_read_column(file: &File, path: &str) -> Result<ColumnData> {
    let ds = file.dataset(path)?;
    // Prefer encoding-type attr; fall back to HDF5 dtype inspection
    let enc = read_str_attr(&ds, "encoding-type").unwrap_or_default();

    if enc == "string-array" {
        return Ok(ColumnData::String(read_strings(file, path)?));
    }

    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => {
            let v: Vec<f32> = ds.read_1d::<f32>()?.to_vec();
            Ok(ColumnData::Float(v.into_iter().map(|x| x as f64).collect()))
        }
        TypeDescriptor::Float(_) => Ok(ColumnData::Float(ds.read_1d::<f64>()?.to_vec())),
        // Native HDF5 boolean type (anndata >= 0.10 uses this for bool columns)
        TypeDescriptor::Boolean => Ok(ColumnData::Bool(ds.read_1d::<bool>()?.to_vec())),
        // 1-byte ints encode bool columns: our writer emits unsigned u8 0/1;
        // some sources use signed i8. (AnnData >= 0.10 uses native Boolean above.)
        TypeDescriptor::Unsigned(IntSize::U1) => {
            let v: Vec<u8> = ds.read_1d::<u8>()?.to_vec();
            Ok(ColumnData::Bool(v.into_iter().map(|x| x != 0).collect()))
        }
        TypeDescriptor::Integer(IntSize::U1) => {
            let v: Vec<i8> = ds.read_1d::<i8>()?.to_vec();
            Ok(ColumnData::Bool(v.into_iter().map(|x| x != 0).collect()))
        }
        TypeDescriptor::Integer(_) => Ok(ColumnData::Int(ds.read_1d::<i32>()?.to_vec())),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            Ok(ColumnData::String(read_strings(file, path)?))
        }
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported column dtype {:?} at '{path}'",
            other
        ))),
    }
}

/// Read a categorical group: codes (i8 or i16) + categories (string-array).
/// Read a categorical group. Code -1 is NA: it becomes code 0 with
/// `missing[i] == true`.
pub(super) fn ad_read_categorical(file: &File, grp_path: &str) -> Result<(ColumnData, Vec<bool>)> {
    let codes_ds = file.dataset(&format!("{grp_path}/codes"))?;
    let raw: Vec<i64> = match codes_ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(IntSize::U1) => codes_ds
            .read_1d::<i8>()?
            .iter()
            .map(|&x| x as i64)
            .collect(),
        TypeDescriptor::Integer(IntSize::U2) => codes_ds
            .read_1d::<i16>()?
            .iter()
            .map(|&x| x as i64)
            .collect(),
        TypeDescriptor::Integer(_) => codes_ds.read_1d::<i64>()?.to_vec(),
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unexpected categorical codes dtype {other:?}"
            )))
        }
    };
    let levels = ad_read_levels(file, &format!("{grp_path}/categories"))?;
    let missing: Vec<bool> = raw.iter().map(|&c| c < 0).collect();
    let codes = raw.iter().map(|&c| c.max(0) as u32).collect();
    Ok((ColumnData::Categorical { codes, levels }, missing))
}

/// Read a categorical `categories` dataset as level labels. AnnData usually
/// stores these as strings, but pandas Categoricals with integer/float levels
/// (e.g. cluster ids stored as small uints) are equally valid. We coerce any
/// scalar level dtype to its string label so `ColumnData::Categorical` always
/// carries `Vec<String>` levels.
fn ad_read_levels(file: &File, path: &str) -> Result<Vec<String>> {
    let ds = file.dataset(path)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => read_strings(file, path),
        TypeDescriptor::Integer(IntSize::U1) => {
            Ok(ds.read_1d::<i8>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            Ok(ds.read_1d::<i16>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            Ok(ds.read_1d::<i64>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Integer(_) => {
            Ok(ds.read_1d::<i32>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            Ok(ds.read_1d::<u8>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            Ok(ds.read_1d::<u16>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            Ok(ds.read_1d::<u64>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Unsigned(_) => {
            Ok(ds.read_1d::<u32>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Float(FloatSize::U4) => {
            Ok(ds.read_1d::<f32>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Float(_) => {
            Ok(ds.read_1d::<f64>()?.iter().map(|v| v.to_string()).collect())
        }
        TypeDescriptor::Boolean => Ok(ds
            .read_1d::<bool>()?
            .iter()
            .map(|v| v.to_string())
            .collect()),
        other => Err(ScxError::InvalidFormat(format!(
            "unsupported categorical level dtype {other:?} at '{path}'"
        ))),
    }
}

/// Read a nullable column group (`values` + `mask`, mask true = NA), as
/// written for pandas' nullable integer, boolean and string dtypes. Returns
/// the values and the per-value missing flags.
fn ad_read_nullable(file: &File, grp_path: &str) -> Result<(ColumnData, Vec<bool>)> {
    let ds = file.dataset(&format!("{grp_path}/values"))?;
    let missing: Vec<bool> = match file.dataset(&format!("{grp_path}/mask")) {
        Ok(mds) => match mds.dtype()?.to_descriptor()? {
            TypeDescriptor::Boolean => mds.read_1d::<bool>()?.to_vec(),
            TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {
                mds.read_1d::<i8>()?.iter().map(|&x| x != 0).collect()
            }
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "unsupported nullable mask dtype {other:?} at '{grp_path}'"
                )))
            }
        },
        Err(_) => Vec::new(),
    };
    let data = match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::FixedUnicode(_)
        | TypeDescriptor::FixedAscii(_) => ColumnData::String(crate::h5_str::read_str_1d(&ds)?),
        TypeDescriptor::Float(_) => ColumnData::Float(ds.read_1d::<f64>()?.to_vec()),
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {
            ColumnData::Int(ds.read_1d::<i32>()?.to_vec())
        }
        TypeDescriptor::Boolean => ColumnData::Bool(ds.read_1d::<bool>()?.to_vec()),
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unsupported nullable column dtype {other:?} at '{grp_path}'"
            )))
        }
    };
    Ok((data, missing))
}

/// The dense 2-D entries of the dict group `group` (obsm, varm). Entries
/// stored as groups (dataframes, sparse matrices) have no dense form and are
/// skipped with a warning. With `n_rows`, an entry stored transposed
/// (k × n_rows, as some writers do) is turned back to n_rows × k.
fn ad_read_dense_dict(
    file: &File,
    group: &str,
    n_rows: Option<usize>,
) -> Result<HashMap<String, DenseMatrix>> {
    let Ok(grp) = file.group(group) else {
        return Ok(HashMap::new());
    };
    let mut map = HashMap::new();
    for name in grp.member_names()? {
        let Ok(ds) = grp.dataset(&name) else {
            let entry = grp.group(&name)?;
            let enc = read_str_attr(&entry, "encoding-type").unwrap_or_default();
            tracing::warn!("skipping {group}['{name}']: {enc} entries are not supported");
            continue;
        };
        let arr = ds
            .read::<f64, ndarray::Ix2>()
            .map_err(|e| ScxError::InvalidFormat(format!("{group}['{name}']: {e}")))?;
        let arr = match n_rows {
            Some(n) if arr.shape()[0] != n && arr.shape()[1] == n => arr.t().to_owned(),
            _ => arr,
        };
        let shape = (arr.shape()[0], arr.shape()[1]);
        let data = arr.as_standard_layout().iter().copied().collect();
        map.insert(name, DenseMatrix { shape, data });
    }
    Ok(map)
}

/// The shape and indptr of the CSR matrix group at `group_path`. CSC is an
/// error: reading it as CSR would silently transpose the matrix.
fn ad_read_sparse_meta(file: &File, name: &str, group_path: &str) -> Result<SparseMatrixMeta> {
    let grp = file.group(group_path)?;
    if read_str_attr(&grp, "encoding-type").is_ok_and(|enc| enc == "csc_matrix") {
        return Err(ScxError::InvalidFormat(format!(
            "{group_path} is stored as CSC. Convert to CSR first, e.g. \
             adata.X = adata.X.tocsr(); adata.write_h5ad(path)"
        )));
    }
    let shape = read_shape(&grp)?;
    let indptr = read_u64(&file.dataset(&format!("{group_path}/indptr"))?)?;
    if indptr.len() != shape.0 + 1 {
        return Err(ScxError::InvalidFormat(format!(
            "{group_path}/indptr has {} entries, expected n_rows + 1 = {}",
            indptr.len(),
            shape.0 + 1
        )));
    }
    Ok(SparseMatrixMeta {
        name: name.to_string(),
        shape,
        indptr,
    })
}

// ---------------------------------------------------------------------------
// DatasetReader impl
// ---------------------------------------------------------------------------

#[async_trait]
impl DatasetReader for H5AdReader {
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
        let file = File::open(&self.path)?;
        let (index, columns) = ad_read_dataframe(&file, "obs")?;
        Ok(ObsTable { index, columns })
    }

    async fn var(&mut self) -> Result<VarTable> {
        let file = File::open(&self.path)?;
        let (index, columns) = ad_read_dataframe(&file, "var")?;
        Ok(VarTable { index, columns })
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        let file = File::open(&self.path)?;
        let map = ad_read_dense_dict(&file, "obsm", Some(self.n_obs))?;
        Ok(Embeddings { map })
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        let file = File::open(&self.path)?;
        match file.group("uns") {
            Err(_) => Ok(UnsTable::default()),
            Ok(_) => {
                let mut raw = crate::h5_json::read_json(&file, "uns")?;
                // scx_provenance is stored as a JSON string to preserve keys
                // containing "/" without HDF5 path-separator mangling.
                // Parse it back to an Object so callers get the expected shape.
                if let Some(obj) = raw.as_object_mut() {
                    if let Some(serde_json::Value::String(s)) = obj.get("scx_provenance") {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                            obj.insert("scx_provenance".to_string(), parsed);
                        }
                    }
                }
                Ok(UnsTable { raw })
            }
        }
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let file = File::open(&self.path)?;
        let Ok(grp) = file.group("layers") else {
            return Ok(Vec::new());
        };
        let mut metas = Vec::new();
        for name in grp.member_names()? {
            let grp_path = format!("layers/{name}");
            // The layer already serving as X (no /X, or open_layer) is not
            // also a layer: every consumer would carry the matrix twice.
            if grp_path == self.x_path {
                continue;
            }
            match grp.dataset(&name) {
                // Dense: an empty indptr marks it for read_rows.
                Ok(ds) => match ds.shape()[..] {
                    [rows, cols] => metas.push(SparseMatrixMeta {
                        name,
                        shape: (rows, cols),
                        indptr: Vec::new(),
                    }),
                    ref other => {
                        return Err(ScxError::InvalidFormat(format!(
                            "dense {grp_path} must be 2-D, has shape {other:?}"
                        )))
                    }
                },
                Err(_) => metas.push(ad_read_sparse_meta(&file, &name, &grp_path)?),
            }
        }
        Ok(metas)
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let file = File::open(&self.path)?;
        let Ok(grp) = file.group("obsp") else {
            return Ok(Vec::new());
        };
        grp.member_names()?
            .into_iter()
            .map(|name| {
                let grp_path = format!("obsp/{name}");
                ad_read_sparse_meta(&file, &name, &grp_path)
            })
            .collect()
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> ChunkStream<'a> {
        let matrix = format!("layers/{}", meta.name);
        row_chunks(meta.shape.0, chunk_size, move |rows| {
            read_rows(&self.path, &matrix, &meta.indptr, rows, meta.shape.1)
        })
    }

    fn obsp_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk_size: usize) -> ChunkStream<'a> {
        let matrix = format!("obsp/{}", meta.name);
        row_chunks(meta.shape.0, chunk_size, move |rows| {
            read_rows(&self.path, &matrix, &meta.indptr, rows, meta.shape.1)
        })
    }

    async fn varm(&mut self) -> Result<Varm> {
        let file = File::open(&self.path)?;
        let map = ad_read_dense_dict(&file, "varm", None)?;
        Ok(Varm { map })
    }

    fn x_stream(&mut self) -> ChunkStream<'_> {
        let this = &*self;
        let indptr = this.indptr.as_deref().unwrap_or(&[]);
        row_chunks(this.n_obs, this.chunk_size, move |rows| {
            read_rows(&this.path, &this.x_path, indptr, rows, this.n_vars)
        })
    }
}
