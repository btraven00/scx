use std::collections::{BTreeMap, HashMap};

use crate::dtype::{DataType, TypedVec};
use crate::error::ScxError;

/// Sparse matrix in CSR (Compressed Sparse Row) format.
/// Row-major — natural for cell-chunked streaming and AnnData output.
#[derive(Debug, Clone)]
pub struct SparseMatrixCSR {
    pub shape: (usize, usize),
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub data: TypedVec,
}

/// A chunk of rows from a streaming matrix read.
#[derive(Debug, Clone)]
pub struct MatrixChunk {
    pub row_offset: usize,
    pub nrows: usize,
    pub data: SparseMatrixCSR,
}

/// Metadata for a named sparse matrix (shape + indptr) used as a streaming handle.
/// The indptr is loaded upfront (cheap: n_obs+1 entries); data/indices are streamed.
#[derive(Debug, Clone)]
pub struct SparseMatrixMeta {
    pub name: String,
    pub shape: (usize, usize),
    pub indptr: Vec<u64>,
}

/// Cell (observation) metadata table.
#[derive(Debug, Clone, Default)]
pub struct ObsTable {
    pub index: Vec<String>,
    pub columns: Vec<Column>,
}

/// Gene/feature (variable) metadata table.
#[derive(Debug, Clone, Default)]
pub struct VarTable {
    pub index: Vec<String>,
    pub columns: Vec<Column>,
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
    /// Which values are present (`true`) or missing (NA, `false`); `None`
    /// when nothing is missing. At a missing position `data` holds a
    /// placeholder (0, NaN, false, "", code 0) that must not be read as a value.
    pub mask: Option<Vec<bool>>,
}

impl Column {
    /// A column with no missing values.
    pub fn new(name: impl Into<String>, data: ColumnData) -> Self {
        Self {
            name: name.into(),
            data,
            mask: None,
        }
    }

    /// A column whose values at `missing[i] == true` are NA. The mask is
    /// dropped when nothing is missing.
    pub fn with_missing(
        name: impl Into<String>,
        data: ColumnData,
        missing: impl IntoIterator<Item = bool>,
    ) -> Self {
        let mask: Vec<bool> = missing.into_iter().map(|m| !m).collect();
        Self {
            name: name.into(),
            data,
            mask: if mask.iter().all(|&p| p) {
                None
            } else {
                Some(mask)
            },
        }
    }

    /// Whether value `i` is missing.
    pub fn is_na(&self, i: usize) -> bool {
        self.mask.as_ref().is_some_and(|m| !m[i])
    }
}

#[derive(Debug, Clone)]
pub enum ColumnData {
    Int(Vec<i32>),
    /// 64-bit integers, as pandas stores them by default. Readers produce
    /// this only for 64-bit sources; 32-bit and narrower stay `Int`.
    Int64(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<String>),
    Bool(Vec<bool>),
    Categorical {
        codes: Vec<u32>,
        /// Level labels. Non-string levels are spelled as pandas prints them
        /// (`517`, `1.0`, `True`); `level_type` says what they were.
        levels: Vec<String>,
        /// Level order is meaningful (pandas `ordered=True`, R `ordered`).
        ordered: bool,
        /// The type of the levels in the source, so writers that can store
        /// typed categories (h5ad, Zarr) write them back as that type.
        level_type: LevelType,
    },
}

/// The type of a categorical's levels. Formats whose levels are always
/// strings (h5seurat, R factors) read and write `Str`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LevelType {
    #[default]
    Str,
    Int,
    Float,
    Bool,
}

impl LevelType {
    pub fn as_str(self) -> &'static str {
        match self {
            LevelType::Str => "str",
            LevelType::Int => "int",
            LevelType::Float => "float",
            LevelType::Bool => "bool",
        }
    }
}

/// `levels` parsed back to integers, when `level_type` is `Int` and every
/// label parses; `None` means write them as strings.
pub(crate) fn int_levels(level_type: LevelType, levels: &[String]) -> Option<Vec<i64>> {
    (level_type == LevelType::Int)
        .then(|| levels.iter().map(|l| l.parse().ok()).collect())
        .flatten()
}

/// As [`int_levels`], for `Float` levels.
pub(crate) fn float_levels(level_type: LevelType, levels: &[String]) -> Option<Vec<f64>> {
    (level_type == LevelType::Float)
        .then(|| levels.iter().map(|l| l.parse().ok()).collect())
        .flatten()
}

/// As [`int_levels`], for `Bool` levels (`True` / `False`).
pub(crate) fn bool_levels(level_type: LevelType, levels: &[String]) -> Option<Vec<bool>> {
    (level_type == LevelType::Bool)
        .then(|| {
            levels
                .iter()
                .map(|l| match l.as_str() {
                    "True" => Some(true),
                    "False" => Some(false),
                    _ => None,
                })
                .collect()
        })
        .flatten()
}

/// A float category level as pandas spells it (`str(1.0)` is `1.0`, not `1`),
/// so a float categorical reads back with the same labels.
pub(crate) fn float_level(v: impl std::fmt::Display) -> String {
    let s = v.to_string();
    if s.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
        s + ".0"
    } else {
        s
    }
}

/// A bool category level as pandas spells it.
pub(crate) fn bool_level(v: bool) -> String {
    if v { "True" } else { "False" }.to_string()
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Categorical { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype_str(&self) -> &'static str {
        match self {
            ColumnData::Float(_) => "float64",
            ColumnData::Int(_) => "int32",
            ColumnData::Int64(_) => "int64",
            ColumnData::Bool(_) => "bool",
            ColumnData::String(_) => "string",
            ColumnData::Categorical { .. } => "categorical",
        }
    }

    /// A categorical read as a dataframe index (anndata allows that; CxG
    /// files have one): one level name per row. A missing entry (`missing`)
    /// or an out-of-range code is an error, since an index has no NA.
    pub(crate) fn categorical_index(
        self,
        missing: &[bool],
        path: &str,
    ) -> Result<Vec<String>, ScxError> {
        let bad =
            |what: &str| ScxError::InvalidFormat(format!("categorical index at '{path}': {what}"));
        let ColumnData::Categorical { codes, levels, .. } = self else {
            return Err(bad("not categorical"));
        };
        if missing.iter().any(|&m| m) {
            return Err(bad("has missing entries"));
        }
        codes
            .iter()
            .map(|&c| {
                levels
                    .get(c as usize)
                    .cloned()
                    .ok_or_else(|| bad("code out of range"))
            })
            .collect()
    }

    /// Value `i` as text, the way pandas' `astype(str)` spells it (bools as
    /// `True`/`False`, categoricals by level). Ignores the column mask.
    pub fn value_str(&self, i: usize) -> String {
        match self {
            ColumnData::Int(v) => v[i].to_string(),
            ColumnData::Int64(v) => v[i].to_string(),
            ColumnData::Float(v) => v[i].to_string(),
            ColumnData::String(v) => v[i].clone(),
            ColumnData::Bool(v) => if v[i] { "True" } else { "False" }.to_string(),
            ColumnData::Categorical { codes, levels, .. } => levels[codes[i] as usize].clone(),
        }
    }
}

impl ObsTable {
    /// Which rows have `column` equal to one of `values`, compared as text
    /// (see [`ColumnData::value_str`]); NA never matches. AND masks of several
    /// columns together, then pass [`mask_to_ranges`] to
    /// [`DatasetReader::x_stream_rows`](crate::stream::DatasetReader::x_stream_rows).
    pub fn rows_where(&self, column: &str, values: &[&str]) -> Result<Vec<bool>, ScxError> {
        let col = self
            .columns
            .iter()
            .find(|c| c.name == column)
            .ok_or_else(|| {
                let have: Vec<&str> = self.columns.iter().map(|c| c.name.as_str()).collect();
                ScxError::MissingField(format!(
                    "obs column '{column}' (obs has: {})",
                    if have.is_empty() {
                        "no columns".to_string()
                    } else {
                        have.join(", ")
                    }
                ))
            })?;
        Ok((0..col.data.len())
            .map(|i| !col.is_na(i) && values.contains(&col.data.value_str(i).as_str()))
            .collect())
    }

    /// The [`rows_where`](Self::rows_where) masks of every `(column, values)`
    /// filter ANDed together: a row is kept when each column holds one of its
    /// values. With no filters, every one of `n_rows` rows is kept.
    pub fn rows_matching(
        &self,
        n_rows: usize,
        filters: &[(String, Vec<String>)],
    ) -> Result<Vec<bool>, ScxError> {
        let mut mask = vec![true; n_rows];
        for (column, values) in filters {
            let values: Vec<&str> = values.iter().map(String::as_str).collect();
            let hit = self.rows_where(column, &values)?;
            if hit.len() != n_rows {
                return Err(ScxError::InvalidFormat(format!(
                    "obs column '{column}' has {} rows, the matrix {n_rows}",
                    hit.len()
                )));
            }
            mask.iter_mut().zip(hit).for_each(|(m, h)| *m &= h);
        }
        Ok(mask)
    }
}

/// The runs of `true` in `mask` as row ranges, sorted and disjoint.
pub fn mask_to_ranges(mask: &[bool]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &m) in mask.iter().enumerate() {
        match (m, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..mask.len());
    }
    out
}

/// Named embedding matrices (e.g., PCA, UMAP).
#[derive(Debug, Clone, Default)]
pub struct Embeddings {
    pub map: HashMap<String, DenseMatrix>,
}

/// Named additional count matrices (e.g., "data", "spliced", "unspliced").
/// Each entry has shape (n_obs, n_vars).
#[derive(Debug, Clone, Default)]
pub struct Layers {
    pub map: HashMap<String, SparseMatrixCSR>,
}

/// Pairwise observation (cell) matrices (e.g., neighbor graphs).
/// Each entry has shape (n_obs, n_obs).
#[derive(Debug, Clone, Default)]
pub struct Obsp {
    pub map: HashMap<String, SparseMatrixCSR>,
}

/// Pairwise variable (gene) matrices (e.g., gene co-expression).
/// Each entry has shape (n_vars, n_vars).
#[derive(Debug, Clone, Default)]
pub struct Varp {
    pub map: HashMap<String, SparseMatrixCSR>,
}

/// Named variable (gene) embedding matrices (e.g., PCA gene loadings).
/// Each entry has shape (n_vars, k).
#[derive(Debug, Clone, Default)]
pub struct Varm {
    pub map: HashMap<String, DenseMatrix>,
}

/// Row-major dense matrix.
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    pub shape: (usize, usize),
    pub data: Vec<f64>,
}

/// Unstructured metadata: JSON, plus the large numeric arrays as typed values.
///
/// In JSON every value is its own heap node, which a Visium full-resolution
/// image in `uns["spatial"]` (~1.2e9 values) cannot afford. Arrays of at
/// least [`UNS_TYPED_MIN`] numbers are therefore read into `arrays`, keyed by
/// their path under uns ("spatial/lib/images/hires"), and are absent from
/// `raw`. Smaller arrays stay in `raw` as (nested) JSON arrays.
#[derive(Debug, Clone, Default)]
pub struct UnsTable {
    pub raw: serde_json::Value,
    pub arrays: BTreeMap<String, UnsArray>,
}

/// Numeric uns arrays with at least this many values are carried typed.
pub const UNS_TYPED_MIN: usize = 1 << 16;

/// A numeric N-d uns array: its shape and row-major values, in their stored type.
#[derive(Debug, Clone, PartialEq)]
pub struct UnsArray {
    pub shape: Vec<usize>,
    pub data: UnsData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnsData {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// `$body` with `$v` bound to the values of `$data` (a `UnsData`), whatever
/// their type.
macro_rules! uns_data_match {
    ($data:expr, $v:ident => $body:expr) => {
        match $data {
            $crate::ir::UnsData::I8($v) => $body,
            $crate::ir::UnsData::I16($v) => $body,
            $crate::ir::UnsData::I32($v) => $body,
            $crate::ir::UnsData::I64($v) => $body,
            $crate::ir::UnsData::U8($v) => $body,
            $crate::ir::UnsData::U16($v) => $body,
            $crate::ir::UnsData::U32($v) => $body,
            $crate::ir::UnsData::U64($v) => $body,
            $crate::ir::UnsData::F32($v) => $body,
            $crate::ir::UnsData::F64($v) => $body,
        }
    };
}
pub(crate) use uns_data_match;

impl UnsData {
    pub fn len(&self) -> usize {
        uns_data_match!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value `i` as f64.
    #[allow(clippy::unnecessary_cast)] // f64 -> f64 in the F64 arm
    pub fn at(&self, i: usize) -> f64 {
        uns_data_match!(self, v => v[i] as f64)
    }

    fn to_json(&self) -> Vec<serde_json::Value> {
        uns_data_match!(self, v => v.iter().map(|&x| serde_json::json!(x)).collect())
    }
}

/// Chunk shape for storing a uns array of `shape`: whole rows along the first
/// axis, about 2^20 values per chunk.
pub(crate) fn uns_chunk(shape: &[usize]) -> Vec<usize> {
    let row: usize = shape.iter().skip(1).product::<usize>().max(1);
    let mut chunk: Vec<usize> = shape.iter().map(|&n| n.max(1)).collect();
    if let Some(first) = chunk.first_mut() {
        *first = (*first).min(((1usize << 20) / row).max(1));
    }
    chunk
}

impl UnsTable {
    /// The top-level keys, in `raw` or holding typed arrays; sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self
            .raw
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        k.extend(
            self.arrays
                .keys()
                .map(|p| p.split('/').next().unwrap_or(p).to_string()),
        );
        k.sort();
        k.dedup();
        k
    }

    /// `raw` with `arrays` put back in as nested JSON arrays. Slow and large
    /// for big arrays: for outputs that only hold JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let mut raw = self.raw.clone();
        for (path, a) in &self.arrays {
            if !raw.is_object() {
                raw = serde_json::Value::Object(Default::default());
            }
            let mut node = &mut raw;
            let mut parts = path.split('/').peekable();
            while let Some(part) = parts.next() {
                let map = match node {
                    serde_json::Value::Object(m) => m,
                    _ => break,
                };
                if parts.peek().is_none() {
                    map.insert(
                        part.to_string(),
                        crate::h5_json::nest(a.data.to_json(), &a.shape),
                    );
                    break;
                }
                node = map
                    .entry(part)
                    .or_insert_with(|| serde_json::Value::Object(Default::default()));
            }
        }
        raw
    }

    /// Move numeric JSON arrays of at least [`UNS_TYPED_MIN`] values out of
    /// `raw` into `arrays`, as the readers of stored formats do, so a table
    /// read from JSON compares like one read from a file.
    pub(crate) fn split_large(&mut self) {
        fn walk(v: &mut serde_json::Value, path: &str, out: &mut BTreeMap<String, UnsArray>) {
            let serde_json::Value::Object(map) = v else {
                return;
            };
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                let p = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}/{k}")
                };
                let typed = match &map[&k] {
                    serde_json::Value::Array(a) => crate::h5_json::numeric_array(a)
                        .filter(|(shape, _)| shape.iter().product::<usize>() >= UNS_TYPED_MIN),
                    _ => None,
                };
                match typed {
                    Some((shape, num)) => {
                        let data = match num {
                            crate::h5_json::Num::I64(v) => UnsData::I64(v),
                            crate::h5_json::Num::F64(v) => UnsData::F64(v),
                        };
                        map.remove(&k);
                        out.insert(p, UnsArray { shape, data });
                    }
                    None => walk(map.get_mut(&k).unwrap(), &p, out),
                }
            }
        }
        walk(&mut self.raw, "", &mut self.arrays);
    }
}

/// Top-level container — used for non-streaming (materialized) datasets.
/// The streaming path uses DatasetReader/DatasetWriter traits instead.
#[derive(Debug, Clone)]
pub struct SingleCellDataset {
    pub x: SparseMatrixCSR,
    pub x_dtype: DataType,
    pub obs: ObsTable,
    pub var: VarTable,
    pub obsm: Embeddings,
    pub uns: UnsTable,
    pub layers: Layers,
    pub obsp: Obsp,
    pub varp: Varp,
    pub varm: Varm,
}

/// Three cells with a categorical of each level type, for the h5ad and Zarr
/// typed-level round trips (#161).
#[cfg(test)]
pub(crate) fn typed_level_obs() -> ObsTable {
    let cat = |levels: &[&str], level_type| ColumnData::Categorical {
        codes: vec![1, 0, 1],
        levels: levels.iter().map(|l| l.to_string()).collect(),
        ordered: false,
        level_type,
    };
    ObsTable {
        index: vec!["c0".into(), "c1".into(), "c2".into()],
        columns: vec![
            Column::new("len", cat(&["517", "1015"], LevelType::Int)),
            Column::new("frac", cat(&["0.5", "1.0"], LevelType::Float)),
            Column::new("flag", cat(&["True", "False"], LevelType::Bool)),
            Column::new("name", cat(&["a", "b"], LevelType::Str)),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columndata_len_is_empty_dtype_str() {
        let cols: [(ColumnData, usize, &str); 6] = [
            (ColumnData::Int(vec![1, 2]), 2, "int32"),
            (ColumnData::Int64(vec![1 << 40]), 1, "int64"),
            (ColumnData::Float(vec![1.0]), 1, "float64"),
            (ColumnData::Bool(vec![true, false, true]), 3, "bool"),
            (ColumnData::String(vec!["a".into()]), 1, "string"),
            (
                ColumnData::Categorical {
                    codes: vec![0, 1],
                    levels: vec!["x".into(), "y".into()],
                    ordered: false,
                    level_type: crate::ir::LevelType::Str,
                },
                2,
                "categorical",
            ),
        ];
        for (col, len, dtype) in &cols {
            assert_eq!(col.len(), *len);
            assert_eq!(col.is_empty(), *len == 0);
            assert_eq!(col.dtype_str(), *dtype);
        }
        assert!(ColumnData::Int(vec![]).is_empty());
    }

    /// to_json puts typed arrays back as nested JSON, creating the objects
    /// on their path; split_large takes them out again (ints as I64, floats
    /// as F64) and leaves small arrays in raw.
    #[test]
    fn uns_json_round_trip_and_keys() {
        let n = UNS_TYPED_MIN;
        let mut t = UnsTable {
            raw: serde_json::Value::Null,
            arrays: [
                (
                    "img/hires".to_string(),
                    UnsArray {
                        shape: vec![n / 2, 2],
                        data: UnsData::U16((0..n).map(|i| i as u16).collect()),
                    },
                ),
                (
                    "w".to_string(),
                    UnsArray {
                        shape: vec![n],
                        data: UnsData::F32(vec![0.5; n]),
                    },
                ),
            ]
            .into(),
        };
        let json = t.to_json();
        assert_eq!(json["img"]["hires"][3], serde_json::json!([6, 7]));
        assert_eq!(json["w"][0], 0.5);
        assert_eq!(t.keys(), ["img", "w"]);

        t.raw = serde_json::json!({"small": [1, 2], "x": 1});
        let mut back = UnsTable {
            raw: t.to_json(),
            ..Default::default()
        };
        back.split_large();
        assert_eq!(
            back.raw,
            serde_json::json!({"img": {}, "small": [1, 2], "x": 1})
        );
        let hires = &back.arrays["img/hires"];
        assert_eq!(hires.shape, [n / 2, 2]);
        assert_eq!(hires.data, UnsData::I64((0..n as i64).collect()));
        assert_eq!(back.arrays["w"].data, UnsData::F64(vec![0.5; n]));
        assert_eq!((hires.data.len(), hires.data.at(9)), (n, 9.0));
        assert!(!hires.data.is_empty());
    }

    #[test]
    fn uns_chunks_are_whole_rows_of_about_a_million_values() {
        assert_eq!(uns_chunk(&[19587, 20330, 3]), [17, 20330, 3]);
        assert_eq!(uns_chunk(&[10, 3]), [10, 3]);
        assert_eq!(uns_chunk(&[3_000_000]), [1 << 20]);
        assert_eq!(uns_chunk(&[0, 4]), [1, 4]);
        assert!(uns_chunk(&[]).is_empty());
    }
}
