use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    dtype::DataType,
    error::{Result, ScxError},
    ir::ColumnData,
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

pub(super) fn slot_matches(filter: &str, target: &str) -> bool {
    filter == target || target.starts_with(&format!("{filter}:"))
}

// ---------------------------------------------------------------------------
// Rich meta.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct SparseArrayMeta {
    pub(super) shape: [usize; 2],
    pub(super) nnz: usize,
    pub(super) dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct DenseArrayMeta {
    pub(super) shape: [usize; 2],
    pub(super) dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct IndexMeta {
    pub(super) n: usize,
}

/// Column metadata entry — stored as an array in meta.json to preserve order.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct ColumnMeta {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) shape: [usize; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) n_levels: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Meta {
    pub(super) scxd_version: String,
    pub(super) n_obs: usize,
    pub(super) n_vars: usize,
    #[serde(rename = "X", skip_serializing_if = "Option::is_none")]
    pub(super) x: Option<SparseArrayMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) obs_index: Option<IndexMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) var_index: Option<IndexMeta>,
    /// Ordered list — preserves IR column order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) obs: Vec<ColumnMeta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) var: Vec<ColumnMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub(super) obsm: HashMap<String, DenseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub(super) varm: HashMap<String, DenseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub(super) layers: HashMap<String, SparseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub(super) obsp: HashMap<String, SparseArrayMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub(super) varp: HashMap<String, SparseArrayMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) uns: Option<bool>,
}

pub(super) fn dtype_str(dt: DataType) -> &'static str {
    match dt {
        DataType::F32 => "f32",
        DataType::F64 => "f64",
        DataType::I32 => "i32",
        DataType::U32 => "u32",
    }
}

pub(super) fn parse_dtype(s: &str) -> Result<DataType> {
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

pub(super) fn col_kind(data: &ColumnData) -> &'static str {
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

pub(super) fn x_dir(root: &Path) -> PathBuf {
    root.join("X")
}
pub(super) fn obs_dir(root: &Path) -> PathBuf {
    root.join("obs")
}
pub(super) fn var_dir(root: &Path) -> PathBuf {
    root.join("var")
}
pub(super) fn obsm_dir(root: &Path) -> PathBuf {
    root.join("obsm")
}
pub(super) fn varm_dir(root: &Path) -> PathBuf {
    root.join("varm")
}
pub(super) fn layers_key_dir(root: &Path, k: &str) -> PathBuf {
    root.join("layers").join(k)
}
pub(super) fn obsp_key_dir(root: &Path, k: &str) -> PathBuf {
    root.join("obsp").join(k)
}
pub(super) fn varp_key_dir(root: &Path, k: &str) -> PathBuf {
    root.join("varp").join(k)
}
