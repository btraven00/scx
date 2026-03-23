use std::collections::HashMap;

use crate::dtype::{DataType, TypedVec};

/// Sparse matrix in CSR (Compressed Sparse Row) format.
/// Row-major — natural for cell-chunked streaming and AnnData output.
#[derive(Debug, Clone)]
pub struct SparseMatrixCSR {
    pub shape: (usize, usize),
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub data: TypedVec,
}

/// Sparse matrix in CSC (Compressed Sparse Column) format.
/// Column-major — used by H5Seurat (dgCMatrix storage).
#[derive(Debug, Clone)]
pub struct SparseMatrixCSC {
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
}

#[derive(Debug, Clone)]
pub enum ColumnData {
    Int(Vec<i32>),
    Float(Vec<f64>),
    String(Vec<String>),
    Bool(Vec<bool>),
    Categorical {
        codes: Vec<u32>,
        levels: Vec<String>,
    },
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int(v) => v.len(),
            ColumnData::Float(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Categorical { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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

/// Opaque unstructured metadata.
#[derive(Debug, Clone, Default)]
pub struct UnsTable {
    pub raw: serde_json::Value,
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
