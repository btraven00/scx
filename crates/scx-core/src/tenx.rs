use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::{self, Stream};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5::File;
use ndarray::s;

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta,
    UnsTable, VarTable, Varm,
};
use crate::stream::DatasetReader;

/// Compact summary of 10x-specific metadata that doesn't fit cleanly into the
/// generic obs/var inspect output: feature-type histogram and unique genome(s).
/// Returned fields are empty/None when the source datasets are absent.
pub struct TenxSummary {
    /// Counts per feature type, sorted by count desc then name.
    pub feature_types: Vec<(String, usize)>,
    /// Unique genome names (deduped, original order).
    pub genomes: Vec<String>,
}

pub fn read_tenx_summary(path: &Path) -> Result<TenxSummary> {
    let file = File::open(path)?;

    let feature_types = if let Ok(ds) = file.dataset("matrix/features/feature_type") {
        let strings = read_str_dataset_raw(&ds)?;
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for s in strings {
            *counts.entry(s).or_insert(0) += 1;
        }
        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        pairs
    } else {
        Vec::new()
    };

    let genomes = if let Ok(ds) = file.dataset("matrix/features/genome") {
        let strings = read_str_dataset_raw(&ds)?;
        let mut seen: Vec<String> = Vec::new();
        for s in strings {
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen
    } else {
        Vec::new()
    };

    Ok(TenxSummary {
        feature_types,
        genomes,
    })
}

fn read_str_dataset_raw(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    crate::h5_str::read_str_1d(ds)
}

/// Datasets that can carry feature ids, most canonical first.
///
/// Cell Ranger v3+ writes a `/matrix/features` **group**; v2 wrote a flat
/// `/matrix/genes` **dataset**. `features/name` is the middle entry because some
/// downstream writers preserve only gene symbols.
const FEATURE_ID_PATHS: [&str; 3] = [
    "matrix/features/id",
    "matrix/features/name",
    "matrix/genes",
];

/// First feature-id dataset present in `file`, or `None` for a non-10x layout.
fn feature_id_path(file: &File) -> Option<&'static str> {
    FEATURE_ID_PATHS
        .into_iter()
        .find(|p| file.dataset(p).is_ok())
}

fn missing_features() -> ScxError {
    ScxError::InvalidFormat(format!(
        "missing feature ids — none of {}",
        FEATURE_ID_PATHS.join(", ")
    ))
}

// ---------------------------------------------------------------------------
// Plain / unrecognized HDF5
// ---------------------------------------------------------------------------

/// A node in the HDF5 file tree.
pub struct H5Node {
    pub name: String,
    pub kind: H5NodeKind,
}

pub enum H5NodeKind {
    Dataset {
        shape: Vec<usize>,
        dtype: String,
    },
    Group {
        children: Vec<H5Node>,
        /// Number of children that were omitted due to depth limit.
        truncated: usize,
    },
}

/// Walk the root of an HDF5 file up to `max_depth` levels deep.
pub fn walk_h5(path: &Path, max_depth: usize) -> Result<Vec<H5Node>> {
    let file = File::open(path)?;
    let root = file
        .group("/")
        .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
    walk_group(&file, &root, max_depth)
}

fn walk_group(file: &File, grp: &hdf5::Group, depth: usize) -> Result<Vec<H5Node>> {
    let names = grp.member_names().unwrap_or_default();
    let mut nodes = Vec::with_capacity(names.len());

    for name in &names {
        let full_path = {
            let grp_name = grp.name();
            if grp_name == "/" {
                format!("/{name}")
            } else {
                format!("{grp_name}/{name}")
            }
        };

        let is_group = file.group(&full_path).is_ok() && file.dataset(&full_path).is_err();

        let kind = if is_group {
            if depth == 0 {
                H5NodeKind::Group {
                    children: Vec::new(),
                    truncated: file
                        .group(&full_path)
                        .ok()
                        .and_then(|g| g.member_names().ok())
                        .map(|v| v.len())
                        .unwrap_or(0),
                }
            } else {
                let child_grp = file
                    .group(&full_path)
                    .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
                let children = walk_group(file, &child_grp, depth - 1)?;
                H5NodeKind::Group {
                    children,
                    truncated: 0,
                }
            }
        } else {
            match file.dataset(&full_path) {
                Ok(ds) => {
                    let shape = ds.shape();
                    let dtype = dtype_str(&ds);
                    H5NodeKind::Dataset { shape, dtype }
                }
                Err(_) => continue,
            }
        };

        nodes.push(H5Node {
            name: name.clone(),
            kind,
        });
    }

    nodes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(nodes)
}

fn dtype_str(ds: &hdf5::Dataset) -> String {
    match ds.dtype().and_then(|d| d.to_descriptor()) {
        Ok(TypeDescriptor::Float(s)) => format!("f{}", (s as usize) * 8),
        Ok(TypeDescriptor::Integer(s)) => format!("i{}", (s as usize) * 8),
        Ok(TypeDescriptor::Unsigned(s)) => format!("u{}", (s as usize) * 8),
        Ok(TypeDescriptor::Boolean) => "bool".into(),
        Ok(TypeDescriptor::VarLenUnicode) => "str".into(),
        Ok(TypeDescriptor::VarLenAscii) => "str".into(),
        Ok(TypeDescriptor::FixedAscii(n)) => format!("str[{n}]"),
        Ok(TypeDescriptor::FixedUnicode(n)) => format!("str[{n}]"),
        _ => "?".into(),
    }
}

// ---------------------------------------------------------------------------
// 10x HDF5 streaming reader
// ---------------------------------------------------------------------------
//
// Cell Ranger stores the feature-barcode matrix under `/matrix` as a sparse
// "CSC of features × barcodes" — but `indptr` is per-barcode (length
// n_barcodes+1) and `indices` are feature indices. From SCX's (obs × var)
// perspective that is exactly CSR with row=cell, col=gene, so no transpose
// is needed: `indptr` maps directly to `x_indptr`, `indices` to var indices.

/// Streaming reader for 10x Genomics HDF5 (`*.h5`) feature-barcode matrices.
///
/// Layout: <https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices>
///
/// Maps:
///   /matrix/barcodes          → obs.index
///   /matrix/features/id       → var.index
///   /matrix/features/name     → var["gene_symbols"]
///   /matrix/features/feature_type → var["feature_types"]
///   /matrix/features/genome   → var["genome"]
///   /matrix/{data,indices,indptr} → CSR X (cells × features)
///
/// Cell Ranger v2 predates the `features` group and stores `/matrix/genes` (ids)
/// and `/matrix/gene_names` (symbols) as flat datasets; both are accepted, see
/// [`FEATURE_ID_PATHS`].
pub struct TenxH5Reader {
    path: PathBuf,
    n_obs: usize,
    n_vars: usize,
    indptr: Vec<u64>,
    chunk_size: usize,
    dtype: DataType,
}

impl TenxH5Reader {
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        let barcodes_ds = file
            .dataset("matrix/barcodes")
            .map_err(|_| ScxError::InvalidFormat("missing /matrix/barcodes".into()))?;
        let n_obs = barcodes_ds.shape().first().copied().unwrap_or(0);

        let feat_id_ds = file.dataset(feature_id_path(&file).ok_or_else(missing_features)?)?;
        let n_vars = feat_id_ds.shape().first().copied().unwrap_or(0);

        // /matrix/shape (optional sanity check) is [n_features, n_barcodes].
        if let Ok(shape_ds) = file.dataset("matrix/shape") {
            if let Ok(s) = shape_ds.read_1d::<i64>() {
                let v = s.to_vec();
                if v.len() == 2 && (v[0] as usize != n_vars || v[1] as usize != n_obs) {
                    tracing::warn!(
                        "10x /matrix/shape {:?} disagrees with barcodes/features lengths ({n_obs}, {n_vars})",
                        v
                    );
                }
            } else if let Ok(s) = shape_ds.read_1d::<i32>() {
                let v = s.to_vec();
                if v.len() == 2 && (v[0] as usize != n_vars || v[1] as usize != n_obs) {
                    tracing::warn!(
                        "10x /matrix/shape {:?} disagrees with barcodes/features lengths ({n_obs}, {n_vars})",
                        v
                    );
                }
            }
        }

        let indptr = read_int_dataset_u64(&file, "matrix/indptr")?;
        if indptr.len() != n_obs + 1 {
            return Err(ScxError::InvalidFormat(format!(
                "10x /matrix/indptr length {} != n_barcodes+1 {}",
                indptr.len(),
                n_obs + 1
            )));
        }

        let dtype = detect_dtype(&file, "matrix/data")?;

        Ok(Self {
            path,
            n_obs,
            n_vars,
            indptr,
            chunk_size,
            dtype,
        })
    }
}

fn detect_dtype(file: &File, path: &str) -> Result<DataType> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_) => DataType::F64,
        TypeDescriptor::Integer(IntSize::U4) => DataType::I32,
        TypeDescriptor::Integer(IntSize::U8) => DataType::I32,
        TypeDescriptor::Integer(_) => DataType::I32,
        TypeDescriptor::Unsigned(_) => DataType::U32,
        _ => DataType::F32,
    })
}

fn read_int_dataset_u64(file: &File, path: &str) -> Result<Vec<u64>> {
    let ds = file.dataset(path)?;
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(IntSize::U8) => {
            ds.read_1d::<i64>()?.iter().map(|&x| x as u64).collect()
        }
        TypeDescriptor::Integer(_) => ds.read_1d::<i32>()?.iter().map(|&x| x as u64).collect(),
        TypeDescriptor::Unsigned(IntSize::U8) => ds.read_1d::<u64>()?.to_vec(),
        TypeDescriptor::Unsigned(_) => ds.read_1d::<u32>()?.iter().map(|&x| x as u64).collect(),
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "unexpected integer dtype {:?} at {path}",
                other
            )))
        }
    })
}

fn read_str_dataset(file: &File, path: &str) -> Result<Vec<String>> {
    let ds = file.dataset(path)?;
    read_str_dataset_raw(&ds)
}

fn read_tenx_chunk(
    path: &Path,
    indptr: &[u64],
    row_start: usize,
    row_end: usize,
    n_vars: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    let file = File::open(path)?;
    let nrows = row_end - row_start;
    let nnz_start = indptr[row_start] as usize;
    let nnz_end = indptr[row_end] as usize;
    let nnz = nnz_end - nnz_start;

    let indices: Vec<u32> = if nnz > 0 {
        let ds = file.dataset("matrix/indices")?;
        match ds.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(IntSize::U8) => ds
                .read_slice_1d::<i64, _>(s![nnz_start..nnz_end])?
                .iter()
                .map(|&x| x as u32)
                .collect(),
            TypeDescriptor::Integer(_) => ds
                .read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?
                .iter()
                .map(|&x| x as u32)
                .collect(),
            TypeDescriptor::Unsigned(_) => {
                ds.read_slice_1d::<u32, _>(s![nnz_start..nnz_end])?.to_vec()
            }
            other => {
                return Err(ScxError::InvalidFormat(format!(
                    "unexpected /matrix/indices dtype {:?}",
                    other
                )))
            }
        }
    } else {
        Vec::new()
    };

    let data: TypedVec = if nnz > 0 {
        let ds = file.dataset("matrix/data")?;
        let descr = ds.dtype()?.to_descriptor()?;
        match (dtype, descr) {
            (DataType::F32, TypeDescriptor::Float(_)) => {
                TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            (DataType::F64, _) => {
                TypedVec::F64(ds.read_slice_1d::<f64, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            (DataType::I32, TypeDescriptor::Integer(IntSize::U8)) => TypedVec::I32(
                ds.read_slice_1d::<i64, _>(s![nnz_start..nnz_end])?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
            ),
            (DataType::I32, _) => {
                TypedVec::I32(ds.read_slice_1d::<i32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            (DataType::U32, _) => {
                TypedVec::U32(ds.read_slice_1d::<u32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
            (DataType::F32, _) => {
                TypedVec::F32(ds.read_slice_1d::<f32, _>(s![nnz_start..nnz_end])?.to_vec())
            }
        }
    } else {
        match dtype {
            DataType::F32 => TypedVec::F32(Vec::new()),
            DataType::F64 => TypedVec::F64(Vec::new()),
            DataType::I32 => TypedVec::I32(Vec::new()),
            DataType::U32 => TypedVec::U32(Vec::new()),
        }
    };

    let chunk_indptr: Vec<u64> = indptr[row_start..=row_end]
        .iter()
        .map(|&p| p - indptr[row_start])
        .collect();

    Ok(MatrixChunk {
        row_offset: row_start,
        nrows,
        data: SparseMatrixCSR {
            shape: (nrows, n_vars),
            indptr: chunk_indptr,
            indices,
            data,
        },
    })
}

#[async_trait]
impl DatasetReader for TenxH5Reader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn x_indptr(&self) -> &[u64] {
        &self.indptr
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        let file = File::open(&self.path)?;
        let index = read_str_dataset(&file, "matrix/barcodes")?;
        Ok(ObsTable {
            index,
            columns: Vec::new(),
        })
    }

    async fn var(&mut self) -> Result<VarTable> {
        let file = File::open(&self.path)?;
        let index = read_str_dataset(&file, feature_id_path(&file).ok_or_else(missing_features)?)?;
        let mut columns: Vec<Column> = Vec::new();

        // `matrix/gene_names` is the v2 spelling of `features/name`; the two are
        // mutually exclusive in practice, and the dedup below covers the case
        // where a converter wrote both.
        for (h5_name, col_name) in &[
            ("matrix/features/name", "gene_symbols"),
            ("matrix/gene_names", "gene_symbols"),
            ("matrix/features/feature_type", "feature_types"),
            ("matrix/features/genome", "genome"),
        ] {
            if columns.iter().any(|c| c.name == *col_name) {
                continue;
            }
            if let Ok(ds) = file.dataset(h5_name) {
                match read_str_dataset_raw(&ds) {
                    Ok(v) if !v.is_empty() => columns.push(Column {
                        name: (*col_name).to_string(),
                        data: ColumnData::String(v),
                    }),
                    Ok(_) => {}
                    Err(e) => tracing::warn!("skipping var column '{col_name}': {e}"),
                }
            }
        }

        Ok(VarTable { index, columns })
    }

    async fn obsm(&mut self) -> Result<Embeddings> {
        Ok(Embeddings::default())
    }

    async fn uns(&mut self) -> Result<UnsTable> {
        Ok(UnsTable::default())
    }

    async fn varm(&mut self) -> Result<Varm> {
        Ok(Varm::default())
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(Vec::new())
    }

    fn layer_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        Box::pin(stream::empty())
    }

    fn obsp_stream<'a>(
        &'a self,
        _meta: &'a SparseMatrixMeta,
        _chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + 'a>> {
        Box::pin(stream::empty())
    }

    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let path = self.path.clone();
        let indptr = self.indptr.clone();
        let n_obs = self.n_obs;
        let n_vars = self.n_vars;
        let chunk_size = self.chunk_size;
        let dtype = self.dtype;

        Box::pin(stream::unfold(0usize, move |row_start| {
            let path = path.clone();
            let indptr = indptr.clone();
            async move {
                if row_start >= n_obs {
                    return None;
                }
                let row_end = (row_start + chunk_size).min(n_obs);
                let chunk = read_tenx_chunk(&path, &indptr, row_start, row_end, n_vars, dtype);
                Some((chunk, row_end))
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use futures::StreamExt;
    use hdf5::types::{FixedAscii, VarLenUnicode};
    use hdf5::{File, Group};
    use ndarray::Array1;
    use std::str::FromStr;

    // ── fixture helpers ──────────────────────────────────────────────────────

    fn wstr(g: &Group, name: &str, vals: &[&str]) {
        let arr: Vec<VarLenUnicode> = vals
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        g.new_dataset::<VarLenUnicode>()
            .shape(arr.len())
            .create(name)
            .unwrap()
            .write(&Array1::from_vec(arr))
            .unwrap();
    }

    /// Fixed-length ASCII, i.e. `|S{N}` — what `rhdf5` / `HDF5Array` write by
    /// default, so what every R-produced 10x file contains.  The whole test corpus
    /// used `wstr` (variable-length), which is why the reader returning empty
    /// vectors for these went unnoticed.
    fn wstr_fixed<const N: usize>(g: &Group, name: &str, vals: &[&str]) {
        let arr: Vec<FixedAscii<N>> = vals
            .iter()
            .map(|s| FixedAscii::from_ascii(s).unwrap())
            .collect();
        g.new_dataset::<FixedAscii<N>>()
            .shape(arr.len())
            .create(name)
            .unwrap()
            .write(&Array1::from_vec(arr))
            .unwrap();
    }

    fn wi32(g: &Group, name: &str, vals: &[i32]) {
        g.new_dataset::<i32>()
            .shape(vals.len())
            .create(name)
            .unwrap()
            .write(&Array1::from_vec(vals.to_vec()))
            .unwrap();
    }

    fn wi64(g: &Group, name: &str, vals: &[i64]) {
        g.new_dataset::<i64>()
            .shape(vals.len())
            .create(name)
            .unwrap()
            .write(&Array1::from_vec(vals.to_vec()))
            .unwrap();
    }

    fn wu32(g: &Group, name: &str, vals: &[u32]) {
        g.new_dataset::<u32>()
            .shape(vals.len())
            .create(name)
            .unwrap()
            .write(&Array1::from_vec(vals.to_vec()))
            .unwrap();
    }

    fn wf32(g: &Group, name: &str, vals: &[f32]) {
        g.new_dataset::<f32>()
            .shape(vals.len())
            .create(name)
            .unwrap()
            .write(&Array1::from_vec(vals.to_vec()))
            .unwrap();
    }

    /// Canonical 4-cell × 3-gene fixture. CSR (cell-major):
    ///   cell0: g0=5, g2=7 | cell1: g1=3 | cell2: (empty) | cell3: g0=1,g1=2,g2=4
    fn write_standard(path: &Path) {
        let f = File::create(path).unwrap();
        let m = f.create_group("matrix").unwrap();
        wstr(&m, "barcodes", &["AAA", "CCC", "GGG", "TTT"]);
        let feat = m.create_group("features").unwrap();
        wstr(&feat, "id", &["ENSG0", "ENSG1", "ENSG2"]);
        wstr(&feat, "name", &["GeneA", "GeneB", "GeneC"]);
        wstr(
            &feat,
            "feature_type",
            &["Gene Expression", "Gene Expression", "Antibody Capture"],
        );
        wstr(&feat, "genome", &["GRCh38", "GRCh38", "GRCh38"]);
        wi32(&m, "data", &[5, 7, 3, 1, 2, 4]);
        wi32(&m, "indices", &[0, 2, 1, 0, 1, 2]);
        wi32(&m, "indptr", &[0, 2, 3, 3, 6]);
        wi32(&m, "shape", &[3, 4]);
    }

    /// Drive `x_stream` and reassemble the global CSR (indptr, indices, data-as-i64).
    fn collect_csr(path: &Path, chunk_size: usize) -> (Vec<u64>, Vec<u32>, Vec<i64>) {
        let mut reader = TenxH5Reader::open(path, chunk_size).unwrap();
        let chunks: Vec<MatrixChunk> = block_on(async {
            let mut s = reader.x_stream();
            let mut v = Vec::new();
            while let Some(c) = s.next().await {
                v.push(c.unwrap());
            }
            v
        });
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for (i, ch) in chunks.iter().enumerate() {
            assert_eq!(ch.row_offset, indptr.len() - 1, "row_offset chunk {i}");
            let base = *indptr.last().unwrap();
            for r in 1..=ch.nrows {
                indptr.push(base + ch.data.indptr[r]);
            }
            indices.extend_from_slice(&ch.data.indices);
            data.extend(ch.data.data.to_f64().into_iter().map(|x| x as i64));
        }
        (indptr, indices, data)
    }

    fn str_col<'a>(var: &'a VarTable, name: &str) -> Option<&'a Vec<String>> {
        var.columns.iter().find(|c| c.name == name).and_then(|c| {
            if let ColumnData::String(v) = &c.data {
                Some(v)
            } else {
                None
            }
        })
    }

    // ── tests ────────────────────────────────────────────────────────────────

    #[test]
    fn open_reads_shape_dtype_indptr() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("std.h5");
        write_standard(&p);
        let r = TenxH5Reader::open(&p, 2).unwrap();
        assert_eq!(r.shape(), (4, 3));
        assert_eq!(r.dtype(), DataType::I32);
        assert_eq!(r.x_indptr(), &[0, 2, 3, 3, 6]);
    }

    #[test]
    fn obs_and_var_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("std.h5");
        write_standard(&p);
        let mut r = TenxH5Reader::open(&p, 2).unwrap();

        let obs = block_on(r.obs()).unwrap();
        assert_eq!(obs.index, ["AAA", "CCC", "GGG", "TTT"]);
        assert!(obs.columns.is_empty());

        let var = block_on(r.var()).unwrap();
        assert_eq!(var.index, ["ENSG0", "ENSG1", "ENSG2"]); // prefers features/id
        assert_eq!(
            str_col(&var, "gene_symbols").unwrap(),
            &["GeneA", "GeneB", "GeneC"]
        );
        assert_eq!(
            str_col(&var, "feature_types").unwrap(),
            &["Gene Expression", "Gene Expression", "Antibody Capture"]
        );
        assert_eq!(str_col(&var, "genome").unwrap(), &["GRCh38"; 3]);
    }

    /// Regression for the silent-empty bug: a 10x file whose barcodes and feature
    /// ids are fixed-length ASCII used to yield `obs().index == []` with `Ok(())`,
    /// so the caller only found out by panicking on an out-of-bounds index far
    /// downstream.  Widths are off the ladder rungs on purpose — HDF5 widens.
    #[test]
    fn fixed_length_strings_are_not_silently_empty() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("fixed.h5");
        {
            let f = File::create(&p).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr_fixed::<18>(&m, "barcodes", &["AAA", "CCC", "GGG", "TTT"]);
            let feat = m.create_group("features").unwrap();
            wstr_fixed::<13>(&feat, "id", &["ENSG0", "ENSG1", "ENSG2"]);
            wstr_fixed::<13>(&feat, "name", &["GeneA", "GeneB", "GeneC"]);
            wi32(&m, "data", &[5, 7, 3, 1, 2, 4]);
            wi32(&m, "indices", &[0, 2, 1, 0, 1, 2]);
            wi32(&m, "indptr", &[0, 2, 3, 3, 6]);
            wi32(&m, "shape", &[3, 4]);
        }
        let mut r = TenxH5Reader::open(&p, 2).unwrap();

        let obs = block_on(r.obs()).unwrap();
        assert_eq!(
            obs.index.len(),
            4,
            "fixed-length barcodes must read, not vanish"
        );
        assert_eq!(obs.index, ["AAA", "CCC", "GGG", "TTT"]);

        let var = block_on(r.var()).unwrap();
        assert_eq!(var.index, ["ENSG0", "ENSG1", "ENSG2"]);
        assert_eq!(
            str_col(&var, "gene_symbols").unwrap(),
            &["GeneA", "GeneB", "GeneC"]
        );
    }

    /// Cell Ranger v2 layout: flat `/matrix/genes` + `/matrix/gene_names` instead
    /// of a `/matrix/features` group. Previously `open()` bailed with "missing
    /// /matrix/features/id and /matrix/features/name" and `detect()` never even
    /// classified the file as 10x.
    #[test]
    fn cellranger_v2_genes_layout() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v2.h5");
        {
            let f = File::create(&p).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr(&m, "barcodes", &["AAA", "CCC", "GGG", "TTT"]);
            wstr(&m, "genes", &["ENSG0", "ENSG1", "ENSG2"]);
            wstr(&m, "gene_names", &["GeneA", "GeneB", "GeneC"]);
            wi32(&m, "data", &[5, 7, 3, 1, 2, 4]);
            wi32(&m, "indices", &[0, 2, 1, 0, 1, 2]);
            wi32(&m, "indptr", &[0, 2, 3, 3, 6]);
            wi32(&m, "shape", &[3, 4]);
        }

        assert_eq!(
            crate::detect::detect(&p),
            Some(crate::detect::Format::TenxH5),
            "v2 files must be recognised as 10x, not fall through to PlainH5"
        );

        let mut r = TenxH5Reader::open(&p, 2).unwrap();
        assert_eq!(r.n_vars, 3);
        assert_eq!(r.n_obs, 4);

        let var = block_on(r.var()).unwrap();
        assert_eq!(var.index, ["ENSG0", "ENSG1", "ENSG2"]);
        assert_eq!(
            str_col(&var, "gene_symbols").unwrap(),
            &["GeneA", "GeneB", "GeneC"],
            "v2 gene_names maps onto the same gene_symbols column as v3 features/name"
        );
    }

    /// The v2 spelling must not produce a duplicate column when a converter has
    /// written both `features/name` and `gene_names`.
    #[test]
    fn v3_and_v2_symbol_datasets_do_not_duplicate_column() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("both.h5");
        {
            let f = File::create(&p).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr(&m, "barcodes", &["AAA", "CCC", "GGG", "TTT"]);
            wstr(&m, "gene_names", &["OldA", "OldB", "OldC"]);
            let feat = m.create_group("features").unwrap();
            wstr(&feat, "id", &["ENSG0", "ENSG1", "ENSG2"]);
            wstr(&feat, "name", &["GeneA", "GeneB", "GeneC"]);
            wi32(&m, "data", &[5, 7, 3, 1, 2, 4]);
            wi32(&m, "indices", &[0, 2, 1, 0, 1, 2]);
            wi32(&m, "indptr", &[0, 2, 3, 3, 6]);
            wi32(&m, "shape", &[3, 4]);
        }
        let mut r = TenxH5Reader::open(&p, 2).unwrap();
        let var = block_on(r.var()).unwrap();
        assert_eq!(
            var.columns.iter().filter(|c| c.name == "gene_symbols").count(),
            1
        );
        assert_eq!(
            str_col(&var, "gene_symbols").unwrap(),
            &["GeneA", "GeneB", "GeneC"],
            "the v3 spelling wins when both are present"
        );
    }

    #[test]
    fn x_stream_roundtrip_all_chunk_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("std.h5");
        write_standard(&p);
        for cs in [1usize, 2, 3, 4, 10] {
            let (indptr, indices, data) = collect_csr(&p, cs);
            assert_eq!(indptr, vec![0, 2, 3, 3, 6], "indptr cs={cs}");
            assert_eq!(indices, vec![0, 2, 1, 0, 1, 2], "indices cs={cs}");
            assert_eq!(data, vec![5, 7, 3, 1, 2, 4], "data cs={cs}");
        }
    }

    #[test]
    fn summary_feature_types_and_genomes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("std.h5");
        write_standard(&p);
        let s = read_tenx_summary(&p).unwrap();
        // sorted by count desc, then name asc
        assert_eq!(
            s.feature_types,
            vec![
                ("Gene Expression".to_string(), 2),
                ("Antibody Capture".to_string(), 1),
            ]
        );
        assert_eq!(s.genomes, vec!["GRCh38".to_string()]);
    }

    #[test]
    fn walk_h5_tree_and_depth_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("std.h5");
        write_standard(&p);

        let nodes = walk_h5(&p, 3).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "matrix");
        match &nodes[0].kind {
            H5NodeKind::Group {
                children,
                truncated,
            } => {
                assert_eq!(*truncated, 0);
                let names: Vec<&str> = children.iter().map(|n| n.name.as_str()).collect();
                for expected in ["barcodes", "data", "features", "indices", "indptr", "shape"] {
                    assert!(names.contains(&expected), "missing {expected}: {names:?}");
                }
            }
            _ => panic!("matrix should be a group"),
        }

        // depth 0: the matrix group's children are omitted but counted.
        let shallow = walk_h5(&p, 0).unwrap();
        match &shallow[0].kind {
            H5NodeKind::Group {
                children,
                truncated,
            } => {
                assert!(children.is_empty());
                assert!(*truncated > 0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn open_missing_barcodes_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("nobc.h5");
        {
            let f = File::create(&p).unwrap();
            f.create_group("matrix").unwrap();
        }
        assert!(matches!(
            TenxH5Reader::open(&p, 2),
            Err(ScxError::InvalidFormat(_))
        ));
    }

    #[test]
    fn open_indptr_length_mismatch_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("badptr.h5");
        {
            let f = File::create(&p).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr(&m, "barcodes", &["AAA", "CCC", "GGG", "TTT"]);
            let feat = m.create_group("features").unwrap();
            wstr(&feat, "id", &["g0", "g1"]);
            wi32(&m, "data", &[1]);
            wi32(&m, "indices", &[0]);
            wi32(&m, "indptr", &[0, 1]); // wrong: needs n_obs+1 = 5
        }
        assert!(matches!(
            TenxH5Reader::open(&p, 2),
            Err(ScxError::InvalidFormat(_))
        ));
    }

    #[test]
    fn features_name_fallback_when_no_id() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("nameonly.h5");
        {
            let f = File::create(&p).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr(&m, "barcodes", &["b0", "b1"]);
            let feat = m.create_group("features").unwrap();
            wstr(&feat, "name", &["GeneA", "GeneB"]); // no `id`
            wi32(&m, "data", &[1, 2]);
            wi32(&m, "indices", &[0, 1]);
            wi32(&m, "indptr", &[0, 1, 2]);
        }
        let mut r = TenxH5Reader::open(&p, 2).unwrap();
        assert_eq!(r.shape(), (2, 2));
        let var = block_on(r.var()).unwrap();
        assert_eq!(var.index, ["GeneA", "GeneB"]); // falls back to name
    }

    #[test]
    fn dtype_detection_float_and_uint_and_i64_indptr() {
        let dir = tempfile::tempdir().unwrap();

        // float32 data + i64 indptr/indices (exercises read_int_dataset_u64 i64 path).
        let pf = dir.path().join("float.h5");
        {
            let f = File::create(&pf).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr(&m, "barcodes", &["b0", "b1"]);
            let feat = m.create_group("features").unwrap();
            wstr(&feat, "id", &["g0", "g1"]);
            wf32(&m, "data", &[1.5, 2.5]);
            wi64(&m, "indices", &[0, 1]);
            wi64(&m, "indptr", &[0, 1, 2]);
        }
        let r = TenxH5Reader::open(&pf, 8).unwrap();
        assert_eq!(r.dtype(), DataType::F32);
        assert_eq!(r.x_indptr(), &[0, 1, 2]);
        let (_, idx, data) = collect_csr(&pf, 8);
        assert_eq!(idx, vec![0, 1]);
        assert_eq!(data, vec![1, 2]); // 1.5,2.5 truncated to i64 in the helper

        // uint32 data.
        let pu = dir.path().join("uint.h5");
        {
            let f = File::create(&pu).unwrap();
            let m = f.create_group("matrix").unwrap();
            wstr(&m, "barcodes", &["b0", "b1"]);
            let feat = m.create_group("features").unwrap();
            wstr(&feat, "id", &["g0", "g1"]);
            wu32(&m, "data", &[10, 20]);
            wi32(&m, "indices", &[0, 1]);
            wi32(&m, "indptr", &[0, 1, 2]);
        }
        let r = TenxH5Reader::open(&pu, 8).unwrap();
        assert_eq!(r.dtype(), DataType::U32);
    }
}
