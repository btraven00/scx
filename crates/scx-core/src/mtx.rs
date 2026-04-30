//! Matrix Market Exchange (.mtx / .mtx.gz) reader.
//!
//! Supports the 10x Genomics directory layout:
//!   matrix.mtx[.gz] + barcodes.tsv[.gz] + features.tsv[.gz] | genes.tsv[.gz]
//! and standalone .mtx[.gz] files (barcodes/features synthesised as integer indices).
//!
//! Cell Ranger writes a genes × cells matrix with entries sorted by column (cell).
//! This reader transposes on the fly to cells × genes (AnnData convention).
//!
//! ## File-name search order (first match wins)
//!
//! | Slot     | Candidates                                                    |
//! |----------|---------------------------------------------------------------|
//! | matrix   | `matrix.mtx.gz`, `matrix.mtx`                                |
//! | barcodes | `barcodes.tsv.gz`, `barcodes.tsv`                            |
//! | features | `features.tsv.gz`, `features.tsv`, `genes.tsv.gz`, `genes.tsv` |
//!
//! `features.tsv` (Cell Ranger v3+) has three tab-separated columns: id, name, feature_type.
//! `genes.tsv` (Cell Ranger v2) has two columns: id, name; feature_type defaults to "Gene Expression".

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use flate2::read::GzDecoder;
use futures::{stream, Stream};

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta,
    UnsTable, VarTable, Varm,
};
use crate::stream::DatasetReader;

// ─── File-name search order ──────────────────────────────────────────────────

const MTX_NAMES: &[&str] = &["matrix.mtx.gz", "matrix.mtx"];
const BARCODES_NAMES: &[&str] = &["barcodes.tsv.gz", "barcodes.tsv"];
const FEATURES_NAMES: &[&str] = &[
    "features.tsv.gz",
    "features.tsv",
    "genes.tsv.gz",
    "genes.tsv",
];

// ─── Public info type (for Inspect) ─────────────────────────────────────────

pub struct MtxInfo {
    /// Number of genes (rows in the MTX file, cols in the output).
    pub n_genes: usize,
    /// Number of cells (cols in the MTX file, rows in the output).
    pub n_cells: usize,
    pub nnz: usize,
    /// `"integer"` or `"real"`.
    pub value_type: &'static str,
    pub software_version: Option<String>,
    pub format_version: Option<u32>,
}

/// Read only the MTX header — fast, does not parse coordinate entries.
pub fn read_mtx_info(path: &Path) -> Result<MtxInfo> {
    let reader = open_reader(path)?;
    let (info, _dtype, _reader) = parse_header(reader)?;
    Ok(info)
}

// ─── Reader ─────────────────────────────────────────────────────────────────

pub struct TenxMtxReader {
    x: SparseMatrixCSR,
    x_dtype: DataType,
    obs: ObsTable,
    var: VarTable,
    chunk_size: usize,
}

impl TenxMtxReader {
    /// Open a 10x MTX directory or a standalone `.mtx[.gz]` file.
    pub fn open(input: &Path, chunk_size: usize) -> Result<Self> {
        let (mtx_path, barcodes_path, features_path) = resolve_paths(input)?;

        let (info, dtype, data_reader) = parse_header(open_reader(&mtx_path)?)?;
        let x = build_csr(data_reader, &info, dtype)?;
        let n_cells = info.n_cells;
        let n_genes = info.n_genes;

        let obs = match barcodes_path {
            Some(p) => ObsTable {
                index: load_lines(&p)?,
                columns: vec![],
            },
            None => ObsTable {
                index: (1..=n_cells).map(|i| i.to_string()).collect(),
                columns: vec![],
            },
        };

        let var = match features_path {
            Some(p) => load_features(&p)?,
            None => VarTable {
                index: (1..=n_genes).map(|i| i.to_string()).collect(),
                columns: vec![],
            },
        };

        Ok(Self {
            x,
            x_dtype: dtype,
            obs,
            var,
            chunk_size,
        })
    }
}

// ─── DatasetReader impl ──────────────────────────────────────────────────────

#[async_trait]
impl DatasetReader for TenxMtxReader {
    fn shape(&self) -> (usize, usize) {
        self.x.shape
    }
    fn dtype(&self) -> DataType {
        self.x_dtype
    }
    async fn obs(&mut self) -> Result<ObsTable> {
        Ok(self.obs.clone())
    }
    async fn var(&mut self) -> Result<VarTable> {
        Ok(self.var.clone())
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
        Ok(vec![])
    }
    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        Ok(vec![])
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
    fn x_indptr(&self) -> &[u64] {
        &self.x.indptr
    }
    fn x_stream(&mut self) -> Pin<Box<dyn Stream<Item = Result<MatrixChunk>> + Send + '_>> {
        let n_obs = self.x.shape.0;
        let n_vars = self.x.shape.1;
        let chunk_size = self.chunk_size;
        let indptr = Arc::new(std::mem::take(&mut self.x.indptr));
        let indices = Arc::new(std::mem::take(&mut self.x.indices));
        let data = Arc::new(std::mem::replace(
            &mut self.x.data,
            TypedVec::F32(vec![]),
        ));
        Box::pin(stream::unfold(0usize, move |row_start| {
            let indptr = Arc::clone(&indptr);
            let indices = Arc::clone(&indices);
            let data = Arc::clone(&data);
            async move {
                if row_start >= n_obs {
                    return None;
                }
                let row_end = (row_start + chunk_size).min(n_obs);
                let nnz_start = indptr[row_start] as usize;
                let nnz_end = indptr[row_end] as usize;
                let nrows = row_end - row_start;
                let chunk_indptr: Vec<u64> = (row_start..=row_end)
                    .map(|i| indptr[i] - indptr[row_start])
                    .collect();
                let chunk_indices = indices[nnz_start..nnz_end].to_vec();
                let chunk_data = match data.as_ref() {
                    TypedVec::F32(v) => TypedVec::F32(v[nnz_start..nnz_end].to_vec()),
                    TypedVec::F64(v) => TypedVec::F64(v[nnz_start..nnz_end].to_vec()),
                    TypedVec::I32(v) => TypedVec::I32(v[nnz_start..nnz_end].to_vec()),
                    TypedVec::U32(v) => TypedVec::U32(v[nnz_start..nnz_end].to_vec()),
                };
                let chunk = MatrixChunk {
                    row_offset: row_start,
                    nrows,
                    data: SparseMatrixCSR {
                        shape: (nrows, n_vars),
                        indptr: chunk_indptr,
                        indices: chunk_indices,
                        data: chunk_data,
                    },
                };
                Some((Ok(chunk), row_end))
            }
        }))
    }
}

// ─── Parsing internals ───────────────────────────────────────────────────────

fn open_reader(path: &Path) -> Result<Box<dyn BufRead + Send>> {
    let file = File::open(path)?;
    let is_gz = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".gz"));
    if is_gz {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Parse the MTX banner, optional metadata comment, and dimension line.
/// Returns `(MtxInfo, DataType, reader-positioned-at-first-data-line)`.
fn parse_header(
    mut reader: Box<dyn BufRead + Send>,
) -> Result<(MtxInfo, DataType, Box<dyn BufRead + Send>)> {
    let mut software_version: Option<String> = None;
    let mut format_version: Option<u32> = None;
    let mut value_type: &'static str = "integer";
    let mut dtype = DataType::I32;
    let mut found_banner = false;
    let mut n_genes = 0usize;
    let mut n_cells = 0usize;
    let mut nnz = 0usize;

    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();

        if trimmed.starts_with("%%MatrixMarket") {
            // %%MatrixMarket matrix coordinate <type> <symmetry>
            let mut parts = trimmed.split_ascii_whitespace().skip(3);
            if let Some(t) = parts.next() {
                if matches!(t, "real" | "double") {
                    value_type = "real";
                    dtype = DataType::F32;
                }
            }
            found_banner = true;
            continue;
        }

        if trimmed.starts_with("%metadata_json:") {
            let json_str = trimmed.strip_prefix("%metadata_json:").unwrap_or("").trim();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                software_version = v["software_version"].as_str().map(str::to_string);
                format_version = v["format_version"].as_u64().map(|n| n as u32);
            }
            continue;
        }

        if trimmed.starts_with('%') {
            continue;
        }

        // First non-comment line: `n_rows n_cols nnz`
        let mut parts = trimmed.split_ascii_whitespace();
        let parse = |s: Option<&str>, field: &str| {
            s.ok_or_else(|| ScxError::InvalidFormat(format!("missing {field} in dimension line")))?
                .parse::<usize>()
                .map_err(|e| ScxError::InvalidFormat(format!("{field} parse error: {e}")))
        };
        n_genes = parse(parts.next(), "n_rows")?;
        n_cells = parse(parts.next(), "n_cols")?;
        nnz = parse(parts.next(), "nnz")?;
        break;
    }

    if !found_banner {
        return Err(ScxError::InvalidFormat(
            "not an MTX file: missing %%MatrixMarket banner".into(),
        ));
    }
    if n_genes == 0 && n_cells == 0 {
        return Err(ScxError::InvalidFormat("missing dimension line".into()));
    }

    let info = MtxInfo {
        n_genes,
        n_cells,
        nnz,
        value_type,
        software_version,
        format_version,
    };
    Ok((info, dtype, reader))
}

/// Build a CSR matrix (cells × genes) from a column-sorted MTX coordinate stream.
///
/// Cell Ranger sorts entries by column (cell index), which maps to output rows —
/// so CSR can be constructed in a single sequential pass without buffering entries.
/// Returns `InvalidFormat` if an out-of-order entry is found.
fn build_csr(
    reader: Box<dyn BufRead + Send>,
    info: &MtxInfo,
    dtype: DataType,
) -> Result<SparseMatrixCSR> {
    let n_cells = info.n_cells;
    let n_genes = info.n_genes;
    let nnz = info.nnz;

    let mut indptr = vec![0u64; n_cells + 1];
    let mut indices: Vec<u32> = Vec::with_capacity(nnz);
    let mut data_i32: Vec<i32> = if dtype == DataType::I32 {
        Vec::with_capacity(nnz)
    } else {
        Vec::new()
    };
    let mut data_f32: Vec<f32> = if dtype == DataType::F32 {
        Vec::with_capacity(nnz)
    } else {
        Vec::new()
    };

    let mut current_cell: usize = 0;

    for line in reader.lines() {
        let line = line.map_err(ScxError::Io)?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_ascii_whitespace();
        let gene_1: usize = parts
            .next()
            .ok_or_else(|| ScxError::InvalidFormat("truncated data line".into()))?
            .parse()
            .map_err(|e| ScxError::InvalidFormat(format!("row index: {e}")))?;
        let cell_1: usize = parts
            .next()
            .ok_or_else(|| ScxError::InvalidFormat("truncated data line".into()))?
            .parse()
            .map_err(|e| ScxError::InvalidFormat(format!("col index: {e}")))?;
        let val_str = parts
            .next()
            .ok_or_else(|| ScxError::InvalidFormat("missing value in data line".into()))?;

        if gene_1 < 1 || gene_1 > n_genes || cell_1 < 1 || cell_1 > n_cells {
            return Err(ScxError::InvalidFormat(format!(
                "index out of range: gene={gene_1} (max {n_genes}), cell={cell_1} (max {n_cells})"
            )));
        }

        let gene_0 = gene_1 - 1;
        let cell_0 = cell_1 - 1;

        if cell_0 < current_cell {
            return Err(ScxError::InvalidFormat(
                "MTX entries must be sorted by column (cell) index; \
                 re-sort or convert via Python/R if your file is unsorted"
                    .into(),
            ));
        }

        // Advance indptr: any cells between current_cell and cell_0 are empty.
        while current_cell < cell_0 {
            current_cell += 1;
            indptr[current_cell] = indices.len() as u64;
        }

        indices.push(gene_0 as u32);
        if dtype == DataType::I32 {
            data_i32.push(
                val_str
                    .parse()
                    .map_err(|e| ScxError::InvalidFormat(format!("value: {e}")))?,
            );
        } else {
            data_f32.push(
                val_str
                    .parse()
                    .map_err(|e| ScxError::InvalidFormat(format!("value: {e}")))?,
            );
        }
    }

    // Fill indptr for any trailing empty cells.
    let final_nnz = indices.len() as u64;
    while current_cell < n_cells {
        current_cell += 1;
        indptr[current_cell] = final_nnz;
    }

    let data = if dtype == DataType::I32 {
        TypedVec::I32(data_i32)
    } else {
        TypedVec::F32(data_f32)
    };

    Ok(SparseMatrixCSR {
        shape: (n_cells, n_genes),
        indptr,
        indices,
        data,
    })
}

// ─── Path helpers ────────────────────────────────────────────────────────────

fn resolve_paths(
    input: &Path,
) -> Result<(PathBuf, Option<PathBuf>, Option<PathBuf>)> {
    if input.is_dir() {
        let mtx = find_first(input, MTX_NAMES).ok_or_else(|| {
            ScxError::InvalidFormat(
                "no matrix.mtx[.gz] found in directory".into(),
            )
        })?;
        let barcodes = find_first(input, BARCODES_NAMES);
        let features = find_first(input, FEATURES_NAMES);
        Ok((mtx, barcodes, features))
    } else {
        Ok((input.to_path_buf(), None, None))
    }
}

fn find_first(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

fn load_lines(path: &Path) -> Result<Vec<String>> {
    open_reader(path)?
        .lines()
        .map(|l| l.map_err(ScxError::Io))
        .collect()
}

/// Load `features.tsv[.gz]` (3 cols: id, name, type) or `genes.tsv[.gz]` (2 cols: id, name).
fn load_features(path: &Path) -> Result<VarTable> {
    let reader = open_reader(path)?;
    let mut ids = Vec::new();
    let mut names = Vec::new();
    let mut types: Vec<String> = Vec::new();
    let mut has_types = false;

    for line in reader.lines() {
        let line = line.map_err(ScxError::Io)?;
        let mut fields = line.split('\t');
        let id = fields.next().unwrap_or("").to_string();
        let name = fields.next().unwrap_or(id.as_str()).to_string();
        if let Some(t) = fields.next() {
            has_types = true;
            types.push(t.to_string());
        } else {
            types.push("Gene Expression".to_string());
        }
        ids.push(id);
        names.push(name);
    }

    let mut columns = vec![Column {
        name: "gene_name".into(),
        data: ColumnData::String(names),
    }];
    if has_types {
        columns.push(Column {
            name: "feature_type".into(),
            data: ColumnData::String(types),
        });
    }

    Ok(VarTable {
        index: ids,
        columns,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::write::GzEncoder;
    use flate2::Compression;
    use futures::StreamExt;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::dtype::DataType;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn write_mtx(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".mtx").unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    fn write_mtx_gz(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".mtx.gz").unwrap();
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(content.as_bytes()).unwrap();
        f.write_all(&enc.finish().unwrap()).unwrap();
        f
    }

    /// A tiny 3×4 (genes×cells) integer matrix:
    ///
    ///   gene\cell   1   2   3   4
    ///   gene 1      5   0   0   0
    ///   gene 2      0   3   0   7
    ///   gene 3      0   0   0   2
    ///
    /// Cell Ranger sorts entries by cell (column), then gene (row) within cell.
    /// After transposition output is 4 cells × 3 genes.
    const TINY_MTX: &str = "\
%%MatrixMarket matrix coordinate integer general
%metadata_json: {\"software_version\": \"cellranger-8.0.0\", \"format_version\": 3}
3 4 4
1 1 5
2 2 3
2 4 7
3 4 2
";

    // ── info / header ─────────────────────────────────────────────────────────

    #[test]
    fn test_read_info_shape() {
        let f = write_mtx(TINY_MTX);
        let info = read_mtx_info(f.path()).unwrap();
        assert_eq!(info.n_genes, 3);
        assert_eq!(info.n_cells, 4);
        assert_eq!(info.nnz, 4);
        assert_eq!(info.value_type, "integer");
        assert_eq!(info.software_version.as_deref(), Some("cellranger-8.0.0"));
        assert_eq!(info.format_version, Some(3));
    }

    #[test]
    fn test_read_info_gz() {
        let f = write_mtx_gz(TINY_MTX);
        let info = read_mtx_info(f.path()).unwrap();
        assert_eq!(info.n_genes, 3);
        assert_eq!(info.n_cells, 4);
        assert_eq!(info.nnz, 4);
    }

    // ── CSR correctness ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_csr_transpose_shape() {
        let f = write_mtx(TINY_MTX);
        let reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        // output is cells × genes
        assert_eq!(reader.shape(), (4, 3));
    }

    #[tokio::test]
    async fn test_csr_nnz() {
        let f = write_mtx(TINY_MTX);
        let reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        assert_eq!(reader.x.indptr[4] as usize, 4, "total nnz");
    }

    #[tokio::test]
    async fn test_csr_indptr_values() {
        let f = write_mtx(TINY_MTX);
        let reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        // cell 0 has 1 entry (gene 0 = 5)
        // cell 1 has 1 entry (gene 1 = 3)
        // cell 2 has 0 entries
        // cell 3 has 2 entries (gene 1 = 7, gene 2 = 2)
        assert_eq!(reader.x.indptr, vec![0, 1, 2, 2, 4]);
    }

    #[tokio::test]
    async fn test_csr_indices_values() {
        let f = write_mtx(TINY_MTX);
        let reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        assert_eq!(reader.x.indices, vec![0, 1, 1, 2]); // gene indices (0-based)
    }

    #[tokio::test]
    async fn test_csr_data_values() {
        let f = write_mtx(TINY_MTX);
        let reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        match &reader.x.data {
            crate::dtype::TypedVec::I32(v) => assert_eq!(v, &[5, 3, 7, 2]),
            _ => panic!("expected I32"),
        }
    }

    // ── streaming ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_stream_total_nnz() {
        let f = write_mtx(TINY_MTX);
        let mut reader = TenxMtxReader::open(f.path(), 2).unwrap();
        let mut total_nnz = 0usize;
        let mut total_cells = 0usize;
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let c = chunk.unwrap();
            total_cells += c.nrows;
            total_nnz += c.data.indices.len();
        }
        assert_eq!(total_cells, 4);
        assert_eq!(total_nnz, 4);
    }

    #[tokio::test]
    async fn test_stream_chunk_size() {
        let f = write_mtx(TINY_MTX);
        let chunk_size = 2usize;
        let mut reader = TenxMtxReader::open(f.path(), chunk_size).unwrap();
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            assert!(chunk.unwrap().nrows <= chunk_size);
        }
    }

    // ── gzip path ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_gz_reader_shape_and_nnz() {
        let f = write_mtx_gz(TINY_MTX);
        let mut reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        assert_eq!(reader.shape(), (4, 3));
        let mut total_nnz = 0usize;
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            total_nnz += chunk.unwrap().data.indices.len();
        }
        assert_eq!(total_nnz, 4);
    }

    // ── real-type matrix ──────────────────────────────────────────────────────

    #[test]
    fn test_real_dtype_detected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 2
1 1 1.5
2 2 3.14
";
        let f = write_mtx(mtx);
        let info = read_mtx_info(f.path()).unwrap();
        assert_eq!(info.value_type, "real");

        let reader = TenxMtxReader::open(f.path(), 100).unwrap();
        assert_eq!(reader.x_dtype, DataType::F32);
    }

    // ── metadata / barcodes / features ───────────────────────────────────────

    #[tokio::test]
    async fn test_obs_synthesized_for_standalone() {
        let f = write_mtx(TINY_MTX);
        let mut reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        let obs = reader.obs().await.unwrap();
        assert_eq!(obs.index.len(), 4); // n_cells
        assert_eq!(obs.index[0], "1");
        assert_eq!(obs.index[3], "4");
    }

    #[tokio::test]
    async fn test_var_synthesized_for_standalone() {
        let f = write_mtx(TINY_MTX);
        let mut reader = TenxMtxReader::open(f.path(), 1000).unwrap();
        let var = reader.var().await.unwrap();
        assert_eq!(var.index.len(), 3); // n_genes
    }

    // ── 10x directory layout ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_tenx_dir_with_metadata() {
        let dir = tempfile::tempdir().unwrap();

        std::fs::write(dir.path().join("matrix.mtx"), TINY_MTX).unwrap();
        std::fs::write(
            dir.path().join("barcodes.tsv"),
            "AAACCTGAGAAACCAT-1\nAAAGGATCACGACTGC-1\nGTATCCTGTACAGGAT-1\nTTGCCGTCAGGCCTCG-1\n",
        )
        .unwrap();
        // features.tsv: id \t name \t feature_type
        std::fs::write(
            dir.path().join("features.tsv"),
            "ENSG001\tGeneA\tGene Expression\nENSG002\tGeneB\tGene Expression\nENSG003\tGeneC\tAntibody Capture\n",
        )
        .unwrap();

        let mut reader = TenxMtxReader::open(dir.path(), 1000).unwrap();
        assert_eq!(reader.shape(), (4, 3));

        let obs = reader.obs().await.unwrap();
        assert_eq!(obs.index[0], "AAACCTGAGAAACCAT-1");
        assert_eq!(obs.index.len(), 4);

        let var = reader.var().await.unwrap();
        assert_eq!(var.index, vec!["ENSG001", "ENSG002", "ENSG003"]);
        assert!(var.columns.iter().any(|c| c.name == "gene_name"));
        assert!(var.columns.iter().any(|c| c.name == "feature_type"));

        let gene_names: Vec<&str> = var
            .columns
            .iter()
            .find(|c| c.name == "gene_name")
            .map(|c| match &c.data {
                crate::ir::ColumnData::String(v) => v.iter().map(|s| s.as_str()).collect(),
                _ => vec![],
            })
            .unwrap_or_default();
        assert_eq!(gene_names, vec!["GeneA", "GeneB", "GeneC"]);
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_missing_banner_errors() {
        let f = write_mtx("3 4 4\n1 1 5\n");
        assert!(read_mtx_info(f.path()).is_err());
    }

    #[test]
    fn test_unsorted_entries_errors() {
        // cell 2 appears before cell 1 — violates sorted order
        let mtx = "\
%%MatrixMarket matrix coordinate integer general
2 3 2
1 2 5
1 1 3
";
        let f = write_mtx(mtx);
        assert!(TenxMtxReader::open(f.path(), 100).is_err());
    }

    // ── integration (skipped if /tmp/be1.mtx absent) ─────────────────────────

    #[tokio::test]
    async fn test_be1_mtx_if_present() {
        let p = std::path::Path::new("/tmp/be1.mtx");
        if !p.exists() {
            return;
        }
        let info = read_mtx_info(p).unwrap();
        assert_eq!(info.n_genes, 36753);
        assert_eq!(info.n_cells, 6898);
        assert_eq!(info.nnz, 16524865);
        assert_eq!(info.value_type, "integer");

        let mut reader = TenxMtxReader::open(p, 5000).unwrap();
        assert_eq!(reader.shape(), (6898, 36753));
        let mut total_nnz = 0usize;
        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            total_nnz += chunk.unwrap().data.indices.len();
        }
        assert_eq!(total_nnz, 16524865);
    }
}
