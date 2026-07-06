//! MatrixMarket (`.mtx`) reader — streaming, cell-major.
//!
//! Supports the 10x "Market Exchange" (MEX) triplet directory
//! (`matrix.mtx[.gz]` + `barcodes.tsv[.gz]` + `features.tsv[.gz]`/`genes.tsv[.gz]`),
//! a GEO-style prefixed triplet (`PREFIX_matrix.mtx.gz`, …), and a bare
//! `.mtx[.gz]` file (synthetic integer obs/var indices).
//!
//! The matrix is streamed one cell-chunk at a time, so a 350M-nonzero file
//! never materializes in RAM. This requires the entries to be sorted by the
//! cell axis (true for all 10x output); an out-of-order transition is a hard
//! error rather than a silent re-sort — re-sorting would defeat the whole
//! point of streaming.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_trait::async_trait;
use flate2::read::MultiGzDecoder;
use futures::stream::{self, Stream};

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    Column, ColumnData, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR, SparseMatrixMeta,
    UnsTable, VarTable, Varm,
};
use crate::stream::DatasetReader;

/// Open a path (dir or file) as a `.mtx` reader.
pub struct MtxReader {
    mtx_path: PathBuf,
    /// Cells are the matrix columns (10x convention). When false, cells are rows.
    cells_are_columns: bool,
    /// `pattern` matrices carry no value field; every stored entry is `1`.
    is_pattern: bool,
    dtype: DataType,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    /// Loaded from `barcodes.tsv` when present; else synthesized on demand.
    obs_index: Option<Vec<String>>,
    /// Feature ids (column 1 of `features.tsv`/`genes.tsv`).
    var_index: Option<Vec<String>>,
    /// Feature symbols (column 2), exposed as a `gene_symbols` var column.
    var_names: Option<Vec<String>>,
}

/// Parsed MatrixMarket banner + dimensions.
struct Header {
    n_rows: usize,
    n_cols: usize,
    is_pattern: bool,
    is_integer: bool,
}

fn open_bufread(path: &Path) -> Result<Box<dyn BufRead + Send>> {
    let file = File::open(path)?;
    let is_gz = path.extension().and_then(|e| e.to_str()) == Some("gz");
    if is_gz {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Read the banner + dimensions, leaving the reader positioned at the first
/// data entry. Comment (`%`) and blank lines are skipped.
fn read_header(reader: &mut dyn BufRead) -> Result<Header> {
    let mut line = String::new();

    // Banner: %%MatrixMarket matrix coordinate <field> <symmetry>
    reader.read_line(&mut line)?;
    let banner = line.to_ascii_lowercase();
    let mut tok = banner.split_whitespace();
    let bad = |m: &str| ScxError::InvalidFormat(format!("{m}: {}", line.trim()));
    if tok.next() != Some("%%matrixmarket") {
        return Err(bad("not a MatrixMarket file"));
    }
    let _object = tok.next(); // "matrix"
    if tok.next() != Some("coordinate") {
        return Err(bad("only coordinate (sparse) .mtx is supported, not"));
    }
    let field = tok.next().unwrap_or("real");
    let is_pattern = field == "pattern";
    let is_integer = field == "integer";
    if field == "complex" {
        return Err(bad("complex .mtx is unsupported"));
    }
    let symmetry = tok.next().unwrap_or("general");
    if symmetry != "general" {
        return Err(bad(&format!(
            "only 'general' symmetry is supported, not '{symmetry}' in"
        )));
    }

    // First non-comment, non-blank line is the dimensions line.
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(ScxError::InvalidFormat(
                ".mtx ended before dimensions line".into(),
            ));
        }
        let s = line.trim();
        if s.is_empty() || s.starts_with('%') {
            continue;
        }
        let mut it = s.split_whitespace();
        let n_rows = it.next().and_then(|v| v.parse().ok());
        let n_cols = it.next().and_then(|v| v.parse().ok());
        match (n_rows, n_cols) {
            (Some(n_rows), Some(n_cols)) => {
                return Ok(Header {
                    n_rows,
                    n_cols,
                    is_pattern,
                    is_integer,
                })
            }
            _ => {
                return Err(ScxError::InvalidFormat(format!(
                    "bad .mtx dimensions line: {s}"
                )))
            }
        }
    }
}

fn first_existing(dir: &Path, names: &[String]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

/// Resolve the `.mtx` path plus optional sibling barcodes/features files from a
/// directory or a file path.
fn resolve_paths(input: &Path) -> Result<(PathBuf, Option<PathBuf>, Option<PathBuf>)> {
    let gz = |stem: &str| vec![format!("{stem}.gz"), stem.to_string()];

    if input.is_dir() {
        let mtx = first_existing(input, &gz("matrix.mtx")).ok_or_else(|| {
            ScxError::InvalidFormat(format!(
                "no matrix.mtx[.gz] in directory '{}'",
                input.display()
            ))
        })?;
        let barcodes = first_existing(input, &gz("barcodes.tsv"));
        let mut feat_names = gz("features.tsv");
        feat_names.extend(gz("genes.tsv"));
        let features = first_existing(input, &feat_names);
        return Ok((mtx, barcodes, features));
    }

    // File path: derive a GEO-style prefix by stripping a trailing
    // `matrix.mtx[.gz]` from the filename, then look for prefixed siblings.
    let dir = input.parent().unwrap_or_else(|| Path::new("."));
    let fname = input
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    let prefix = fname
        .strip_suffix("matrix.mtx.gz")
        .or_else(|| fname.strip_suffix("matrix.mtx"))
        .unwrap_or("");
    let prefixed = |stem: &str| vec![format!("{prefix}{stem}.gz"), format!("{prefix}{stem}")];
    let barcodes = first_existing(dir, &prefixed("barcodes.tsv"));
    let mut feat_names = prefixed("features.tsv");
    feat_names.extend(prefixed("genes.tsv"));
    let features = first_existing(dir, &feat_names);
    Ok((input.to_path_buf(), barcodes, features))
}

/// Read a tsv, returning column 0 and (if present) column 1.
fn read_tsv_cols(path: &Path) -> Result<(Vec<String>, Vec<String>)> {
    let mut reader = open_bufread(path)?;
    let (mut col0, mut col1) = (Vec::new(), Vec::new());
    let mut line = String::new();
    while reader.read_line({
        line.clear();
        &mut line
    })? != 0
    {
        let s = line.trim_end_matches(['\n', '\r']);
        if s.is_empty() {
            continue;
        }
        let mut it = s.split('\t');
        col0.push(it.next().unwrap_or("").to_string());
        if let Some(c1) = it.next() {
            col1.push(c1.to_string());
        }
    }
    Ok((col0, col1))
}

impl MtxReader {
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let input = path.as_ref();
        let (mtx_path, barcodes_path, features_path) = resolve_paths(input)?;

        let header = {
            let mut reader = open_bufread(&mtx_path)?;
            read_header(&mut *reader)?
        };

        // Load metadata (small: one entry per barcode / feature).
        let obs_index = match &barcodes_path {
            Some(p) => Some(read_tsv_cols(p)?.0),
            None => None,
        };
        let (var_index, var_names) = match &features_path {
            Some(p) => {
                let (ids, names) = read_tsv_cols(p)?;
                (Some(ids), (!names.is_empty()).then_some(names))
            }
            None => (None, None),
        };

        // Orient: match the metadata lengths to the matrix axes. 10x lays out
        // genes(rows) × cells(cols); we honor that but flip if the barcode count
        // only matches the row axis.
        let nb = obs_index.as_ref().map(|v| v.len());
        let cells_are_columns = match nb {
            Some(nb) if nb == header.n_cols => true,
            Some(nb) if nb == header.n_rows => false,
            Some(nb) => {
                tracing::warn!(
                    "barcodes count {nb} matches neither .mtx axis ({}×{}); assuming cells are columns",
                    header.n_rows,
                    header.n_cols
                );
                true
            }
            None => true,
        };

        let (n_obs, n_vars) = if cells_are_columns {
            (header.n_cols, header.n_rows)
        } else {
            (header.n_rows, header.n_cols)
        };

        let dtype = if header.is_integer || header.is_pattern {
            DataType::I32
        } else {
            DataType::F32
        };

        Ok(Self {
            mtx_path,
            cells_are_columns,
            is_pattern: header.is_pattern,
            dtype,
            n_obs,
            n_vars,
            chunk_size,
            obs_index,
            var_index,
            var_names,
        })
    }
}

/// One parsed coordinate entry, oriented to (cell, gene).
struct Entry {
    cell: usize,
    gene: u32,
    val: f64,
}

/// Streaming scan over the entries of a `.mtx`, grouping consecutive entries
/// into per-cell CSR rows.
struct MtxScan {
    reader: Box<dyn BufRead + Send>,
    cells_are_columns: bool,
    is_pattern: bool,
    n_obs: usize,
    n_vars: usize,
    dtype: DataType,
    next_obs: usize,
    /// Lookahead entry read but not yet assigned to a row.
    pending: Option<Entry>,
    eof: bool,
    line: String,
}

impl MtxScan {
    fn start(path: &Path, cfg: &ScanCfg) -> Result<Self> {
        let mut reader = open_bufread(path)?;
        read_header(&mut *reader)?; // reposition past the header
        Ok(Self {
            reader,
            cells_are_columns: cfg.cells_are_columns,
            is_pattern: cfg.is_pattern,
            n_obs: cfg.n_obs,
            n_vars: cfg.n_vars,
            dtype: cfg.dtype,
            next_obs: 0,
            pending: None,
            eof: false,
            line: String::new(),
        })
    }

    /// Read and orient the next data entry, or `None` at EOF.
    fn read_entry(&mut self) -> Result<Option<Entry>> {
        loop {
            self.line.clear();
            if self.reader.read_line(&mut self.line)? == 0 {
                return Ok(None);
            }
            let s = self.line.trim();
            if s.is_empty() || s.starts_with('%') {
                continue;
            }
            let mut it = s.split_whitespace();
            let a: usize = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| ScxError::InvalidFormat(format!("bad .mtx entry: {s}")))?;
            let b: usize = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| ScxError::InvalidFormat(format!("bad .mtx entry: {s}")))?;
            let val = if self.is_pattern {
                1.0
            } else {
                it.next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| ScxError::InvalidFormat(format!("bad .mtx value: {s}")))?
            };
            // MatrixMarket is 1-indexed.
            let (cell1, gene1) = if self.cells_are_columns {
                (b, a)
            } else {
                (a, b)
            };
            if cell1 == 0 || cell1 > self.n_obs || gene1 == 0 || gene1 > self.n_vars {
                return Err(ScxError::InvalidFormat(format!(
                    ".mtx coordinate out of range: {s}"
                )));
            }
            return Ok(Some(Entry {
                cell: cell1 - 1,
                gene: (gene1 - 1) as u32,
                val,
            }));
        }
    }

    /// Produce the next chunk of up to `chunk_size` cells, or `None` when done.
    fn next_chunk(&mut self, chunk_size: usize) -> Option<Result<MatrixChunk>> {
        if self.next_obs >= self.n_obs {
            return None;
        }
        let row_start = self.next_obs;
        let row_end = (row_start + chunk_size).min(self.n_obs);

        let mut indptr: Vec<u64> = Vec::with_capacity(row_end - row_start + 1);
        indptr.push(0);
        let mut indices: Vec<u32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();

        for cell in row_start..row_end {
            let mut row: Vec<(u32, f64)> = Vec::new();
            loop {
                if self.pending.is_none() && !self.eof {
                    match self.read_entry() {
                        Ok(Some(e)) => self.pending = Some(e),
                        Ok(None) => self.eof = true,
                        Err(e) => return Some(Err(e)),
                    }
                }
                match &self.pending {
                    Some(e) if e.cell == cell => {
                        let e = self.pending.take().unwrap();
                        row.push((e.gene, e.val));
                    }
                    // ponytail: hard-error on unsorted input rather than silently
                    // mis-reading. Upgrade path if a real unsorted .mtx shows up:
                    // two-pass build (count per-cell nnz → indptr, then scatter),
                    // ~nnz*8 bytes resident vs. the full-COO blowup we're avoiding.
                    Some(e) if e.cell < cell => {
                        return Some(Err(ScxError::InvalidFormat(format!(
                            ".mtx not sorted by cell axis (saw cell {} while filling cell {}); \
                             re-sort the file or ingest it with a buffering tool",
                            e.cell, cell
                        ))));
                    }
                    // pending belongs to a later cell, or EOF: this row is done.
                    _ => break,
                }
            }
            // 10x writes gene-sorted rows already; sort defensively so output CSR
            // has ascending indices regardless of source ordering.
            row.sort_unstable_by_key(|&(g, _)| g);
            for (g, v) in row {
                indices.push(g);
                vals.push(v);
            }
            indptr.push(indices.len() as u64);
        }

        self.next_obs = row_end;
        let data = typed_from_f64(self.dtype, vals);
        Some(Ok(MatrixChunk {
            row_offset: row_start,
            nrows: row_end - row_start,
            data: SparseMatrixCSR {
                shape: (row_end - row_start, self.n_vars),
                indptr,
                indices,
                data,
            },
        }))
    }
}

// ponytail: values buffered as f64 then cast — exact for integer counts up to
// 2^53, which covers any real single-cell matrix.
fn typed_from_f64(dtype: DataType, v: Vec<f64>) -> TypedVec {
    match dtype {
        DataType::F32 => TypedVec::F32(v.into_iter().map(|x| x as f32).collect()),
        DataType::F64 => TypedVec::F64(v),
        DataType::I32 => TypedVec::I32(v.into_iter().map(|x| x as i32).collect()),
        DataType::U32 => TypedVec::U32(v.into_iter().map(|x| x as u32).collect()),
    }
}

/// The scalar parse config a streaming scan needs (no metadata, no paths).
#[derive(Clone, Copy)]
struct ScanCfg {
    cells_are_columns: bool,
    is_pattern: bool,
    dtype: DataType,
    n_obs: usize,
    n_vars: usize,
}

enum ScanState {
    Init,
    Run(MtxScan),
    Done,
}

#[async_trait]
impl DatasetReader for MtxReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        let index = self
            .obs_index
            .clone()
            .unwrap_or_else(|| (0..self.n_obs).map(|i| i.to_string()).collect());
        Ok(ObsTable {
            index,
            columns: Vec::new(),
        })
    }

    async fn var(&mut self) -> Result<VarTable> {
        let index = self
            .var_index
            .clone()
            .unwrap_or_else(|| (0..self.n_vars).map(|i| i.to_string()).collect());
        let columns = match &self.var_names {
            Some(names) if names.len() == index.len() => vec![Column {
                name: "gene_symbols".to_string(),
                data: ColumnData::String(names.clone()),
            }],
            _ => Vec::new(),
        };
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
        let mtx_path = self.mtx_path.clone();
        let chunk_size = self.chunk_size;
        let cfg = ScanCfg {
            cells_are_columns: self.cells_are_columns,
            is_pattern: self.is_pattern,
            dtype: self.dtype,
            n_obs: self.n_obs,
            n_vars: self.n_vars,
        };

        Box::pin(stream::unfold(ScanState::Init, move |state| {
            let mtx_path = mtx_path.clone();
            async move {
                let mut scan = match state {
                    ScanState::Done => return None,
                    ScanState::Run(scan) => scan,
                    ScanState::Init => match MtxScan::start(&mtx_path, &cfg) {
                        Ok(s) => s,
                        Err(e) => return Some((Err(e), ScanState::Done)),
                    },
                };
                match scan.next_chunk(chunk_size) {
                    None => None,
                    Some(Ok(c)) => Some((Ok(c), ScanState::Run(scan))),
                    Some(Err(e)) => Some((Err(e), ScanState::Done)),
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Format;
    use futures::executor::block_on;
    use futures::StreamExt;
    use std::io::Write;

    fn write_file(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    // genes(rows)=3 × cells(cols)=4, cell-major (column-sorted).
    const MTX: &str = "%%MatrixMarket matrix coordinate integer general\n\
                       %\n\
                       3 4 5\n\
                       1 1 5\n\
                       3 1 7\n\
                       2 2 1\n\
                       1 4 9\n\
                       2 4 2\n";

    fn collect_x(reader: &mut MtxReader) -> Vec<MatrixChunk> {
        block_on(async {
            reader
                .x_stream()
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .map(|c| c.unwrap())
                .collect()
        })
    }

    #[test]
    fn bare_mtx_shape_and_stream() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_file(dir.path(), "matrix.mtx", MTX);

        let mut r = MtxReader::open(&p, 2).unwrap();
        assert_eq!(r.shape(), (4, 3)); // 4 cells × 3 genes
        assert_eq!(r.dtype(), DataType::I32);

        let chunks = collect_x(&mut r);
        // chunk_size 2 → cells [0,1] then [2,3].
        assert_eq!(chunks.len(), 2);

        // Reassemble a dense view to verify orientation + values.
        let mut dense = vec![vec![0i32; 3]; 4];
        for ch in &chunks {
            let TypedVec::I32(vals) = &ch.data.data else {
                panic!("expected i32")
            };
            for row in 0..ch.nrows {
                let lo = ch.data.indptr[row] as usize;
                let hi = ch.data.indptr[row + 1] as usize;
                for k in lo..hi {
                    dense[ch.row_offset + row][ch.data.indices[k] as usize] = vals[k];
                }
            }
        }
        assert_eq!(dense[0], vec![5, 0, 7]); // cell 1: gene1=5, gene3=7
        assert_eq!(dense[1], vec![0, 1, 0]); // cell 2: gene2=1
        assert_eq!(dense[2], vec![0, 0, 0]); // cell 3: empty
        assert_eq!(dense[3], vec![9, 2, 0]); // cell 4: gene1=9, gene2=2
    }

    #[test]
    fn mex_dir_with_metadata_orients_and_labels() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "matrix.mtx", MTX);
        write_file(dir.path(), "barcodes.tsv", "AAA\nBBB\nCCC\nDDD\n");
        write_file(
            dir.path(),
            "features.tsv",
            "ENSG1\tGeneA\tGene Expression\n\
             ENSG2\tGeneB\tGene Expression\n\
             ENSG3\tGeneC\tGene Expression\n",
        );

        let mut r = MtxReader::open(dir.path(), 1000).unwrap();
        assert_eq!(r.shape(), (4, 3));
        assert!(r.cells_are_columns);

        let obs = block_on(r.obs()).unwrap();
        assert_eq!(obs.index, vec!["AAA", "BBB", "CCC", "DDD"]);
        let var = block_on(r.var()).unwrap();
        assert_eq!(var.index, vec!["ENSG1", "ENSG2", "ENSG3"]);
        assert_eq!(var.columns[0].name, "gene_symbols");
    }

    #[test]
    fn unsorted_mtx_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        // cell 3 appears before cell 1 → out of order on the cell axis.
        let bad = "%%MatrixMarket matrix coordinate integer general\n\
                   2 3 2\n\
                   1 3 5\n\
                   1 1 4\n";
        let p = write_file(dir.path(), "matrix.mtx", bad);
        let mut r = MtxReader::open(&p, 1000).unwrap();
        let err = block_on(async { r.x_stream().collect::<Vec<_>>().await });
        assert!(
            err.iter().any(|c| c.is_err()),
            "expected an out-of-order error"
        );
    }

    fn write_gz(dir: &Path, name: &str, body: &str) -> PathBuf {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let p = dir.join(name);
        let mut enc = GzEncoder::new(File::create(&p).unwrap(), Compression::fast());
        enc.write_all(body.as_bytes()).unwrap();
        enc.finish().unwrap();
        p
    }

    /// Dense reconstruction from the streamed chunks, for value assertions.
    fn to_dense(chunks: &[MatrixChunk], n_obs: usize, n_vars: usize) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0f64; n_vars]; n_obs];
        for ch in chunks {
            let vals = ch.data.data.to_f64();
            for row in 0..ch.nrows {
                let lo = ch.data.indptr[row] as usize;
                let hi = ch.data.indptr[row + 1] as usize;
                for k in lo..hi {
                    dense[ch.row_offset + row][ch.data.indices[k] as usize] = vals[k];
                }
            }
        }
        dense
    }

    #[test]
    fn gzipped_mex_directory_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        write_gz(dir.path(), "matrix.mtx.gz", MTX);
        write_gz(dir.path(), "barcodes.tsv.gz", "AAA\nBBB\nCCC\nDDD\n");
        write_gz(
            dir.path(),
            "features.tsv.gz",
            "ENSG1\tGeneA\nENSG2\tGeneB\nENSG3\tGeneC\n",
        );

        // Directory detection routes here, and every file is gz.
        assert_eq!(crate::detect::detect(dir.path()), Some(Format::Mtx));

        let mut r = MtxReader::open(dir.path(), 1000).unwrap();
        assert_eq!(r.shape(), (4, 3));
        assert_eq!(block_on(r.obs()).unwrap().index[3], "DDD");
        let dense = to_dense(&collect_x(&mut r), 4, 3);
        assert_eq!(dense[0], vec![5.0, 0.0, 7.0]);
        assert_eq!(dense[3], vec![9.0, 2.0, 0.0]);
    }

    #[test]
    fn geo_prefixed_file_resolves_siblings() {
        // GEO layout: PREFIX_matrix.mtx.gz + PREFIX_barcodes/features.
        let dir = tempfile::tempdir().unwrap();
        let mtx = write_gz(dir.path(), "GSM1_matrix.mtx.gz", MTX);
        write_gz(dir.path(), "GSM1_barcodes.tsv.gz", "AAA\nBBB\nCCC\nDDD\n");
        write_gz(dir.path(), "GSM1_genes.tsv.gz", "G1\tA\nG2\tB\nG3\tC\n");

        assert_eq!(crate::detect::detect(&mtx), Some(Format::Mtx));
        let mut r = MtxReader::open(&mtx, 1000).unwrap();
        assert_eq!(r.shape(), (4, 3));
        // barcodes + genes.tsv (v2 name) both picked up via the prefix.
        assert_eq!(
            block_on(r.obs()).unwrap().index,
            vec!["AAA", "BBB", "CCC", "DDD"]
        );
        assert_eq!(block_on(r.var()).unwrap().index, vec!["G1", "G2", "G3"]);
    }

    #[test]
    fn orientation_flips_when_barcodes_match_rows() {
        // Cells-as-rows layout: 4 cells (rows) × 3 genes (cols), row-major.
        let dir = tempfile::tempdir().unwrap();
        let body = "%%MatrixMarket matrix coordinate integer general\n\
                    4 3 3\n\
                    1 1 5\n\
                    2 2 8\n\
                    4 3 9\n";
        write_file(dir.path(), "matrix.mtx", body);
        write_file(dir.path(), "barcodes.tsv", "AAA\nBBB\nCCC\nDDD\n"); // len 4 == rows
        let mut r = MtxReader::open(dir.path(), 1000).unwrap();
        assert!(!r.cells_are_columns);
        assert_eq!(r.shape(), (4, 3));
        let dense = to_dense(&collect_x(&mut r), 4, 3);
        assert_eq!(dense[0], vec![5.0, 0.0, 0.0]);
        assert_eq!(dense[1], vec![0.0, 8.0, 0.0]);
        assert_eq!(dense[3], vec![0.0, 0.0, 9.0]);
    }

    #[test]
    fn pattern_and_real_fields() {
        let dir = tempfile::tempdir().unwrap();
        // pattern: no value column, every stored entry is 1, dtype i32.
        let pat = "%%MatrixMarket matrix coordinate pattern general\n\
                   2 2 2\n\
                   1 1\n\
                   2 2\n";
        let pp = write_file(dir.path(), "pattern.mtx", pat);
        let mut rp = MtxReader::open(&pp, 1000).unwrap();
        assert_eq!(rp.dtype(), DataType::I32);
        assert_eq!(to_dense(&collect_x(&mut rp), 2, 2)[0], vec![1.0, 0.0]);

        // real: fractional values, dtype f32.
        let real = "%%MatrixMarket matrix coordinate real general\n\
                    1 2 2\n\
                    1 1 1.5\n\
                    1 2 2.5\n";
        let rrp = write_file(dir.path(), "real.mtx", real);
        let mut rr = MtxReader::open(&rrp, 1000).unwrap();
        assert_eq!(rr.dtype(), DataType::F32);
        // 1 gene (row) × 2 cells (cols) → 2 obs × 1 var.
        assert_eq!(
            to_dense(&collect_x(&mut rr), 2, 1),
            vec![vec![1.5], vec![2.5]]
        );
    }

    #[test]
    fn header_and_coordinate_errors() {
        let dir = tempfile::tempdir().unwrap();
        let cases = [
            ("not MatrixMarket", "# nope\n1 1 0\n"),
            (
                "dense array",
                "%%MatrixMarket matrix array real general\n1 1\n",
            ),
            (
                "symmetric",
                "%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 1 1\n",
            ),
            (
                "complex",
                "%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1 0\n",
            ),
        ];
        for (name, body) in cases {
            let p = write_file(dir.path(), &format!("{name}.mtx").replace(' ', "_"), body);
            assert!(
                MtxReader::open(&p, 1000).is_err(),
                "expected open error for {name}"
            );
        }

        // Out-of-range coordinate is caught while streaming.
        let oob = "%%MatrixMarket matrix coordinate integer general\n\
                   2 2 1\n\
                   9 1 3\n";
        let p = write_file(dir.path(), "oob.mtx", oob);
        let mut r = MtxReader::open(&p, 1000).unwrap();
        let out = block_on(async { r.x_stream().collect::<Vec<_>>().await });
        assert!(
            out.iter().any(|c| c.is_err()),
            "expected out-of-range error"
        );
    }

    #[test]
    fn empty_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(MtxReader::open(dir.path(), 1000).is_err());
    }
}
