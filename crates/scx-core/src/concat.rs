//! Streaming concatenation along the obs (cell) axis: many datasets → one h5ad.
//!
//! Mirrors `anndata.concat()` vocabulary and defaults (`join`, `label`, `keys`,
//! `index_unique`, `merge`) so users familiar with AnnData get no surprises.
//!
//! Bounded memory: the count matrix and every layer are streamed chunk-by-chunk
//! straight into the writer.  obs/var/obsm are materialised, as everywhere else
//! in scx (the `DatasetWriter` API takes whole frames).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;

use futures::StreamExt;

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::h5ad::H5AdWriter;
use crate::ir::{
    Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
    SparseMatrixMeta, UnsTable, VarTable,
};
use crate::merge::align::reindex_column;
use crate::stream::DatasetWriter;
use crate::OpenOptions;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Gene-axis join mode — `anndata.concat(join=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Join {
    /// Genes present in every input (anndata's default).
    #[default]
    Inner,
    /// Union of genes; genes absent from an input are implicit zeros there.
    Outer,
}

impl FromStr for Join {
    type Err = ScxError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "inner" => Ok(Join::Inner),
            "outer" => Ok(Join::Outer),
            other => Err(ScxError::InvalidFormat(format!(
                "unknown join '{other}'; use inner or outer"
            ))),
        }
    }
}

/// How to combine var (gene) columns — `anndata.concat(merge=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Drop all var columns, keep only the gene index (anndata's default).
    #[default]
    None,
    /// Keep columns present in every input and agreeing on shared genes.
    Same,
    /// Keep columns agreeing on shared genes across the inputs that have them.
    Unique,
    /// Keep every column, taken from the first input that has it.
    First,
    /// Keep columns present in exactly one input.
    Only,
}

impl FromStr for MergeStrategy {
    type Err = ScxError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(MergeStrategy::None),
            "same" => Ok(MergeStrategy::Same),
            "unique" => Ok(MergeStrategy::Unique),
            "first" => Ok(MergeStrategy::First),
            "only" => Ok(MergeStrategy::Only),
            other => Err(ScxError::InvalidFormat(format!(
                "unknown merge strategy '{other}'; use none, same, unique, first, or only"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConcatOptions {
    pub join: Join,
    /// Name of the obs column recording each cell's source dataset.
    pub label: Option<String>,
    /// One level value per input; empty means "derive from file stems".
    pub keys: Vec<String>,
    /// Separator appended to obs_names (`"cell1-sample_a"` for `Some("-")`).
    pub index_unique: Option<String>,
    pub merge: MergeStrategy,
    pub chunk_size: usize,
    pub dtype: DataType,
    pub compress: Option<u8>,
}

impl Default for ConcatOptions {
    fn default() -> Self {
        Self {
            join: Join::default(),
            label: None,
            keys: Vec::new(),
            index_unique: None,
            merge: MergeStrategy::default(),
            chunk_size: 5000,
            dtype: DataType::F32,
            compress: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Concatenate `inputs` along the obs axis into one h5ad at `output`.
///
/// Returns the output shape `(n_obs, n_vars)`.
pub async fn concat(
    inputs: &[String],
    output: &Path,
    opts: &ConcatOptions,
) -> Result<(usize, usize)> {
    if inputs.is_empty() {
        return Err(ScxError::InvalidFormat(
            "concat needs at least one input".into(),
        ));
    }
    let keys = resolve_keys(inputs, &opts.keys)?;

    let open_opts = OpenOptions::new(opts.chunk_size);
    let mut readers = Vec::with_capacity(inputs.len());
    for input in inputs {
        readers.push(
            crate::open(input, &open_opts)
                .await
                .map_err(|e| ScxError::InvalidFormat(format!("cannot open '{input}': {e}")))?,
        );
    }

    // --- axis metadata ---------------------------------------------------
    let n_obs_each: Vec<usize> = readers.iter().map(|r| r.shape().0).collect();
    let total_obs: usize = n_obs_each.iter().sum();

    let mut obs_tables = Vec::with_capacity(readers.len());
    let mut var_tables = Vec::with_capacity(readers.len());
    for (i, r) in readers.iter_mut().enumerate() {
        let (n_obs, n_vars) = r.shape();
        let mut obs = r.obs().await?;
        if obs.index.len() != n_obs {
            // Readers without a cell index (mtx, some 10x paths) get synthetic
            // names; without them the concatenated frame index is unusable.
            obs.index = (0..n_obs).map(|j| format!("{}_{j}", keys[i])).collect();
        }
        let var = r.var().await?;
        if var.index.len() != n_vars {
            return Err(ScxError::InvalidFormat(format!(
                "input '{}' has no var index — concat aligns genes by name",
                inputs[i]
            )));
        }
        obs_tables.push(obs);
        var_tables.push(var);
    }

    let out_var: Vec<String> = join_var_index(&var_tables, opts.join);
    if out_var.is_empty() {
        return Err(ScxError::InvalidFormat(
            "no genes left after the join — inputs share no gene names (try --join outer)".into(),
        ));
    }
    let n_out_vars = out_var.len();
    let maps: Vec<VarMap> = var_tables
        .iter()
        .map(|v| VarMap::build(&v.index, &out_var))
        .collect();

    let obs = build_obs(&obs_tables, &n_obs_each, &keys, opts)?;
    let var = VarTable {
        columns: merge_var_columns(&var_tables, &maps, n_out_vars, opts.merge),
        index: out_var,
    };

    // obsm and layer metadata need &mut, so collect before the streaming pass.
    let mut obsms = Vec::with_capacity(readers.len());
    let mut layer_metas = Vec::with_capacity(readers.len());
    for r in readers.iter_mut() {
        obsms.push(r.obsm().await?);
        layer_metas.push(r.layer_metas().await?);
    }
    let obsm = concat_obsm(&mut obsms, &n_obs_each, opts.join)?;
    let layer_names = join_names(
        &layer_metas
            .iter()
            .map(|ms| ms.iter().map(|m| m.name.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        opts.join,
    );

    tracing::info!(
        inputs = inputs.len(),
        n_obs = total_obs,
        n_vars = n_out_vars,
        join = ?opts.join,
        layers = layer_names.len(),
        "starting concat"
    );

    // --- write -----------------------------------------------------------
    let mut writer =
        H5AdWriter::create_compressed(output, total_obs, n_out_vars, opts.dtype, opts.compress)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    if !obsm.map.is_empty() {
        writer.write_obsm(&obsm).await?;
    }
    writer
        .write_uns(&UnsTable {
            raw: provenance_json(inputs, &keys, &n_obs_each, n_out_vars, opts),
        })
        .await?;

    for name in &layer_names {
        let meta_out = SparseMatrixMeta {
            name: name.clone(),
            shape: (total_obs, n_out_vars),
            indptr: Vec::new(),
        };
        writer.begin_sparse("layers", name, &meta_out).await?;
        for (i, r) in readers.iter().enumerate() {
            match layer_metas[i].iter().find(|m| &m.name == name) {
                Some(meta) => {
                    let mut stream = r.layer_stream(meta, opts.chunk_size);
                    while let Some(chunk) = stream.next().await {
                        let chunk = remap_chunk(chunk?, &maps[i], n_out_vars);
                        writer.write_sparse_chunk(&chunk).await?;
                    }
                }
                // Outer join: an input without this layer contributes zero rows.
                None => {
                    let mut left = n_obs_each[i];
                    while left > 0 {
                        let n = left.min(opts.chunk_size);
                        writer
                            .write_sparse_chunk(&empty_chunk(n, n_out_vars))
                            .await?;
                        left -= n;
                    }
                }
            }
        }
        writer.end_sparse().await?;
    }

    let mut total_nnz = 0usize;
    for (i, r) in readers.iter_mut().enumerate() {
        let mut stream = r.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = remap_chunk(chunk?, &maps[i], n_out_vars);
            total_nnz += chunk.data.indices.len();
            writer.write_x_chunk(&chunk).await?;
        }
    }
    writer.finalize().await?;

    tracing::info!(
        n_obs = total_obs,
        n_vars = n_out_vars,
        total_nnz,
        output = %output.display(),
        "concat complete"
    );
    Ok((total_obs, n_out_vars))
}

// ---------------------------------------------------------------------------
// Gene axis
// ---------------------------------------------------------------------------

/// Union (outer) or intersection (inner) of the per-input gene indices.
/// Order is first-appearance for outer, first-input order for inner.
fn join_var_index(vars: &[VarTable], join: Join) -> Vec<String> {
    join_names(
        &vars.iter().map(|v| v.index.clone()).collect::<Vec<_>>(),
        join,
    )
}

fn join_names(per_input: &[Vec<String>], join: Join) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    match join {
        Join::Outer => {
            for names in per_input {
                for n in names {
                    if seen.insert(n.as_str()) {
                        out.push(n.clone());
                    }
                }
            }
        }
        Join::Inner => {
            let others: Vec<HashSet<&str>> = per_input[1..]
                .iter()
                .map(|v| v.iter().map(|s| s.as_str()).collect())
                .collect();
            for n in per_input.first().into_iter().flatten() {
                if seen.insert(n.as_str()) && others.iter().all(|o| o.contains(n.as_str())) {
                    out.push(n.clone());
                }
            }
        }
    }
    out
}

/// Maps one input's gene positions onto the output gene axis.
struct VarMap {
    to_out: Vec<Option<u32>>,
    /// Source axis is already the output axis — chunks pass through untouched.
    identity: bool,
    /// Remapped column indices are not ascending — rows need re-sorting to stay
    /// canonical CSR.
    needs_sort: bool,
}

impl VarMap {
    fn build(src: &[String], out: &[String]) -> Self {
        let pos: HashMap<&str, u32> = out
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();
        let to_out: Vec<Option<u32>> = src.iter().map(|s| pos.get(s.as_str()).copied()).collect();
        let identity =
            src.len() == out.len() && to_out.iter().enumerate().all(|(i, m)| *m == Some(i as u32));
        let mut needs_sort = false;
        let mut last: Option<u32> = None;
        for m in to_out.iter().flatten() {
            if last.is_some_and(|l| *m < l) {
                needs_sort = true;
                break;
            }
            last = Some(*m);
        }
        Self {
            to_out,
            identity,
            needs_sort,
        }
    }
}

/// Rewrite a chunk's column indices onto the output gene axis, dropping entries
/// for genes the join excluded.
fn remap_chunk(mut chunk: MatrixChunk, map: &VarMap, n_out_vars: usize) -> MatrixChunk {
    if map.identity {
        chunk.data.shape = (chunk.nrows, n_out_vars);
        return chunk;
    }
    let csr = &chunk.data;
    let mut indptr = Vec::with_capacity(chunk.nrows + 1);
    indptr.push(0u64);
    let mut indices: Vec<u32> = Vec::with_capacity(csr.indices.len());
    let mut keep: Vec<usize> = Vec::with_capacity(csr.indices.len());
    let mut row: Vec<(u32, usize)> = Vec::new();

    for r in 0..chunk.nrows {
        let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
        row.clear();
        for p in s..e {
            if let Some(col) = map.to_out[csr.indices[p] as usize] {
                row.push((col, p));
            }
        }
        if map.needs_sort {
            row.sort_unstable_by_key(|&(c, _)| c);
        }
        for &(c, p) in &row {
            indices.push(c);
            keep.push(p);
        }
        indptr.push(indices.len() as u64);
    }

    let data = gather(&csr.data, &keep);
    MatrixChunk {
        row_offset: chunk.row_offset,
        nrows: chunk.nrows,
        data: SparseMatrixCSR {
            shape: (chunk.nrows, n_out_vars),
            indptr,
            indices,
            data,
        },
    }
}

fn gather(src: &TypedVec, idx: &[usize]) -> TypedVec {
    macro_rules! g {
        ($v:expr, $variant:path) => {
            $variant(idx.iter().map(|&i| $v[i]).collect())
        };
    }
    match src {
        TypedVec::F32(v) => g!(v, TypedVec::F32),
        TypedVec::F64(v) => g!(v, TypedVec::F64),
        TypedVec::I32(v) => g!(v, TypedVec::I32),
        TypedVec::U32(v) => g!(v, TypedVec::U32),
    }
}

fn empty_chunk(nrows: usize, n_vars: usize) -> MatrixChunk {
    MatrixChunk {
        row_offset: 0,
        nrows,
        data: SparseMatrixCSR {
            shape: (nrows, n_vars),
            indptr: vec![0u64; nrows + 1],
            indices: Vec::new(),
            data: TypedVec::F32(Vec::new()),
        },
    }
}

// ---------------------------------------------------------------------------
// Cell axis (obs)
// ---------------------------------------------------------------------------

fn resolve_keys(inputs: &[String], keys: &[String]) -> Result<Vec<String>> {
    if !keys.is_empty() {
        if keys.len() != inputs.len() {
            return Err(ScxError::InvalidFormat(format!(
                "got {} keys for {} inputs — pass one key per input",
                keys.len(),
                inputs.len()
            )));
        }
        return Ok(keys.to_vec());
    }
    let stems: Vec<String> = inputs
        .iter()
        .map(|p| {
            Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.clone())
        })
        .collect();
    let unique: HashSet<&str> = stems.iter().map(|s| s.as_str()).collect();
    if unique.len() != stems.len() {
        return Err(ScxError::InvalidFormat(
            "input file names are not unique — pass explicit keys".into(),
        ));
    }
    Ok(stems)
}

fn build_obs(
    tables: &[ObsTable],
    lens: &[usize],
    keys: &[String],
    opts: &ConcatOptions,
) -> Result<ObsTable> {
    let total: usize = lens.iter().sum();
    let mut index = Vec::with_capacity(total);
    for (t, key) in tables.iter().zip(keys) {
        match &opts.index_unique {
            Some(sep) => index.extend(t.index.iter().map(|n| format!("{n}{sep}{key}"))),
            None => index.extend(t.index.iter().cloned()),
        }
    }
    let mut seen: HashSet<&str> = HashSet::with_capacity(index.len());
    let dups = index.iter().filter(|n| !seen.insert(n.as_str())).count();
    if dups > 0 {
        tracing::warn!(
            duplicates = dups,
            "concatenated obs_names are not unique — pass index_unique to disambiguate"
        );
    }

    let mut columns = Vec::new();
    for name in join_names(
        &tables
            .iter()
            .map(|t| t.columns.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        opts.join,
    ) {
        let cols: Vec<Option<&ColumnData>> = tables
            .iter()
            .map(|t| t.columns.iter().find(|c| c.name == name).map(|c| &c.data))
            .collect();
        columns.push(Column {
            data: concat_column(&cols, lens),
            name,
        });
    }

    if let Some(label) = &opts.label {
        if columns.iter().any(|c| &c.name == label) {
            return Err(ScxError::InvalidFormat(format!(
                "label '{label}' collides with an existing obs column"
            )));
        }
        let codes = lens
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| std::iter::repeat_n(i as u32, n))
            .collect();
        columns.push(Column {
            name: label.clone(),
            data: ColumnData::Categorical {
                codes,
                levels: keys.to_vec(),
            },
        });
    }

    Ok(ObsTable { index, columns })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Float,
    Bool,
    Str,
    Cat,
}

fn kind_of(c: &ColumnData) -> Kind {
    match c {
        ColumnData::Int(_) => Kind::Int,
        ColumnData::Float(_) => Kind::Float,
        ColumnData::Bool(_) => Kind::Bool,
        ColumnData::String(_) => Kind::Str,
        ColumnData::Categorical { .. } => Kind::Cat,
    }
}

/// Stack one obs column across all inputs. Inputs missing the column are
/// NA-filled (Float→NaN, Int→0, Bool→false, String→"", Categorical→"NA" level),
/// matching the merge path's fill policy. Mixed dtypes promote:
/// int+float→float, string+categorical→categorical, anything else→string
/// (pandas' object fallback).
fn concat_column(cols: &[Option<&ColumnData>], lens: &[usize]) -> ColumnData {
    let kinds: Vec<Kind> = cols.iter().flatten().map(|c| kind_of(c)).collect();
    if kinds.is_empty() {
        return ColumnData::String(vec![String::new(); lens.iter().sum()]);
    }
    let target = if kinds.iter().all(|k| *k == kinds[0]) {
        kinds[0]
    } else if kinds.iter().all(|k| matches!(k, Kind::Int | Kind::Float)) {
        Kind::Float
    } else if kinds.iter().all(|k| matches!(k, Kind::Str | Kind::Cat)) {
        Kind::Cat
    } else {
        Kind::Str
    };

    match target {
        Kind::Int => {
            let mut out = Vec::new();
            for (c, &n) in cols.iter().zip(lens) {
                match c {
                    Some(ColumnData::Int(v)) => out.extend_from_slice(v),
                    _ => out.extend(std::iter::repeat_n(0, n)),
                }
            }
            ColumnData::Int(out)
        }
        Kind::Float => {
            let mut out = Vec::new();
            for (c, &n) in cols.iter().zip(lens) {
                match c {
                    Some(c) => out.extend(col_f64(c)),
                    None => out.extend(std::iter::repeat_n(f64::NAN, n)),
                }
            }
            ColumnData::Float(out)
        }
        Kind::Bool => {
            let mut out = Vec::new();
            for (c, &n) in cols.iter().zip(lens) {
                match c {
                    Some(ColumnData::Bool(v)) => out.extend_from_slice(v),
                    _ => out.extend(std::iter::repeat_n(false, n)),
                }
            }
            ColumnData::Bool(out)
        }
        Kind::Str => {
            let mut out = Vec::new();
            for (c, &n) in cols.iter().zip(lens) {
                match c {
                    Some(c) => out.extend(col_str(c)),
                    None => out.extend(std::iter::repeat_n(String::new(), n)),
                }
            }
            ColumnData::String(out)
        }
        Kind::Cat => {
            let mut b = CatBuilder::default();
            for (c, &n) in cols.iter().zip(lens) {
                match c {
                    Some(ColumnData::Categorical { codes, levels }) => {
                        let remap: Vec<u32> = levels.iter().map(|l| b.level(l)).collect();
                        for &code in codes {
                            // Out-of-range codes only appear in malformed input;
                            // "NA" is created lazily so a clean concat has no
                            // spurious level.
                            let mapped = match remap.get(code as usize) {
                                Some(&m) => m,
                                None => b.level("NA"),
                            };
                            b.codes.push(mapped);
                        }
                    }
                    Some(other) => {
                        for s in col_str(other) {
                            let code = b.level(&s);
                            b.codes.push(code);
                        }
                    }
                    None => {
                        let na = b.level("NA");
                        b.codes.extend(std::iter::repeat_n(na, n));
                    }
                }
            }
            ColumnData::Categorical {
                codes: b.codes,
                levels: b.levels,
            }
        }
    }
}

#[derive(Default)]
struct CatBuilder {
    levels: Vec<String>,
    pos: HashMap<String, u32>,
    codes: Vec<u32>,
}

impl CatBuilder {
    fn level(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.pos.get(s) {
            return i;
        }
        let i = self.levels.len() as u32;
        self.levels.push(s.to_string());
        self.pos.insert(s.to_string(), i);
        i
    }
}

fn col_f64(c: &ColumnData) -> Vec<f64> {
    match c {
        ColumnData::Float(v) => v.clone(),
        ColumnData::Int(v) => v.iter().map(|&x| x as f64).collect(),
        ColumnData::Bool(v) => v.iter().map(|&x| x as u8 as f64).collect(),
        other => vec![f64::NAN; other.len()],
    }
}

fn col_str(c: &ColumnData) -> Vec<String> {
    match c {
        ColumnData::String(v) => v.clone(),
        ColumnData::Int(v) => v.iter().map(|x| x.to_string()).collect(),
        ColumnData::Float(v) => v.iter().map(|x| x.to_string()).collect(),
        ColumnData::Bool(v) => v
            .iter()
            .map(|&x| if x { "True" } else { "False" }.to_string())
            .collect(),
        ColumnData::Categorical { codes, levels } => codes
            .iter()
            .map(|&c| levels.get(c as usize).cloned().unwrap_or_default())
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// var columns + obsm
// ---------------------------------------------------------------------------

/// ponytail: a merged var column is taken whole from the first input that has
/// it, so on an outer join the genes that input lacks are NA. Coalescing across
/// inputs is the upgrade if outer-join gene metadata starts mattering.
fn merge_var_columns(
    vars: &[VarTable],
    maps: &[VarMap],
    n_out_vars: usize,
    strategy: MergeStrategy,
) -> Vec<Column> {
    if strategy == MergeStrategy::None {
        return Vec::new();
    }
    // out position -> source position, per input.
    let inverses: Vec<Vec<Option<usize>>> = maps
        .iter()
        .map(|m| {
            let mut inv = vec![None; n_out_vars];
            for (src, out) in m.to_out.iter().enumerate() {
                if let Some(o) = out {
                    inv[*o as usize] = Some(src);
                }
            }
            inv
        })
        .collect();

    let names = join_names(
        &vars
            .iter()
            .map(|v| v.columns.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        Join::Outer,
    );

    let mut out = Vec::new();
    for name in names {
        let aligned: Vec<Option<ColumnData>> = vars
            .iter()
            .zip(&inverses)
            .map(|(v, inv)| {
                v.columns
                    .iter()
                    .find(|c| c.name == name)
                    .map(|c| reindex_column(&c.data, inv))
            })
            .collect();
        let present: Vec<usize> = aligned
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_some())
            .map(|(i, _)| i)
            .collect();

        let agree = |a: usize, b: usize| {
            let (x, y) = (
                col_str(aligned[a].as_ref().unwrap()),
                col_str(aligned[b].as_ref().unwrap()),
            );
            (0..n_out_vars)
                .all(|i| inverses[a][i].is_none() || inverses[b][i].is_none() || x[i] == y[i])
        };
        let all_agree = || present.windows(2).all(|w| agree(w[0], w[1]));

        let keep = match strategy {
            MergeStrategy::None => false,
            MergeStrategy::First => true,
            MergeStrategy::Only => present.len() == 1,
            MergeStrategy::Same => present.len() == vars.len() && all_agree(),
            MergeStrategy::Unique => all_agree(),
        };
        if keep {
            let first = present[0];
            out.push(Column {
                name,
                data: aligned[first].clone().unwrap(),
            });
        }
    }
    out
}

/// Stack obsm matrices row-wise. Keys follow the join; an input missing a key
/// under an outer join contributes NaN rows.
fn concat_obsm(obsms: &mut [Embeddings], lens: &[usize], join: Join) -> Result<Embeddings> {
    let keys = join_names(
        &obsms
            .iter()
            .map(|e| {
                let mut k: Vec<String> = e.map.keys().cloned().collect();
                k.sort();
                k
            })
            .collect::<Vec<_>>(),
        join,
    );
    let total: usize = lens.iter().sum();
    let mut out = Embeddings::default();
    for key in keys {
        // Take (not clone) each input's matrix — obsm is materialised, so peak
        // memory stays at one copy of the data.
        let parts: Vec<Option<DenseMatrix>> =
            obsms.iter_mut().map(|e| e.map.remove(&key)).collect();
        let n_dims = parts
            .iter()
            .flatten()
            .map(|m| m.shape.1)
            .next()
            .unwrap_or(0);
        if let Some(bad) = parts.iter().flatten().find(|m| m.shape.1 != n_dims) {
            return Err(ScxError::InvalidFormat(format!(
                "obsm['{key}'] has {} columns in one input and {n_dims} in another",
                bad.shape.1
            )));
        }
        let mut data = Vec::with_capacity(total * n_dims);
        for (part, &n) in parts.into_iter().zip(lens) {
            match part {
                Some(m) => data.extend(m.data),
                None => data.extend(std::iter::repeat_n(f64::NAN, n * n_dims)),
            }
        }
        out.map.insert(
            key,
            DenseMatrix {
                shape: (total, n_dims),
                data,
            },
        );
    }
    Ok(out)
}

fn provenance_json(
    inputs: &[String],
    keys: &[String],
    lens: &[usize],
    n_out_vars: usize,
    opts: &ConcatOptions,
) -> serde_json::Value {
    serde_json::json!({
        "scx_concat": {
            "scx_version": env!("CARGO_PKG_VERSION"),
            "created_at": crate::provenance::utc_now_rfc3339(),
            "join": match opts.join { Join::Inner => "inner", Join::Outer => "outer" },
            "label": opts.label,
            "index_unique": opts.index_unique,
            "n_vars": n_out_vars,
            "inputs": inputs.iter().zip(keys).zip(lens)
                .map(|((path, key), n_obs)| serde_json::json!({
                    "path": path, "key": key, "n_obs": n_obs,
                }))
                .collect::<Vec<_>>(),
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn join_index_inner_and_outer() {
        let a = sv(&["g1", "g2", "g3"]);
        let b = sv(&["g3", "g2", "g4"]);
        assert_eq!(
            join_names(&[a.clone(), b.clone()], Join::Inner),
            sv(&["g2", "g3"])
        );
        assert_eq!(
            join_names(&[a, b], Join::Outer),
            sv(&["g1", "g2", "g3", "g4"])
        );
    }

    #[test]
    fn varmap_identity_and_reorder() {
        let out = sv(&["g1", "g2", "g3"]);
        let m = VarMap::build(&out, &out);
        assert!(m.identity && !m.needs_sort);

        let src = sv(&["g3", "g1", "g9"]);
        let m = VarMap::build(&src, &out);
        assert_eq!(m.to_out, vec![Some(2), Some(0), None]);
        assert!(!m.identity && m.needs_sort);
    }

    /// Row 0 holds g3=7, g1=5, g9=9; g9 is not in the output axis so it drops,
    /// and the survivors come out sorted by output column.
    #[test]
    fn remap_drops_and_sorts() {
        let map = VarMap::build(&sv(&["g3", "g1", "g9"]), &sv(&["g1", "g2", "g3"]));
        let chunk = MatrixChunk {
            row_offset: 0,
            nrows: 1,
            data: SparseMatrixCSR {
                shape: (1, 3),
                indptr: vec![0, 3],
                indices: vec![0, 1, 2],
                data: TypedVec::F32(vec![7.0, 5.0, 9.0]),
            },
        };
        let out = remap_chunk(chunk, &map, 3);
        assert_eq!(out.data.indices, vec![0, 2]);
        assert_eq!(out.data.indptr, vec![0, 2]);
        match out.data.data {
            TypedVec::F32(v) => assert_eq!(v, vec![5.0, 7.0]),
            _ => panic!("dtype changed"),
        }
    }

    #[test]
    fn concat_column_promotes_and_na_fills() {
        // int + float -> float, missing -> NaN
        let a = ColumnData::Int(vec![1, 2]);
        let b = ColumnData::Float(vec![3.5]);
        match concat_column(&[Some(&a), Some(&b), None], &[2, 1, 2]) {
            ColumnData::Float(v) => {
                assert_eq!(v[..3], [1.0, 2.0, 3.5]);
                assert!(v[3].is_nan() && v[4].is_nan());
            }
            _ => panic!("expected float"),
        }

        // categorical + string -> categorical with unified levels
        let a = ColumnData::Categorical {
            codes: vec![0, 1],
            levels: sv(&["x", "y"]),
        };
        let b = ColumnData::String(sv(&["y", "z"]));
        match concat_column(&[Some(&a), Some(&b), None], &[2, 2, 1]) {
            ColumnData::Categorical { codes, levels } => {
                assert_eq!(levels, sv(&["x", "y", "z", "NA"]));
                assert_eq!(codes, vec![0, 1, 1, 2, 3]);
            }
            _ => panic!("expected categorical"),
        }
    }

    #[test]
    fn keys_default_to_file_stems() {
        let inputs = sv(&["/data/a.h5ad", "/other/b.h5ad"]);
        assert_eq!(resolve_keys(&inputs, &[]).unwrap(), sv(&["a", "b"]));
        assert!(resolve_keys(&sv(&["x/a.h5ad", "y/a.h5ad"]), &[]).is_err());
        assert!(resolve_keys(&inputs, &sv(&["only_one"])).is_err());
    }

    #[test]
    fn merge_var_columns_strategies() {
        let idx = sv(&["g1", "g2"]);
        let col = |name: &str, vals: &[&str]| Column {
            name: name.to_string(),
            data: ColumnData::String(sv(vals)),
        };
        let vars = vec![
            VarTable {
                index: idx.clone(),
                columns: vec![col("sym", &["A", "B"]), col("only_a", &["1", "2"])],
            },
            VarTable {
                index: idx.clone(),
                columns: vec![col("sym", &["A", "B"])],
            },
        ];
        let maps: Vec<VarMap> = vars.iter().map(|v| VarMap::build(&v.index, &idx)).collect();

        let names = |s: MergeStrategy| -> Vec<String> {
            merge_var_columns(&vars, &maps, 2, s)
                .into_iter()
                .map(|c| c.name)
                .collect()
        };
        assert_eq!(names(MergeStrategy::None), Vec::<String>::new());
        assert_eq!(names(MergeStrategy::Same), sv(&["sym"]));
        assert_eq!(names(MergeStrategy::Only), sv(&["only_a"]));
        assert_eq!(names(MergeStrategy::First), sv(&["sym", "only_a"]));
        assert_eq!(names(MergeStrategy::Unique), sv(&["sym", "only_a"]));
    }
}
