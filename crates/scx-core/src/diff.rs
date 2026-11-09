//! `scx diff`: compare two datasets slot by slot. See docs/diff.md.
//!
//! [`diff`] takes two readers, so it compares any pair of formats scx reads.
//! Metadata is read whole; matrices are streamed a chunk per side, so memory
//! stays about what `convert` uses.

use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use serde::Serialize;

use crate::error::Result;
use crate::ir::{Column, ColumnData, DenseMatrix, MatrixChunk, SparseMatrixMeta, UnsTable};
use crate::npy::SlotFilter;
use crate::stream::{ChunkStream, DatasetReader};

/// How cells and genes of A and B are paired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchBy {
    /// By index name; falls back to position when names repeat.
    Name,
    /// Row i of A against row i of B.
    Position,
}

#[derive(Debug)]
pub struct DiffOptions {
    /// Floats are equal when `|a - b| <= atol + rtol * |b|` (numpy `isclose`).
    pub rtol: f64,
    pub atol: f64,
    pub by: MatchBy,
    /// Mismatching entries listed per slot.
    pub max_examples: usize,
    /// Rows per streamed chunk.
    pub chunk_size: usize,
    /// Slots to compare (`convert --only/--exclude` syntax).
    pub filter: SlotFilter,
    /// Matrix slot pairs whose names differ, as (slot in A, slot in B):
    /// `X`, `raw`, `layers/<name>`, `obsp/<name>`.
    pub map: Vec<(String, String)>,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 0.0,
            by: MatchBy::Name,
            max_examples: 5,
            chunk_size: 5000,
            filter: SlotFilter::all(),
            map: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    Same,
    Diff,
    OnlyA,
    OnlyB,
    /// Present on both sides but not compared (e.g. cells in another order).
    Skipped,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Same => "SAME",
            Status::Diff => "DIFF",
            Status::OnlyA => "ONLY A",
            Status::OnlyB => "ONLY B",
            Status::Skipped => "SKIPPED",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotDiff {
    pub slot: String,
    pub status: Status,
    pub summary: String,
    /// Specific differences, at most `max_examples` of each kind.
    pub details: Vec<String>,
    /// What conversion legitimately changes (dtype, level order): reported,
    /// but not a difference.
    pub notes: Vec<String>,
}

impl SlotDiff {
    fn new(slot: impl Into<String>, status: Status, summary: impl Into<String>) -> Self {
        Self {
            slot: slot.into(),
            status,
            summary: summary.into(),
            details: Vec::new(),
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiffReport {
    pub slots: Vec<SlotDiff>,
}

impl DiffReport {
    /// Every compared slot is the same (notes allowed).
    pub fn equal(&self) -> bool {
        self.slots.iter().all(|s| s.status == Status::Same)
    }

    pub fn n_differing(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.status != Status::Same)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Compare A and B slot by slot.
pub async fn diff(
    a: &mut (dyn DatasetReader + Send),
    b: &mut (dyn DatasetReader + Send),
    opts: &DiffOptions,
) -> Result<DiffReport> {
    let mut out = Vec::new();
    let f = &opts.filter;

    let (sa, sb) = (a.shape(), b.shape());
    let same_shape = sa == sb;
    out.push(SlotDiff::new(
        "shape",
        if same_shape {
            Status::Same
        } else {
            Status::Diff
        },
        if same_shape {
            format!("{} x {}", sa.0, sa.1)
        } else {
            format!("{} x {} vs {} x {}", sa.0, sa.1, sb.0, sb.1)
        },
    ));

    let (obs_a, obs_b) = (a.obs().await?, b.obs().await?);
    let (var_a, var_b) = (a.var().await?, b.var().await?);
    let (rows, row_slot) = align("obs index", &obs_a.index, &obs_b.index, opts.by);
    let (cols, col_slot) = align("var index", &var_a.index, &var_b.index, opts.by);
    if f.includes("obs_index") {
        out.push(row_slot);
    }
    if f.includes("var_index") {
        out.push(col_slot);
    }
    if f.includes_any("obs") {
        out.push(compare_columns(
            "obs",
            &obs_a.columns,
            &obs_b.columns,
            &rows,
            opts,
        ));
    }
    if f.includes_any("var") {
        out.push(compare_columns(
            "var",
            &var_a.columns,
            &var_b.columns,
            &cols,
            opts,
        ));
    }
    if f.includes_any("obsm") {
        let (ma, mb) = (a.obsm().await?.map, b.obsm().await?.map);
        out.extend(compare_dense("obsm", &ma, &mb, &rows, opts));
    }
    if f.includes_any("varm") {
        let (ma, mb) = (a.varm().await?.map, b.varm().await?.map);
        out.extend(compare_dense("varm", &ma, &mb, &cols, opts));
    }
    if f.includes("uns") {
        let (mut ua, mut ub) = (a.uns().await?, b.uns().await?);
        // A side read from JSON (npy) holds large arrays in raw.
        ua.split_large();
        ub.split_large();
        out.push(compare_uns(&ua, &ub, opts));
    }
    out.extend(compare_matrices(a, b, &rows, &cols, same_shape, opts).await?);
    Ok(DiffReport { slots: out })
}

// ---------------------------------------------------------------------------
// Index alignment
// ---------------------------------------------------------------------------

/// Where each of A's rows is in B.
#[derive(Debug, Clone)]
struct Align {
    b_of_a: Vec<Option<usize>>,
    /// A's row i is B's row i, and B has no other rows.
    identity: bool,
}

fn align(slot: &str, a: &[String], b: &[String], by: MatchBy) -> (Align, SlotDiff) {
    let unique = |v: &[String]| v.iter().collect::<HashSet<_>>().len() == v.len();
    let mut d;
    let by_name = by == MatchBy::Name && unique(a) && unique(b);
    let b_of_a: Vec<Option<usize>> = if by_name {
        let pos: HashMap<&String, usize> = b.iter().enumerate().map(|(i, n)| (n, i)).collect();
        a.iter().map(|n| pos.get(n).copied()).collect()
    } else {
        (0..a.len()).map(|i| (i < b.len()).then_some(i)).collect()
    };
    let identity = a.len() == b.len() && b_of_a.iter().enumerate().all(|(i, &j)| j == Some(i));
    if by_name {
        let only_a: Vec<&String> = a
            .iter()
            .zip(&b_of_a)
            .filter(|(_, j)| j.is_none())
            .map(|(n, _)| n)
            .collect();
        let in_a: HashSet<&String> = a.iter().collect();
        let only_b: Vec<&String> = b.iter().filter(|n| !in_a.contains(n)).collect();
        if identity {
            d = SlotDiff::new(slot, Status::Same, format!("{} names, same order", a.len()));
        } else if only_a.is_empty() && only_b.is_empty() {
            d = SlotDiff::new(slot, Status::Same, format!("{} names", a.len()));
            d.notes
                .push("same names in another order; compared by name".into());
        } else {
            d = SlotDiff::new(
                slot,
                Status::Diff,
                format!(
                    "{} shared, {} only in A, {} only in B",
                    a.len() - only_a.len(),
                    only_a.len(),
                    only_b.len()
                ),
            );
            if !only_a.is_empty() {
                d.details.push(format!("only in A: {}", preview(&only_a)));
            }
            if !only_b.is_empty() {
                d.details.push(format!("only in B: {}", preview(&only_b)));
            }
        }
    } else {
        let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
        d = if a == b {
            SlotDiff::new(slot, Status::Same, format!("{} names, same order", a.len()))
        } else if by == MatchBy::Position {
            let mut d = SlotDiff::new(
                slot,
                Status::Same,
                format!("{} vs {} names, compared by position", a.len(), b.len()),
            );
            d.notes
                .push(format!("{differ} names differ at the same position"));
            d
        } else {
            let mut d = SlotDiff::new(
                slot,
                Status::Diff,
                format!("{differ} of {} names differ", a.len()),
            );
            d.notes
                .push("names repeat, so they are matched by position".into());
            d
        };
        if a.len() != b.len() && d.status == Status::Same {
            d.status = Status::Diff;
        }
    }
    (Align { b_of_a, identity }, d)
}

fn preview(names: &[&String]) -> String {
    let shown: Vec<&str> = names.iter().take(5).map(|s| s.as_str()).collect();
    let more = names.len().saturating_sub(5);
    if more > 0 {
        format!("{} and {more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/// A difference for the report: `0`, or three significant digits.
fn num(x: f64) -> String {
    if x == 0.0 {
        "0".into()
    } else {
        format!("{x:.3e}")
    }
}

fn close(a: f64, b: f64, o: &DiffOptions) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= o.atol + o.rtol * b.abs()
}

/// One cell of a column, by class rather than storage type.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Na,
    Num(f64),
    Int(i64),
    Bool(bool),
    Str(String),
}

impl std::fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cell::Na => write!(f, "NA"),
            Cell::Num(x) => write!(f, "{x}"),
            Cell::Int(x) => write!(f, "{x}"),
            Cell::Bool(x) => write!(f, "{x}"),
            Cell::Str(x) => write!(f, "{x:?}"),
        }
    }
}

fn cell(c: &Column, i: usize) -> Cell {
    if c.mask.as_ref().is_some_and(|m| !m[i]) {
        return Cell::Na;
    }
    match &c.data {
        ColumnData::Int(v) => Cell::Int(v[i] as i64),
        ColumnData::Int64(v) => Cell::Int(v[i]),
        // A float NA is NaN in most formats.
        ColumnData::Float(v) if v[i].is_nan() => Cell::Na,
        ColumnData::Float(v) => Cell::Num(v[i]),
        ColumnData::Bool(v) => Cell::Bool(v[i]),
        ColumnData::String(v) => Cell::Str(v[i].clone()),
        ColumnData::Categorical { codes, levels, .. } => levels
            .get(codes[i] as usize)
            .map_or(Cell::Na, |l| Cell::Str(l.clone())),
    }
}

fn cells_equal(a: &Cell, b: &Cell, o: &DiffOptions) -> bool {
    match (a, b) {
        (Cell::Num(x), Cell::Num(y)) => close(*x, *y, o),
        (Cell::Num(x), Cell::Int(y)) | (Cell::Int(y), Cell::Num(x)) => *x == *y as f64,
        (Cell::Bool(x), Cell::Int(y)) | (Cell::Int(y), Cell::Bool(x)) => *x as i64 == *y,
        _ => a == b,
    }
}

fn class(d: &ColumnData) -> &'static str {
    match d {
        ColumnData::Int(_) | ColumnData::Int64(_) => "integer",
        ColumnData::Float(_) => "float",
        ColumnData::Bool(_) => "bool",
        ColumnData::String(_) => "string",
        ColumnData::Categorical { .. } => "categorical",
    }
}

// ---------------------------------------------------------------------------
// obs / var columns
// ---------------------------------------------------------------------------

fn compare_columns(
    slot: &str,
    a: &[Column],
    b: &[Column],
    rows: &Align,
    o: &DiffOptions,
) -> SlotDiff {
    let in_filter = |c: &Column| o.filter.includes(&format!("{slot}:{}", c.name));
    let by_name: HashMap<&str, &Column> = b.iter().map(|c| (c.name.as_str(), c)).collect();
    let in_a: HashSet<&str> = a.iter().map(|c| c.name.as_str()).collect();
    let (mut same, mut differ, mut only_a) = (0, Vec::new(), Vec::new());
    let mut notes = Vec::new();
    for ca in a.iter().filter(|c| in_filter(c)) {
        let Some(cb) = by_name.get(ca.name.as_str()) else {
            only_a.push(&ca.name);
            continue;
        };
        let mut n_diff = 0;
        let mut example = None;
        for (i, j) in rows.b_of_a.iter().enumerate() {
            let Some(j) = *j else { continue };
            let (x, y) = (cell(ca, i), cell(cb, j));
            if !cells_equal(&x, &y, o) {
                n_diff += 1;
                example.get_or_insert((i, x, y));
            }
        }
        if let (
            ColumnData::Categorical {
                levels: la,
                ordered: oa,
                level_type: ta,
                ..
            },
            ColumnData::Categorical {
                levels: lb,
                ordered: ob,
                level_type: tb,
                ..
            },
        ) = (&ca.data, &cb.data)
        {
            if oa != ob {
                n_diff = n_diff.max(1);
                differ.push(format!("{}: ordered {oa} vs {ob}", ca.name));
            } else if la != lb && n_diff == 0 {
                notes.push(format!("{}: same values, levels in another order", ca.name));
            }
            // h5seurat and R store levels as strings: a note, like a dtype.
            if ta != tb {
                notes.push(format!(
                    "{}: {} levels vs {} levels",
                    ca.name,
                    ta.as_str(),
                    tb.as_str()
                ));
            }
        }
        if n_diff == 0 {
            same += 1;
            let (ka, kb) = (class(&ca.data), class(&cb.data));
            if ka != kb {
                notes.push(format!("{}: {ka} vs {kb}, same values", ca.name));
            }
        } else if let Some((i, x, y)) = example {
            differ.push(format!(
                "{}: {n_diff} of {} differ, e.g. row {i}: {x} vs {y}",
                ca.name,
                rows.b_of_a.len()
            ));
        }
    }
    let only_b: Vec<&String> = b
        .iter()
        .filter(|c| in_filter(c) && !in_a.contains(c.name.as_str()))
        .map(|c| &c.name)
        .collect();
    let status = if differ.is_empty() && only_a.is_empty() && only_b.is_empty() {
        Status::Same
    } else {
        Status::Diff
    };
    let mut d = SlotDiff::new(
        format!("{slot} columns"),
        status,
        format!(
            "{same} same, {} different, {} only in A, {} only in B",
            differ.len(),
            only_a.len(),
            only_b.len()
        ),
    );
    d.details.extend(differ.into_iter().take(o.max_examples));
    if !only_a.is_empty() {
        d.details.push(format!("only in A: {}", preview(&only_a)));
    }
    if !only_b.is_empty() {
        d.details.push(format!("only in B: {}", preview(&only_b)));
    }
    d.notes = notes;
    d
}

// ---------------------------------------------------------------------------
// obsm / varm
// ---------------------------------------------------------------------------

fn compare_dense(
    slot: &str,
    a: &HashMap<String, DenseMatrix>,
    b: &HashMap<String, DenseMatrix>,
    rows: &Align,
    o: &DiffOptions,
) -> Vec<SlotDiff> {
    let mut keys: Vec<&String> = a
        .keys()
        .chain(b.keys())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    keys.sort();
    let mut out = Vec::new();
    for key in keys {
        if !o.filter.includes(&format!("{slot}:{key}")) {
            continue;
        }
        let name = format!("{slot}/{key}");
        let (ma, mb) = match (a.get(key), b.get(key)) {
            (Some(x), Some(y)) => (x, y),
            (Some(x), None) if x.data.is_empty() => continue,
            (None, Some(y)) if y.data.is_empty() => continue,
            (Some(x), None) => {
                out.push(SlotDiff::new(
                    name,
                    Status::OnlyA,
                    format!("{} x {}", x.shape.0, x.shape.1),
                ));
                continue;
            }
            (None, Some(y)) => {
                out.push(SlotDiff::new(
                    name,
                    Status::OnlyB,
                    format!("{} x {}", y.shape.0, y.shape.1),
                ));
                continue;
            }
            (None, None) => continue,
        };
        let k = ma.shape.1;
        if mb.shape.1 != k || ma.shape.0 != rows.b_of_a.len() {
            out.push(SlotDiff::new(
                name,
                Status::Diff,
                format!(
                    "shape {} x {} vs {} x {}",
                    ma.shape.0, ma.shape.1, mb.shape.0, mb.shape.1
                ),
            ));
            continue;
        }
        let (mut n_diff, mut max_abs) = (0usize, 0f64);
        let mut details = Vec::new();
        for (i, j) in rows.b_of_a.iter().enumerate() {
            let Some(j) = *j else { continue };
            if j >= mb.shape.0 {
                continue;
            }
            for c in 0..k {
                let (x, y) = (ma.data[i * k + c], mb.data[j * k + c]);
                if !(x.is_nan() && y.is_nan()) {
                    max_abs = max_abs.max((x - y).abs());
                }
                if !close(x, y, o) {
                    n_diff += 1;
                    if details.len() < o.max_examples {
                        details.push(format!("[{i}, {c}]: {x} vs {y}"));
                    }
                }
            }
        }
        let mut d = SlotDiff::new(
            name,
            if n_diff == 0 {
                Status::Same
            } else {
                Status::Diff
            },
            if n_diff == 0 {
                format!("{} x {k}, max abs diff {}", ma.shape.0, num(max_abs))
            } else {
                format!(
                    "{n_diff} of {} values differ, max abs diff {}",
                    ma.data.len(),
                    num(max_abs)
                )
            },
        );
        d.details = details;
        out.push(d);
    }
    out
}

// ---------------------------------------------------------------------------
// uns
// ---------------------------------------------------------------------------

fn compare_uns(ua: &UnsTable, ub: &UnsTable, o: &DiffOptions) -> SlotDiff {
    use serde_json::Value;
    fn strip(v: &Value) -> Value {
        match v {
            Value::Object(m) => Value::Object(
                m.iter()
                    .filter(|(k, _)| k.as_str() != "scx_provenance")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
            Value::Null => Value::Object(Default::default()),
            v => v.clone(),
        }
    }
    fn walk(path: &str, a: &Value, b: &Value, o: &DiffOptions, out: &mut Vec<String>) {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
                keys.sort();
                keys.dedup();
                for k in keys {
                    let p = if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}/{k}")
                    };
                    match (x.get(k), y.get(k)) {
                        (Some(u), Some(v)) => walk(&p, u, v, o, out),
                        (Some(_), None) => out.push(format!("{p}: only in A")),
                        (None, Some(_)) => out.push(format!("{p}: only in B")),
                        (None, None) => {}
                    }
                }
            }
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
                for (i, (u, v)) in x.iter().zip(y).enumerate() {
                    walk(&format!("{path}[{i}]"), u, v, o, out);
                }
            }
            (Value::Number(x), Value::Number(y)) => {
                let (x, y) = (
                    x.as_f64().unwrap_or(f64::NAN),
                    y.as_f64().unwrap_or(f64::NAN),
                );
                if !close(x, y, o) {
                    out.push(format!("{path}: {x} vs {y}"));
                }
            }
            (x, y) if x == y => {}
            (x, y) => out.push(format!("{path}: {} vs {}", short(x), short(y))),
        }
    }
    fn short(v: &Value) -> String {
        let s = v.to_string();
        if s.len() > 60 {
            format!(
                "{}…",
                &s[..s.char_indices().nth(60).map_or(s.len(), |(i, _)| i)]
            )
        } else {
            s
        }
    }
    let (a, b) = (strip(&ua.raw), strip(&ub.raw));
    let mut found = Vec::new();
    walk("", &a, &b, o, &mut found);
    let mut paths: Vec<&String> = ua.arrays.keys().chain(ub.arrays.keys()).collect();
    paths.sort();
    paths.dedup();
    for p in paths {
        match (ua.arrays.get(p), ub.arrays.get(p)) {
            (Some(x), Some(y)) if x.shape != y.shape => {
                found.push(format!("{p}: shape {:?} vs {:?}", x.shape, y.shape))
            }
            // Same type and values: one memcmp, even for an image.
            (Some(x), Some(y)) if x.data == y.data => {}
            (Some(x), Some(y)) => {
                let (u, v) = (&x.data, &y.data);
                let mut bad = (0..u.len()).filter(|&i| !close(u.at(i), v.at(i), o));
                if let Some(i) = bad.next() {
                    found.push(format!(
                        "{p}: {} values differ, first at {i}: {} vs {}",
                        1 + bad.count(),
                        u.at(i),
                        v.at(i)
                    ));
                }
            }
            (Some(_), None) => found.push(format!("{p}: only in A")),
            (None, Some(_)) => found.push(format!("{p}: only in B")),
            (None, None) => {}
        }
    }
    let n_keys = a.as_object().map_or(0, |m| m.len());
    let mut d = SlotDiff::new(
        "uns",
        if found.is_empty() {
            Status::Same
        } else {
            Status::Diff
        },
        if found.is_empty() {
            format!("{n_keys} keys")
        } else {
            format!("{} differences", found.len())
        },
    );
    d.details = found.into_iter().take(o.max_examples).collect();
    d
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

/// A sparse matrix of a reader, by its diff name.
#[derive(Debug, Clone)]
enum Slot {
    X,
    Raw(SparseMatrixMeta),
    Layer(SparseMatrixMeta),
    Obsp(SparseMatrixMeta),
}

impl Slot {
    fn name(&self) -> String {
        match self {
            Slot::X => "X".into(),
            Slot::Raw(_) => "raw".into(),
            Slot::Layer(m) => format!("layers/{}", m.name),
            Slot::Obsp(m) => format!("obsp/{}", m.name),
        }
    }

    /// Spec for the slot filter.
    fn filter_key(&self) -> String {
        match self {
            Slot::X | Slot::Raw(_) => "X".into(),
            Slot::Layer(m) => format!("layers:{}", m.name),
            Slot::Obsp(m) => format!("obsp:{}", m.name),
        }
    }

    fn describe(&self) -> String {
        match self {
            Slot::X => String::new(),
            Slot::Raw(m) | Slot::Layer(m) | Slot::Obsp(m) => format!(
                "{} x {}, {} nnz",
                m.shape.0,
                m.shape.1,
                m.indptr.last().copied().unwrap_or(0)
            ),
        }
    }

    /// Stored nonzeros, when known without reading the matrix.
    fn nnz(&self) -> Option<u64> {
        match self {
            Slot::X => None,
            Slot::Raw(m) | Slot::Layer(m) | Slot::Obsp(m) => m.indptr.last().copied(),
        }
    }

    fn stream<'r>(
        &'r self,
        r: &'r mut (dyn DatasetReader + Send),
        chunk: usize,
    ) -> ChunkStream<'r> {
        match self {
            // TODO: holds all of a gene-major X; compare by gene when both
            // sides are gene-major if that ever matters at atlas scale.
            Slot::X if !r.x_col_ptr().is_empty() => {
                tracing::warn!(
                    "X is gene-major (CSC): reading all of it into memory to compare by cell"
                );
                crate::transpose::x_cell_major_stream(r, chunk)
            }
            Slot::X => r.x_stream(),
            Slot::Raw(m) => r.raw_stream(m, chunk),
            Slot::Layer(m) => r.layer_stream(m, chunk),
            Slot::Obsp(m) => r.obsp_stream(m, chunk),
        }
    }
}

async fn slots(r: &mut (dyn DatasetReader + Send), f: &SlotFilter) -> Result<Vec<Slot>> {
    let mut v = vec![Slot::X];
    if let Some(m) = r.raw_meta().await? {
        v.push(Slot::Raw(m));
    }
    if f.includes_any("layers") {
        v.extend(r.layer_metas().await?.into_iter().map(Slot::Layer));
    }
    if f.includes_any("obsp") {
        v.extend(r.obsp_metas().await?.into_iter().map(Slot::Obsp));
    }
    Ok(v.into_iter()
        .filter(|s| f.includes(&s.filter_key()))
        .collect())
}

/// Pair A's and B's matrices: `--map` first, then the counts/data rule
/// `scx convert` applies to h5seurat (`h5seurat::auto_counts_source`): when
/// one side keeps its counts in raw.X or a `counts` layer and the other has
/// a `data` layer instead, X pairs with `layers/data` and the counts with
/// the other side's X. Then equal names.
fn pair(a: &[Slot], b: &[Slot], map: &[(String, String)]) -> Vec<(Option<usize>, Option<usize>)> {
    let find = |v: &[Slot], n: &str| v.iter().position(|s| s.name() == n);
    let mut pairs: Vec<(usize, usize)> = map
        .iter()
        .filter_map(|(x, y)| Some((find(a, x)?, find(b, y)?)))
        .collect();
    let has = |v: &[Slot], n: &str| find(v, n).is_some();
    // h5ad (X normalised, counts in raw or a layer) vs h5seurat (X = counts,
    // layers/data), either way round.
    for (p, q, flip) in [(a, b, false), (b, a, true)] {
        let Some(counts) = ["raw", "layers/counts"]
            .into_iter()
            .find(|c| has(p, c) && !has(q, c))
        else {
            continue;
        };
        if has(q, "layers/data") && !has(p, "layers/data") {
            for (x, y) in [("X", "layers/data"), (counts, "X")] {
                let (i, j) = (find(p, x).unwrap(), find(q, y).unwrap());
                let pr = if flip { (j, i) } else { (i, j) };
                if !pairs.iter().any(|&(u, w)| u == pr.0 || w == pr.1) {
                    pairs.push(pr);
                }
            }
        }
    }
    for (i, s) in a.iter().enumerate() {
        if let Some(j) = find(b, &s.name()) {
            if !pairs.iter().any(|&(u, w)| u == i || w == j) {
                pairs.push((i, j));
            }
        }
    }
    let mut out: Vec<(Option<usize>, Option<usize>)> =
        pairs.iter().map(|&(i, j)| (Some(i), Some(j))).collect();
    out.extend(
        (0..a.len())
            .filter(|i| !pairs.iter().any(|p| p.0 == *i))
            .map(|i| (Some(i), None)),
    );
    out.extend(
        (0..b.len())
            .filter(|j| !pairs.iter().any(|p| p.1 == *j))
            .map(|j| (None, Some(j))),
    );
    out
}

async fn compare_matrices(
    a: &mut (dyn DatasetReader + Send),
    b: &mut (dyn DatasetReader + Send),
    rows: &Align,
    cols: &Align,
    same_shape: bool,
    o: &DiffOptions,
) -> Result<Vec<SlotDiff>> {
    let (sa, sb) = (slots(a, &o.filter).await?, slots(b, &o.filter).await?);
    let mut out = Vec::new();
    for (i, j) in pair(&sa, &sb, &o.map) {
        let d = match (i.map(|i| &sa[i]), j.map(|j| &sb[j])) {
            (Some(x), None) if x.nnz() == Some(0) => continue,
            (None, Some(y)) if y.nnz() == Some(0) => continue,
            (Some(x), None) => SlotDiff::new(x.name(), Status::OnlyA, x.describe()),
            (None, Some(y)) => SlotDiff::new(y.name(), Status::OnlyB, y.describe()),
            (Some(x), Some(y)) => {
                let name = if x.name() == y.name() {
                    x.name()
                } else {
                    format!("{} (vs {})", x.name(), y.name())
                };
                // obsp is cells x cells: its columns follow the rows.
                let col_align = if matches!(x, Slot::Obsp(_)) {
                    rows
                } else {
                    cols
                };
                if !same_shape && matches!(x, Slot::X) {
                    SlotDiff::new(name, Status::Skipped, "shapes differ")
                } else if !rows.identity {
                    let mut d = SlotDiff::new(name, Status::Skipped, "cells in another order");
                    d.details
                        .push("reorder one file, or compare with --by position".into());
                    d
                } else {
                    let mut d = compare_matrix(
                        x.stream(a, o.chunk_size),
                        y.stream(b, o.chunk_size),
                        col_align,
                        o,
                    )
                    .await?;
                    d.slot = name;
                    d
                }
            }
            (None, None) => continue,
        };
        out.push(d);
    }
    Ok(out)
}

/// Rows of a chunk stream, one at a time, as (columns, values).
struct Rows<'a> {
    stream: ChunkStream<'a>,
    chunk: Option<(MatrixChunk, Vec<f64>)>,
    next: usize,
}

impl<'a> Rows<'a> {
    fn new(stream: ChunkStream<'a>) -> Self {
        Self {
            stream,
            chunk: None,
            next: 0,
        }
    }

    async fn row(&mut self) -> Result<Option<(Vec<u32>, Vec<f64>)>> {
        loop {
            if let Some((c, values)) = &self.chunk {
                if self.next < c.nrows {
                    let span =
                        c.data.indptr[self.next] as usize..c.data.indptr[self.next + 1] as usize;
                    self.next += 1;
                    return Ok(Some((
                        c.data.indices[span.clone()].to_vec(),
                        values[span].to_vec(),
                    )));
                }
            }
            match self.stream.next().await {
                Some(c) => {
                    let c = c?;
                    let values = c.data.data.to_f64();
                    self.chunk = Some((c, values));
                    self.next = 0;
                }
                None => return Ok(None),
            }
        }
    }
}

/// Compare two matrices row by row (rows already aligned). `cols` maps A's
/// columns to B's; B's entries are brought to A's columns and sorted.
async fn compare_matrix(
    sa: ChunkStream<'_>,
    sb: ChunkStream<'_>,
    cols: &Align,
    o: &DiffOptions,
) -> Result<SlotDiff> {
    // B column -> A column.
    let mut a_of_b: HashMap<usize, usize> = HashMap::new();
    for (i, j) in cols.b_of_a.iter().enumerate() {
        if let Some(j) = j {
            a_of_b.insert(*j, i);
        }
    }
    let shared = |c: usize| cols.b_of_a.get(c).is_some_and(|j| j.is_some());
    let (mut ra, mut rb) = (Rows::new(sa), Rows::new(sb));
    let (mut nnz_a, mut nnz_b, mut n_diff, mut row) = (0u64, 0u64, 0u64, 0usize);
    let (mut max_abs, mut max_rel) = (0f64, 0f64);
    let mut details = Vec::new();
    loop {
        let (x, y) = (ra.row().await?, rb.row().await?);
        let ((ia, va), (ib, vb)) = match (x, y) {
            (None, None) => break,
            (Some(x), Some(y)) => (x, y),
            _ => {
                details.push(format!("row counts differ after row {row}"));
                n_diff += 1;
                break;
            }
        };
        nnz_a += ia.len() as u64;
        nnz_b += ib.len() as u64;
        // A's row on shared columns; B's row mapped to A's columns.
        let mut ea: Vec<(usize, f64)> = ia
            .iter()
            .zip(&va)
            .map(|(&c, &v)| (c as usize, v))
            .filter(|(c, _)| shared(*c))
            .collect();
        let mut eb: Vec<(usize, f64)> = ib
            .iter()
            .zip(&vb)
            .filter_map(|(&c, &v)| a_of_b.get(&(c as usize)).map(|&c| (c, v)))
            .collect();
        ea.sort_by_key(|e| e.0);
        eb.sort_by_key(|e| e.0);
        // Merge; an entry missing on one side is a zero there.
        let (mut p, mut q) = (0, 0);
        while p < ea.len() || q < eb.len() {
            let (col, x, y) = match (ea.get(p), eb.get(q)) {
                (Some(&(c, x)), Some(&(d, y))) if c == d => {
                    p += 1;
                    q += 1;
                    (c, x, y)
                }
                (Some(&(c, x)), Some(&(d, _))) if c < d => {
                    p += 1;
                    (c, x, 0.0)
                }
                (Some(&(c, x)), None) => {
                    p += 1;
                    (c, x, 0.0)
                }
                (_, Some(&(d, y))) => {
                    q += 1;
                    (d, 0.0, y)
                }
                (None, None) => unreachable!(),
            };
            if x.is_nan() && y.is_nan() {
                continue;
            }
            let abs = (x - y).abs();
            max_abs = max_abs.max(abs);
            if y != 0.0 {
                max_rel = max_rel.max(abs / y.abs());
            }
            if !close(x, y, o) {
                n_diff += 1;
                if details.len() < o.max_examples {
                    details.push(format!("[{row}, {col}]: {x} vs {y}"));
                }
            }
        }
        row += 1;
    }
    let mut d = SlotDiff::new(
        "",
        if n_diff == 0 {
            Status::Same
        } else {
            Status::Diff
        },
        if n_diff == 0 {
            format!("{nnz_a} nnz, max abs diff {}", num(max_abs))
        } else {
            format!(
                "{n_diff} entries differ; nnz {nnz_a} vs {nnz_b}, max abs diff {}, max rel diff {}",
                num(max_abs),
                num(max_rel)
            )
        },
    );
    if n_diff == 0 && nnz_a != nnz_b {
        d.notes.push(format!(
            "nnz {nnz_a} vs {nnz_b}: the difference is stored zeros"
        ));
    }
    d.details = details;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{DataType, TypedVec};
    use crate::ir::{Embeddings, ObsTable, SparseMatrixCSR, UnsTable, VarTable};

    /// An in-memory dataset; X is dense rows, streamed as CSR (zeros
    /// dropped unless `keep_zeros`).
    #[derive(Clone)]
    struct Mem {
        x: Vec<Vec<f64>>,
        obs: ObsTable,
        var: VarTable,
        obsm: Embeddings,
        uns: serde_json::Value,
        keep_zeros: bool,
    }

    #[async_trait::async_trait]
    impl DatasetReader for Mem {
        fn shape(&self) -> (usize, usize) {
            (self.obs.index.len(), self.var.index.len())
        }
        fn dtype(&self) -> DataType {
            DataType::F64
        }
        async fn obs(&mut self) -> Result<ObsTable> {
            Ok(self.obs.clone())
        }
        async fn var(&mut self) -> Result<VarTable> {
            Ok(self.var.clone())
        }
        async fn obsm(&mut self) -> Result<Embeddings> {
            Ok(self.obsm.clone())
        }
        async fn uns(&mut self) -> Result<UnsTable> {
            Ok(UnsTable {
                raw: self.uns.clone(),
                ..Default::default()
            })
        }
        fn x_stream(&mut self) -> ChunkStream<'_> {
            let n_vars = self.var.index.len();
            // One row per chunk: chunk boundaries never line up with A's.
            let chunks: Vec<_> = self
                .x
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    let (mut idx, mut val) = (vec![], vec![]);
                    for (c, &v) in row.iter().enumerate() {
                        if v != 0.0 || self.keep_zeros {
                            idx.push(c as u32);
                            val.push(v);
                        }
                    }
                    Ok(MatrixChunk {
                        row_offset: r,
                        nrows: 1,
                        data: SparseMatrixCSR {
                            shape: (1, n_vars),
                            indptr: vec![0, idx.len() as u64],
                            indices: idx,
                            data: TypedVec::F64(val),
                        },
                    })
                })
                .collect();
            futures::stream::iter(chunks).boxed()
        }
    }

    fn base() -> Mem {
        let mut obsm = Embeddings::default();
        obsm.map.insert(
            "X_pca".into(),
            DenseMatrix {
                shape: (3, 2),
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );
        Mem {
            x: vec![
                vec![1.0, 0.0, 2.0],
                vec![0.0, 0.0, 0.0],
                vec![3.0, 4.0, 0.0],
            ],
            obs: ObsTable {
                index: vec!["c0".into(), "c1".into(), "c2".into()],
                columns: vec![
                    Column {
                        name: "type".into(),
                        data: ColumnData::Categorical {
                            codes: vec![0, 1, 0],
                            levels: vec!["B".into(), "T".into()],
                            ordered: false,
                            level_type: crate::ir::LevelType::Str,
                        },
                        mask: None,
                    },
                    Column {
                        name: "n".into(),
                        data: ColumnData::Int64(vec![10, 20, 30]),
                        mask: None,
                    },
                    Column {
                        name: "score".into(),
                        data: ColumnData::Float(vec![0.5, f64::NAN, 1.5]),
                        mask: None,
                    },
                ],
            },
            var: VarTable {
                index: vec!["g0".into(), "g1".into(), "g2".into()],
                columns: vec![],
            },
            obsm,
            uns: serde_json::json!({"title": "t", "params": {"k": 15, "res": 0.5}}),
            keep_zeros: false,
        }
    }

    fn run(a: &Mem, b: &Mem, opts: &DiffOptions) -> DiffReport {
        let (mut a, mut b) = (a.clone(), b.clone());
        futures::executor::block_on(diff(&mut a, &mut b, opts)).unwrap()
    }

    fn slot<'r>(r: &'r DiffReport, name: &str) -> &'r SlotDiff {
        r.slots
            .iter()
            .find(|s| s.slot == name)
            .unwrap_or_else(|| panic!("no slot {name} in {:#?}", r.slots))
    }

    #[test]
    fn identical_datasets_are_equal() {
        let r = run(&base(), &base(), &DiffOptions::default());
        assert!(r.equal(), "{:#?}", r.slots);
    }

    #[test]
    fn a_changed_value_is_reported_with_its_position() {
        let mut b = base();
        b.x[2][1] = 4.5;
        let r = run(&base(), &b, &DiffOptions::default());
        let x = slot(&r, "X");
        assert_eq!(x.status, Status::Diff);
        assert_eq!(x.details, ["[2, 1]: 4 vs 4.5"]);
        assert!(!r.equal());
    }

    #[test]
    fn tolerance_decides_small_differences() {
        let mut b = base();
        b.x[0][0] = 1.0 + 1e-8;
        assert_eq!(
            slot(&run(&base(), &b, &DiffOptions::default()), "X").status,
            Status::Same
        );
        let exact = DiffOptions {
            rtol: 0.0,
            ..Default::default()
        };
        assert_eq!(slot(&run(&base(), &b, &exact), "X").status, Status::Diff);
    }

    #[test]
    fn an_extra_nonzero_differs_but_a_stored_zero_does_not() {
        let mut extra = base();
        extra.x[1][1] = 7.0;
        let x = run(&base(), &extra, &DiffOptions::default());
        assert_eq!(slot(&x, "X").details, ["[1, 1]: 0 vs 7"]);

        let mut zeros = base();
        zeros.keep_zeros = true;
        let r = run(&base(), &zeros, &DiffOptions::default());
        let x = slot(&r, "X");
        assert_eq!(x.status, Status::Same);
        assert!(x.notes[0].contains("stored zeros"), "{:?}", x.notes);
    }

    #[test]
    fn recoded_categoricals_are_equal_with_a_note() {
        let mut b = base();
        b.obs.columns[0].data = ColumnData::Categorical {
            codes: vec![1, 0, 1],
            levels: vec!["T".into(), "B".into()],
            ordered: false,
            level_type: crate::ir::LevelType::Str,
        };
        let r = run(&base(), &b, &DiffOptions::default());
        let cols = slot(&r, "obs columns");
        assert_eq!(cols.status, Status::Same, "{cols:#?}");
        assert!(cols.notes[0].contains("levels in another order"));

        if let ColumnData::Categorical { ordered, .. } = &mut b.obs.columns[0].data {
            *ordered = true;
        }
        let cols = run(&base(), &b, &DiffOptions::default());
        assert!(slot(&cols, "obs columns").details[0].contains("ordered false vs true"));
    }

    #[test]
    fn storage_type_changes_are_notes_and_values_still_count() {
        let mut b = base();
        // int64 -> int32 (h5seurat), categorical -> string: same values.
        b.obs.columns[1].data = ColumnData::Int(vec![10, 20, 30]);
        b.obs.columns[0].data = ColumnData::String(vec!["B".into(), "T".into(), "B".into()]);
        // A float NA (NaN) against a masked value.
        b.obs.columns[2].mask = Some(vec![true, false, true]);
        let r = run(&base(), &b, &DiffOptions::default());
        let cols = slot(&r, "obs columns");
        assert_eq!(cols.status, Status::Same, "{cols:#?}");
        assert_eq!(cols.notes, ["type: categorical vs string, same values"]);

        b.obs.columns[1].data = ColumnData::Int(vec![10, 21, 30]);
        b.obs.columns.pop();
        let r = run(&base(), &b, &DiffOptions::default());
        let cols = slot(&r, "obs columns");
        assert_eq!(cols.status, Status::Diff);
        assert_eq!(
            cols.details,
            ["n: 1 of 3 differ, e.g. row 1: 20 vs 21", "only in A: score"]
        );
    }

    #[test]
    fn permuted_cells_realign_metadata_and_skip_matrices() {
        // B lists the cells as c2, c0, c1.
        let a = base();
        let mut b = base();
        let perm = [2, 0, 1];
        b.obs.index = perm.iter().map(|&i| a.obs.index[i].clone()).collect();
        b.x = perm.iter().map(|&i| a.x[i].clone()).collect();
        for col in &mut b.obs.columns {
            col.data = match &col.data {
                ColumnData::Categorical {
                    codes,
                    levels,
                    ordered,
                    ..
                } => ColumnData::Categorical {
                    codes: perm.iter().map(|&i| codes[i]).collect(),
                    levels: levels.clone(),
                    ordered: *ordered,
                    level_type: crate::ir::LevelType::Str,
                },
                ColumnData::Int64(v) => ColumnData::Int64(perm.iter().map(|&i| v[i]).collect()),
                ColumnData::Float(v) => ColumnData::Float(perm.iter().map(|&i| v[i]).collect()),
                other => other.clone(),
            };
        }
        let pca = &a.obsm.map["X_pca"].data;
        b.obsm.map.get_mut("X_pca").unwrap().data = perm
            .iter()
            .flat_map(|&i| pca[i * 2..i * 2 + 2].to_vec())
            .collect();

        let r = run(&a, &b, &DiffOptions::default());
        assert_eq!(slot(&r, "obs index").status, Status::Same);
        assert_eq!(slot(&r, "obs columns").status, Status::Same);
        assert_eq!(slot(&r, "obsm/X_pca").status, Status::Same);
        assert_eq!(slot(&r, "X").status, Status::Skipped);
        assert!(!r.equal());

        let by_pos = DiffOptions {
            by: MatchBy::Position,
            ..Default::default()
        };
        let r = run(&a, &b, &by_pos);
        assert_eq!(slot(&r, "X").status, Status::Diff);
    }

    #[test]
    fn permuted_genes_are_compared_by_name() {
        // B lists the genes as g2, g0, g1.
        let a = base();
        let mut b = base();
        b.var.index = vec!["g2".into(), "g0".into(), "g1".into()];
        b.x = a.x.iter().map(|r| vec![r[2], r[0], r[1]]).collect();
        let r = run(&a, &b, &DiffOptions::default());
        assert_eq!(slot(&r, "X").status, Status::Same, "{:#?}", r.slots);
        assert!(r.equal());
    }

    #[test]
    fn uns_ignores_provenance_and_reports_paths() {
        let mut b = base();
        b.uns = serde_json::json!({
            "title": "t",
            "params": {"k": 16, "res": 0.5000000001},
            "scx_provenance": {"tool": "scx"},
        });
        let r = run(&base(), &b, &DiffOptions::default());
        assert_eq!(slot(&r, "uns").details, ["params/k: 15 vs 16"]);
    }

    #[test]
    fn empty_and_absent_slots_compare_equal() {
        let mut b = base();
        b.obsm.map.clear();
        b.obsm.map.insert(
            "X_umap".into(),
            DenseMatrix {
                shape: (3, 0),
                data: vec![],
            },
        );
        b.uns = serde_json::Value::Null;
        let mut a = base();
        a.obsm.map.clear();
        a.uns = serde_json::json!({});
        let r = run(&a, &b, &DiffOptions::default());
        assert!(r.equal(), "{:#?}", r.slots);
    }

    #[test]
    fn counts_and_data_pair_across_formats() {
        let meta = |name: &str| SparseMatrixMeta {
            name: name.into(),
            shape: (1, 1),
            indptr: vec![0, 1],
        };
        // h5ad: X normalised, counts in raw. h5seurat: X = counts, layers/data.
        let a = vec![Slot::X, Slot::Raw(meta("raw"))];
        let b = vec![Slot::X, Slot::Layer(meta("data"))];
        let pairs = pair(&a, &b, &[]);
        assert!(pairs.contains(&(Some(0), Some(1))) && pairs.contains(&(Some(1), Some(0))));
        // The same with a counts layer, and the other way round.
        let a = vec![Slot::X, Slot::Layer(meta("counts"))];
        let pairs = pair(&b, &a, &[]);
        assert!(pairs.contains(&(Some(1), Some(0))) && pairs.contains(&(Some(0), Some(1))));
        // --map wins over the rule.
        let pairs = pair(&a, &b, &[("X".into(), "X".into())]);
        assert!(pairs.contains(&(Some(0), Some(0))));
    }

    /// Large uns arrays are compared typed: equal to the same table read
    /// back from JSON (an npy snapshot), and a changed value is reported.
    #[test]
    fn uns_arrays_compare_typed_and_json() {
        use crate::ir::{UnsArray, UnsData, UNS_TYPED_MIN};
        let img: Vec<u8> = (0..UNS_TYPED_MIN).map(|i| i as u8).collect();
        let typed = UnsTable {
            raw: serde_json::json!({"spatial": {"k": 1}}),
            arrays: [(
                "spatial/img".to_string(),
                UnsArray {
                    shape: vec![UNS_TYPED_MIN / 4, 4],
                    data: UnsData::U8(img),
                },
            )]
            .into(),
        };
        let mut json = UnsTable {
            raw: typed.to_json(),
            ..Default::default()
        };
        json.split_large();
        let o = DiffOptions::default();
        let d = compare_uns(&typed, &json, &o);
        assert_eq!(d.status, Status::Same, "{:?}", d.details);

        let mut other = typed.clone();
        if let UnsData::U8(v) = &mut other.arrays.get_mut("spatial/img").unwrap().data {
            v[5] += 1;
        }
        let d = compare_uns(&typed, &other, &o);
        assert_eq!(d.status, Status::Diff);
        assert_eq!(
            d.details,
            ["spatial/img: 1 values differ, first at 5: 5 vs 6"]
        );

        let mut other = typed.clone();
        let img = other.arrays.remove("spatial/img").unwrap();
        other.arrays.insert(
            "spatial/img".into(),
            UnsArray {
                shape: vec![UNS_TYPED_MIN],
                ..img.clone()
            },
        );
        other.arrays.insert("extra".into(), img);
        let d = compare_uns(&typed, &other, &o);
        assert_eq!(
            d.details,
            [
                "extra: only in B".to_string(),
                format!("spatial/img: shape [16384, 4] vs [{UNS_TYPED_MIN}]"),
            ]
        );
        let d = compare_uns(&other, &typed, &o);
        assert_eq!(d.details[0], "extra: only in A");
    }
}
