//! A Seurat object saved with `saveRDS` (or `save`): counts, names, meta.data.
//!
//! Supports the v3/v4 `Assay` class (`@counts`, `@data` dgCMatrix slots) and
//! the v5 `Assay5` class (`@layers`, with cell and feature names in the
//! `@cells` / `@features` LogMaps), including layers split by group
//! (`counts.g1`, `counts.g2`, ...). Everything else on the object
//! (reductions, graphs, scale.data, commands) is ignored.
//!
//! The matrices come before the names and meta.data in the stream, so the
//! reader decompresses the file up to three times: a skeleton pass at open
//! that skips large vectors, a pass fetching the metadata vectors it needs,
//! and one pass per `x_stream`.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use super::xdr::{be_f64s, be_i32s, Kind, Node, Num, Parser, Xdr, NA_INT};
use super::{open_stream, Spool};
use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{Column, ColumnData, MatrixChunk, ObsTable, SparseMatrixCSR, VarTable};
use crate::stream::{ChunkStream, DatasetReader};

/// One dgCMatrix layer, genes x cells: its columns are rows of X.
struct Layer {
    i: Node,
    x: Node,
    /// Column pointer (cells + 1 entries).
    p: Vec<u64>,
    /// First X row this layer fills.
    row_offset: usize,
}

pub struct RdsSeuratReader {
    path: PathBuf,
    chunk_size: usize,
    n_vars: usize,
    layers: Vec<Layer>,
    indptr: Vec<u64>,
    obs: ObsTable,
    var: VarTable,
    /// The downloaded copy this reader owns, for a remote file.
    _spool: Option<Spool>,
}

/// Fetched lazy vectors, keyed by stream offset.
type Fetched = HashMap<u64, Vec<u8>>;

/// LogMap columns: layer name -> which rows belong to it.
type Membership = HashMap<String, Vec<bool>>;

impl RdsSeuratReader {
    /// Open the Seurat object in `path`, reading `layer` (default "counts")
    /// of `assay` (default: the object's active assay).
    pub fn open(
        path: &Path,
        chunk_size: usize,
        assay: Option<&str>,
        layer: Option<&str>,
    ) -> Result<Self> {
        let mut xdr = open_stream(path)?;
        xdr.header()?;
        let root = Parser::new(&mut xdr).item()?;
        drop(xdr);

        let obj = find_seurat(&root)?;
        let assay_name = match assay {
            Some(a) => a.to_string(),
            None => str1(obj.attr("active.assay")).unwrap_or_else(|| "RNA".into()),
        };
        let assays = obj
            .attr("assays")
            .ok_or_else(|| bad("Seurat object has no @assays".into()))?;
        let a = assays.get(&assay_name).ok_or_else(|| {
            bad(format!(
                "no assay '{assay_name}' (have: {})",
                assays.names().join(", ")
            ))
        })?;
        let layer = layer.unwrap_or("counts");
        let meta = obj.attr("meta.data");

        // Everything below `p` and the LogMaps / meta.data columns can be lazy;
        // fetch the ones we need in one sweep.
        let mut want: Vec<&Node> = Vec::new();
        let picked = pick_layers(a, layer)?;
        for (_, m) in &picked {
            want.push(slot(m, "p")?);
        }
        if a.has_class("Assay5") {
            want.push(slot(a, "cells")?);
            want.push(slot(a, "features")?);
        }
        if let Some(meta) = meta {
            want.extend(meta.items().iter().map(|(_, c)| c));
        }
        let fetched = fetch(path, &want)?;

        // Cell and gene names, and which cells each layer holds.
        let (cells, genes, layer_rows) = if a.has_class("Assay5") {
            let (cells, member) = logmap(slot(a, "cells")?, &fetched)?;
            let (genes, fmember) = logmap(slot(a, "features")?, &fetched)?;
            let mut rows = Vec::new();
            for (name, _) in &picked {
                let col = member
                    .get(name)
                    .ok_or_else(|| bad(format!("layer '{name}' missing from the @cells LogMap")))?;
                if !fmember.get(name).is_some_and(|f| f.iter().all(|&b| b)) {
                    return Err(bad(format!(
                        "layer '{name}' covers only some features; unsupported"
                    )));
                }
                rows.push((0..cells.len()).filter(|&k| col[k]).collect::<Vec<_>>());
            }
            (cells, genes, rows)
        } else {
            let (_, m) = &picked[0];
            let dn = slot(m, "Dimnames")?.items();
            let names = |k: usize| match dn.get(k).map(|(_, n)| &n.kind) {
                Some(Kind::Str(v)) => Ok(v.iter().map(|s| s.clone().unwrap_or_default()).collect()),
                _ => Err(bad(format!("'{layer}' matrix has no dimnames"))),
            };
            let cells: Vec<String> = names(1)?;
            let rows = vec![(0..cells.len()).collect()];
            (cells, names(0)?, rows)
        };

        // The layers must partition the object's cells; X rows follow layer order.
        let order: Vec<usize> = layer_rows.iter().flatten().copied().collect();
        let mut seen = vec![false; cells.len()];
        for &k in &order {
            if std::mem::replace(&mut seen[k], true) {
                return Err(bad(format!(
                    "cell '{}' is in two '{layer}' layers",
                    cells[k]
                )));
            }
        }
        if order.len() != cells.len() {
            return Err(bad(format!(
                "'{layer}' layers hold {} of {} cells",
                order.len(),
                cells.len()
            )));
        }

        let n_vars = genes.len();
        let mut layers = Vec::new();
        let mut indptr = vec![0u64];
        for ((name, m), rows) in picked.iter().zip(&layer_rows) {
            let dim = ints(slot(m, "Dim")?, &fetched)?;
            if dim.len() != 2 || dim[0] as usize != n_vars || dim[1] as usize != rows.len() {
                return Err(bad(format!(
                    "layer '{name}' is {dim:?}, expected [{n_vars}, {}]",
                    rows.len()
                )));
            }
            let p: Vec<u64> = ints(slot(m, "p")?, &fetched)?
                .into_iter()
                .map(|v| v as u64)
                .collect();
            if p.len() != rows.len() + 1 {
                return Err(bad(format!("layer '{name}': @p has {} entries", p.len())));
            }
            let base = *indptr.last().unwrap();
            indptr.extend(p[1..].iter().map(|v| base + v));
            layers.push(Layer {
                i: slot(m, "i")?.clone(),
                x: slot(m, "x")?.clone(),
                p,
                row_offset: indptr.len() - 1 - rows.len(),
            });
        }

        let obs = ObsTable {
            index: order.iter().map(|&k| cells[k].clone()).collect(),
            columns: match meta {
                Some(meta) => meta_columns(meta, &cells, &order, &fetched)?,
                None => Vec::new(),
            },
        };
        Ok(Self {
            path: path.to_path_buf(),
            chunk_size,
            n_vars,
            layers,
            indptr,
            obs,
            var: VarTable {
                index: genes,
                columns: Vec::new(),
            },
            _spool: None,
        })
    }

    /// Open a downloaded copy, which is deleted along with the reader.
    #[cfg(feature = "net")]
    pub(crate) fn open_spooled(
        spool: Spool,
        chunk_size: usize,
        assay: Option<&str>,
        layer: Option<&str>,
    ) -> Result<Self> {
        let mut r = Self::open(&spool.0, chunk_size, assay, layer)?;
        r._spool = Some(spool);
        Ok(r)
    }
}

/// The Seurat object: the root, or (for a `.rda`) the first saved Seurat object.
fn find_seurat(root: &Node) -> Result<&Node> {
    if root.has_class("Seurat") {
        return Ok(root);
    }
    root.items()
        .iter()
        .map(|(_, n)| n)
        .find(|n| n.has_class("Seurat"))
        .ok_or_else(|| {
            let class = root.class().join("/");
            bad(format!(
                "not a Seurat object (top-level class: {})",
                if class.is_empty() { "none" } else { &class }
            ))
        })
}

/// The dgCMatrix layers named `layer`, or split as `layer.<group>`, in stream order.
fn pick_layers<'a>(a: &'a Node, layer: &str) -> Result<Vec<(String, &'a Node)>> {
    let picked: Vec<(String, &Node)> = if a.has_class("Assay5") {
        let layers = slot(a, "layers")?;
        let prefix = format!("{layer}.");
        layers
            .names()
            .into_iter()
            .zip(layers.items().iter().map(|(_, n)| n))
            .filter(|(n, _)| n == layer || n.starts_with(&prefix))
            .collect()
    } else if a.has_class("Assay") {
        match a.attr(layer) {
            Some(m) if !m.is_null() => vec![(layer.to_string(), m)],
            _ => Vec::new(),
        }
    } else {
        return Err(bad(format!(
            "assay class {:?} is not supported (Assay or Assay5 only)",
            a.class()
        )));
    };
    if picked.is_empty() {
        return Err(bad(format!("assay has no '{layer}' layer")));
    }
    for (name, m) in &picked {
        if !m.has_class("dgCMatrix") {
            return Err(bad(format!(
                "layer '{name}' is {:?}; only dgCMatrix layers are supported \
                 (on-disk BPCells layers live outside the .rds)",
                m.class()
            )));
        }
    }
    Ok(picked)
}

/// A LogMap (logical matrix, rows = names, one column per layer):
/// the row names and each column's membership.
fn logmap(m: &Node, f: &Fetched) -> Result<(Vec<String>, Membership)> {
    let dn = m
        .attr("dimnames")
        .ok_or_else(|| bad("LogMap without dimnames".into()))?
        .items();
    let strs = |k: usize| -> Vec<String> {
        match dn.get(k).map(|(_, n)| &n.kind) {
            Some(Kind::Str(v)) => v.iter().map(|s| s.clone().unwrap_or_default()).collect(),
            _ => Vec::new(),
        }
    };
    let (rows, cols) = (strs(0), strs(1));
    let v = ints(m, f)?;
    if v.len() != rows.len() * cols.len() {
        return Err(bad("LogMap size does not match its dimnames".into()));
    }
    let member = cols
        .into_iter()
        .enumerate()
        .map(|(j, c)| {
            (
                c,
                v[j * rows.len()..(j + 1) * rows.len()]
                    .iter()
                    .map(|&b| b == 1)
                    .collect(),
            )
        })
        .collect();
    Ok((rows, member))
}

/// meta.data columns, reordered to X's row order. Columns of a type with no
/// equivalent (lists, complex, ...) are skipped with a warning.
fn meta_columns(
    meta: &Node,
    cells: &[String],
    order: &[usize],
    f: &Fetched,
) -> Result<Vec<Column>> {
    if let Some(Kind::Str(rn)) = meta.attr("row.names").map(|n| &n.kind) {
        if rn.len() != cells.len() || rn.iter().zip(cells).any(|(a, b)| a.as_deref() != Some(b)) {
            return Err(bad("meta.data rows are not the object's cells".into()));
        }
    }
    let mut out = Vec::new();
    for (name, (_, c)) in meta.names().into_iter().zip(meta.items()) {
        if c.len() != cells.len() {
            tracing::warn!(column = %name, "rds: meta.data column has the wrong length; skipped");
            continue;
        }
        let pick = |v: &[i32]| -> Vec<i32> { order.iter().map(|&k| v[k]).collect() };
        let col = match &c.kind {
            _ if c.has_class("factor") => {
                let levels = match c.attr("levels").map(|l| &l.kind) {
                    Some(Kind::Str(l)) => l.iter().map(|s| s.clone().unwrap_or_default()).collect(),
                    _ => Vec::new(),
                };
                let codes = pick(&ints(c, f)?);
                let missing: Vec<bool> = codes.iter().map(|&v| v == NA_INT).collect();
                let codes = codes
                    .iter()
                    .map(|&v| if v == NA_INT { 0 } else { (v - 1) as u32 })
                    .collect();
                let ordered = c.has_class("ordered");
                Column::with_missing(
                    name,
                    ColumnData::Categorical {
                        codes,
                        levels,
                        ordered,
                    },
                    missing,
                )
            }
            Kind::Int(_) | Kind::Lazy { num: Num::Int, .. } => {
                let v = pick(&ints(c, f)?);
                let missing: Vec<bool> = v.iter().map(|&x| x == NA_INT).collect();
                let v = v
                    .into_iter()
                    .map(|x| if x == NA_INT { 0 } else { x })
                    .collect();
                Column::with_missing(name, ColumnData::Int(v), missing)
            }
            Kind::Lgl(_) | Kind::Lazy { num: Num::Lgl, .. } => {
                let v = pick(&ints(c, f)?);
                let missing: Vec<bool> = v.iter().map(|&x| x == NA_INT).collect();
                Column::with_missing(
                    name,
                    ColumnData::Bool(v.iter().map(|&x| x == 1).collect()),
                    missing,
                )
            }
            Kind::Real(_) | Kind::Lazy { num: Num::Real, .. } => {
                let all = reals(c, f)?;
                let v: Vec<f64> = order.iter().map(|&k| all[k]).collect();
                let missing: Vec<bool> = v.iter().map(|&x| is_na_real(x)).collect();
                Column::with_missing(name, ColumnData::Float(v), missing)
            }
            Kind::Str(s) => {
                let v: Vec<Option<String>> = order.iter().map(|&k| s[k].clone()).collect();
                let missing: Vec<bool> = v.iter().map(Option::is_none).collect();
                let v = v.into_iter().map(Option::unwrap_or_default).collect();
                Column::with_missing(name, ColumnData::String(v), missing)
            }
            _ => {
                tracing::warn!(column = %name, "rds: unsupported meta.data column type; skipped");
                continue;
            }
        };
        out.push(col);
    }
    Ok(out)
}

/// R's NA_real_ is a NaN with low word 1954; other NaNs are values.
fn is_na_real(x: f64) -> bool {
    x.is_nan() && (x.to_bits() & 0xFFFF_FFFF) == 1954
}

/// Read the lazy vectors among `nodes` in one forward pass.
fn fetch(path: &Path, nodes: &[&Node]) -> Result<Fetched> {
    let mut lazy: Vec<(u64, u64)> = nodes
        .iter()
        .filter_map(|n| match n.kind {
            Kind::Lazy { num, offset, len } => Some((offset, len * num.width())),
            _ => None,
        })
        .collect();
    lazy.sort_unstable();
    lazy.dedup();
    let mut out = HashMap::new();
    if lazy.is_empty() {
        return Ok(out);
    }
    let mut xdr = open_stream(path)?;
    for (offset, bytes) in lazy {
        xdr.seek_to(offset)?;
        out.insert(offset, xdr.bytes(bytes as usize)?);
    }
    Ok(out)
}

fn ints(n: &Node, f: &Fetched) -> Result<Vec<i32>> {
    match &n.kind {
        Kind::Int(v) | Kind::Lgl(v) => Ok(v.clone()),
        Kind::Lazy {
            num: Num::Int | Num::Lgl,
            offset,
            ..
        } => {
            let raw = f
                .get(offset)
                .ok_or_else(|| bad("vector was not fetched".into()))?;
            Ok(be_i32s(raw))
        }
        _ => Err(bad("expected an integer vector".into())),
    }
}

fn reals(n: &Node, f: &Fetched) -> Result<Vec<f64>> {
    match &n.kind {
        Kind::Real(v) => Ok(v.clone()),
        Kind::Lazy {
            num: Num::Real,
            offset,
            ..
        } => {
            let raw = f
                .get(offset)
                .ok_or_else(|| bad("vector was not fetched".into()))?;
            Ok(be_f64s(raw))
        }
        _ => Err(bad("expected a double vector".into())),
    }
}

fn slot<'a>(n: &'a Node, name: &str) -> Result<&'a Node> {
    n.attr(name)
        .ok_or_else(|| bad(format!("missing slot @{name} on {:?}", n.class())))
}

fn str1(n: Option<&Node>) -> Option<String> {
    match &n?.kind {
        Kind::Str(v) => v.first().cloned().flatten(),
        _ => None,
    }
}

fn bad(msg: String) -> ScxError {
    ScxError::InvalidFormat(format!("rds: {msg}"))
}

/// Forward-only X scan: per layer, `i` is buffered whole (it precedes `x` in
/// the stream), then `x` is read chunk by chunk.
// ponytail: `i` costs 4 bytes per nonzero of the largest layer; spill it to a
// temp file if that ever outgrows RAM at atlas scale.
struct Scan {
    path: PathBuf,
    xdr: Option<Xdr<Box<dyn Read + Send>>>,
    layer: usize,
    row: usize,
    started: bool,
    i: Vec<u32>,
    x_small: Option<Vec<f64>>,
}

impl Scan {
    fn xdr(&mut self) -> Result<&mut Xdr<Box<dyn Read + Send>>> {
        if self.xdr.is_none() {
            self.xdr = Some(open_stream(&self.path)?);
        }
        Ok(self.xdr.as_mut().unwrap())
    }

    fn start_layer(&mut self, l: &Layer) -> Result<()> {
        let nnz = *l.p.last().unwrap() as usize;
        let i = match &l.i.kind {
            Kind::Int(v) => v.clone(),
            Kind::Lazy {
                num: Num::Int,
                offset,
                len,
            } => {
                let xdr = self.xdr()?;
                xdr.seek_to(*offset)?;
                xdr.ints(*len as usize)?
            }
            _ => return Err(bad("@i is not an integer vector".into())),
        };
        if i.len() != nnz || l.x.len() != nnz {
            return Err(bad("@i / @x lengths do not match @p".into()));
        }
        self.i = i.into_iter().map(|v| v as u32).collect();
        self.x_small = match &l.x.kind {
            Kind::Real(v) => Some(v.clone()),
            Kind::Lazy {
                num: Num::Real,
                offset,
                ..
            } => {
                self.xdr()?.seek_to(*offset)?;
                None
            }
            _ => return Err(bad("@x is not a double vector".into())),
        };
        Ok(())
    }

    fn next_chunk(
        &mut self,
        layers: &[Layer],
        n_vars: usize,
        chunk: usize,
    ) -> Option<Result<MatrixChunk>> {
        loop {
            let l = layers.get(self.layer)?;
            let n_rows = l.p.len() - 1;
            if self.row >= n_rows {
                self.layer += 1;
                self.row = 0;
                self.started = false;
                continue;
            }
            if !self.started {
                self.started = true;
                if let Err(e) = self.start_layer(l) {
                    return Some(Err(e));
                }
            }
            let (a, b) = (self.row, (self.row + chunk.max(1)).min(n_rows));
            let (lo, hi) = (l.p[a] as usize, l.p[b] as usize);
            let data = match &self.x_small {
                Some(x) => Ok(x[lo..hi].to_vec()),
                None => self.xdr().and_then(|x| x.reals(hi - lo)),
            };
            let data = match data {
                Ok(d) => d,
                Err(e) => return Some(Err(e)),
            };
            self.row = b;
            return Some(Ok(MatrixChunk {
                row_offset: l.row_offset + a,
                nrows: b - a,
                data: SparseMatrixCSR {
                    shape: (b - a, n_vars),
                    indptr: l.p[a..=b].iter().map(|v| v - l.p[a]).collect(),
                    indices: self.i[lo..hi].to_vec(),
                    data: TypedVec::F64(data),
                },
            }));
        }
    }
}

#[async_trait]
impl DatasetReader for RdsSeuratReader {
    fn shape(&self) -> (usize, usize) {
        (self.indptr.len() - 1, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        DataType::F64 // dgCMatrix @x is always double
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        Ok(self.obs.clone())
    }

    async fn var(&mut self) -> Result<VarTable> {
        Ok(self.var.clone())
    }

    fn x_indptr(&self) -> &[u64] {
        &self.indptr
    }

    fn x_stream(&mut self) -> ChunkStream<'_> {
        let mut scan = Scan {
            path: self.path.clone(),
            xdr: None,
            layer: 0,
            row: 0,
            started: false,
            i: Vec::new(),
            x_small: None,
        };
        let (layers, n_vars, chunk) = (&self.layers, self.n_vars, self.chunk_size);
        let mut failed = false;
        Box::pin(stream::iter(std::iter::from_fn(move || {
            if failed {
                return None;
            }
            let next = scan.next_chunk(layers, n_vars, chunk);
            failed = matches!(next, Some(Err(_)));
            next
        })))
    }
}
