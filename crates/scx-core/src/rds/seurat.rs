//! A Seurat object saved with `saveRDS` (or `save`): counts, names, meta.data.
//!
//! Supports the v3/v4 `Assay` class (`@counts`, `@data` dgCMatrix slots) and
//! the v5 `Assay5` class (`@layers`, with cell and feature names in the
//! `@cells` / `@features` LogMaps), including layers split by group
//! (`counts.g1`, `counts.g2`, ...). The assay's other dgCMatrix layers
//! (`data`, ...) are layers; dense ones (`scale.data`) are skipped with a
//! warning. The assay's feature metadata (`@meta.features` on `Assay`,
//! `@meta.data` on `Assay5`) is var. `@reductions` become obsm (`X_<name>`,
//! cell embeddings) and varm (feature loadings of reductions on the read
//! assay), `@graphs` become obsp. Everything else on the object (neighbors,
//! scale.data, images, commands) is ignored.
//!
//! The matrices come before the names and meta.data in the stream, so the
//! reader decompresses the file up to three times: a skeleton pass at open
//! that skips large vectors, a pass fetching the metadata vectors it needs,
//! and one pass per `x_stream`. Reductions and graphs are fetched in their
//! own pass when `obsm` / `varm` / `obsp_metas` ask for them.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use super::open_stream;
use super::xdr::{be_f64s, be_i32s, Kind, Node, Num, Parser, Xdr, NA_INT};
use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{
    Column, ColumnData, DenseMatrix, Embeddings, MatrixChunk, ObsTable, SparseMatrixCSR,
    SparseMatrixMeta, VarTable, Varm,
};
use crate::stream::{row_chunks, ChunkStream, DatasetReader};

/// One dgCMatrix layer, genes x cells: its columns are rows of X.
struct Layer {
    i: Node,
    x: Node,
    /// Column pointer (cells + 1 entries).
    p: Vec<u64>,
    /// First X row this layer fills.
    row_offset: usize,
}

/// One matrix of the assay (X, or another layer): its dgCMatrix parts (one,
/// or one per split group) in X row order, and the CSR row pointer over them.
struct Matrix {
    parts: Vec<Layer>,
    indptr: Vec<u64>,
}

pub struct RdsSeuratReader {
    path: PathBuf,
    chunk_size: usize,
    n_vars: usize,
    x: Matrix,
    /// The assay's other dgCMatrix layers, by base name (`data`, ...).
    layers: Vec<(String, Matrix)>,
    obs: ObsTable,
    var: VarTable,
    /// The read assay's name: varm keeps only reductions computed on it.
    assay: String,
    /// The object's cells in object order, and each one's X row.
    cells: Vec<String>,
    row_of: HashMap<String, usize>,
    /// `@reductions` (DimReduc) and `@graphs` (Graph) entries, still lazy.
    reductions: Vec<(String, Node)>,
    graphs: Vec<(String, Node)>,
    /// Graphs as CSR in X row order, built by `obsp_metas`.
    obsp: HashMap<String, SparseMatrixCSR>,
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
        // The assay's other layers; one that is not dgCMatrix is skipped.
        let others: Vec<(String, Vec<(String, &Node)>)> = other_layers(a, layer)
            .into_iter()
            .filter_map(|base| match pick_layers(a, &base) {
                Ok(p) => Some((base, p)),
                Err(e) => {
                    tracing::warn!("rds: layer '{base}' is not read: {e}");
                    None
                }
            })
            .collect();
        let var_meta = if a.has_class("Assay5") {
            a.attr("meta.data")
        } else {
            a.attr("meta.features")
        }
        .filter(|m| !m.is_null() && !m.items().is_empty());

        // Everything below `p` and the LogMaps / meta.data columns can be lazy;
        // fetch the ones we need in one sweep.
        let mut want: Vec<&Node> = Vec::new();
        let picked = pick_layers(a, layer)?;
        for (_, m) in picked.iter().chain(others.iter().flat_map(|(_, p)| p)) {
            want.push(slot(m, "p")?);
        }
        if a.has_class("Assay5") {
            want.push(slot(a, "cells")?);
            want.push(slot(a, "features")?);
        }
        for m in meta.iter().chain(var_meta.iter()) {
            want.extend(m.items().iter().map(|(_, c)| c));
        }
        let fetched = fetch(path, &want)?;

        // Cell and gene names, and which cells each layer holds.
        let (cells, genes, members) = if a.has_class("Assay5") {
            let (cells, member) = logmap(slot(a, "cells")?, &fetched)?;
            let (genes, fmember) = logmap(slot(a, "features")?, &fetched)?;
            (cells, genes, Some((member, fmember)))
        } else {
            let (_, m) = &picked[0];
            let dn = slot(m, "Dimnames")?.items();
            let names = |k: usize| match dn.get(k).map(|(_, n)| &n.kind) {
                Some(Kind::Str(v)) => Ok(v.iter().map(|s| s.clone().unwrap_or_default()).collect()),
                _ => Err(bad(format!("'{layer}' matrix has no dimnames"))),
            };
            (names(1)?, names(0)?, None)
        };
        // The object cells each part of a layer holds, in part order.
        let part_rows = |picked: &[(String, &Node)]| -> Result<Vec<Vec<usize>>> {
            let Some((member, fmember)) = &members else {
                return Ok(vec![(0..cells.len()).collect()]);
            };
            picked
                .iter()
                .map(|(name, _)| {
                    let col = member.get(name).ok_or_else(|| {
                        bad(format!("layer '{name}' missing from the @cells LogMap"))
                    })?;
                    if !fmember.get(name).is_some_and(|f| f.iter().all(|&b| b)) {
                        return Err(bad(format!(
                            "layer '{name}' covers only some features; unsupported"
                        )));
                    }
                    Ok((0..cells.len()).filter(|&k| col[k]).collect())
                })
                .collect()
        };
        let layer_rows = part_rows(&picked)?;

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
        let x = build_matrix(&picked, &layer_rows, n_vars, &fetched)?;
        // Another layer streams in X's row order, so its parts must hold
        // X's cells in X's order: Seurat splits every layer of an assay alike.
        let mut layers = Vec::new();
        for (base, p) in &others {
            let got = part_rows(p).and_then(|rows| {
                if rows.iter().flatten().ne(order.iter()) {
                    return Err(bad("its cells are not split as X's are".into()));
                }
                build_matrix(p, &rows, n_vars, &fetched)
            });
            match got {
                Ok(m) => layers.push((base.clone(), m)),
                Err(e) => tracing::warn!("rds: layer '{base}' is not read: {e}"),
            }
        }
        let var_columns = match var_meta {
            Some(m) => {
                let all: Vec<usize> = (0..n_vars).collect();
                meta_columns(m, &genes, &all, &fetched).unwrap_or_else(|e| {
                    tracing::warn!("rds: feature metadata is not read: {e}");
                    Vec::new()
                })
            }
            None => Vec::new(),
        };

        let row_of = order
            .iter()
            .enumerate()
            .map(|(row, &k)| (cells[k].clone(), row))
            .collect();
        let named = |slot_name: &str| -> Vec<(String, Node)> {
            obj.attr(slot_name).map_or_else(Vec::new, |l| {
                l.names()
                    .into_iter()
                    .zip(l.items().iter().map(|(_, n)| n.clone()))
                    .collect()
            })
        };
        let (reductions, graphs) = (named("reductions"), named("graphs"));
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
            x,
            layers,
            obs,
            var: VarTable {
                index: genes,
                columns: var_columns,
            },
            assay: assay_name,
            cells,
            row_of,
            reductions,
            graphs,
            obsp: HashMap::new(),
        })
    }

    /// The X row of each of `n` cells named by `names` (a matrix's row or
    /// column names), or in object order when the matrix has no names.
    fn rows_for(&self, names: Option<Vec<String>>, n: usize) -> Result<Vec<usize>> {
        let names = names.unwrap_or_else(|| self.cells.clone());
        if names.len() != n || n != self.cells.len() {
            return Err(bad(format!(
                "{n} rows, but the object has {} cells",
                self.cells.len()
            )));
        }
        names
            .iter()
            .map(|c| {
                self.row_of
                    .get(c)
                    .copied()
                    .ok_or_else(|| bad(format!("cell '{c}' is not in the object")))
            })
            .collect()
    }

    /// Every reduction's cell embeddings, (n_obs, k) in X row order.
    fn read_obsm(&self) -> Result<Embeddings> {
        let embs: Vec<(&str, &Node)> = self
            .reductions
            .iter()
            .filter_map(|(name, r)| Some((name.as_str(), r.attr("cell.embeddings")?)))
            .collect();
        let fetched = fetch(
            &self.path,
            &embs.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
        )?;
        let mut map = HashMap::new();
        for (name, n) in embs {
            let got = dense(n, &fetched).and_then(
                |RMatrix {
                     nrow,
                     ncol,
                     values: v,
                     rownames,
                 }| {
                    let rows = self.rows_for(rownames, nrow)?;
                    let mut data = vec![0.0; nrow * ncol];
                    for (i, &row) in rows.iter().enumerate() {
                        for c in 0..ncol {
                            data[row * ncol + c] = v[c * nrow + i];
                        }
                    }
                    Ok(DenseMatrix {
                        shape: (nrow, ncol),
                        data,
                    })
                },
            );
            match got {
                Ok(m) => {
                    map.insert(format!("X_{}", name.to_lowercase()), m);
                }
                Err(e) => tracing::warn!("rds: skipping reduction '{name}': {e}"),
            }
        }
        Ok(Embeddings { map })
    }

    /// Feature loadings of the reductions computed on the read assay, on
    /// the full var axis by name: NaN where a gene has no loading, as
    /// Seurat's `Loadings()` covers only the features the reduction used.
    fn read_varm(&self) -> Result<Varm> {
        let loads: Vec<(&str, &Node)> = self
            .reductions
            .iter()
            .filter(|(name, r)| {
                let used = str1(r.attr("assay.used"));
                let ok = used.as_deref().is_none_or(|a| a == self.assay);
                if !ok {
                    tracing::info!("rds: varm '{name}' is on assay {used:?}; skipped");
                }
                ok
            })
            .filter_map(|(name, r)| Some((name.as_str(), r.attr("feature.loadings")?)))
            .filter(|(_, n)| n.len() > 0)
            .collect();
        let fetched = fetch(
            &self.path,
            &loads.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
        )?;
        let var_pos: HashMap<&str, usize> = self
            .var
            .index
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_str(), i))
            .collect();
        let n_vars = self.var.index.len();
        let mut map = HashMap::new();
        for (name, n) in loads {
            let got = dense(n, &fetched).and_then(|RMatrix { nrow, ncol, values: v, rownames }| {
                let genes = match rownames {
                    Some(g) => g,
                    None if nrow == n_vars => self.var.index.clone(),
                    None => return Err(bad(format!("{nrow} unnamed loading rows"))),
                };
                let mut data = vec![f64::NAN; n_vars * ncol];
                let mut dropped = 0;
                for (i, g) in genes.iter().enumerate() {
                    let Some(&row) = var_pos.get(g.as_str()) else {
                        dropped += 1;
                        continue;
                    };
                    for c in 0..ncol {
                        data[row * ncol + c] = v[c * nrow + i];
                    }
                }
                if dropped > 0 {
                    tracing::warn!(
                        "rds: varm '{name}': {dropped} of {nrow} loading features are not in the assay"
                    );
                }
                Ok(DenseMatrix {
                    shape: (n_vars, ncol),
                    data,
                })
            });
            match got {
                Ok(m) => {
                    map.insert(format!("X_{}", name.to_lowercase()), m);
                }
                Err(e) => tracing::warn!("rds: skipping loadings of '{name}': {e}"),
            }
        }
        Ok(Varm { map })
    }

    /// Every graph as CSR in X row order. A Graph is a dgCMatrix, so CSC:
    /// column c holds the edges into cell c. Transposing gives rows; both
    /// axes are then renumbered to X rows and each row's columns sorted.
    fn read_graphs(&self) -> Result<HashMap<String, SparseMatrixCSR>> {
        let mut want = Vec::new();
        for (_, g) in &self.graphs {
            for s in ["i", "p", "x"] {
                if let Some(n) = g.attr(s) {
                    want.push(n);
                }
            }
        }
        let fetched = fetch(&self.path, &want)?;
        let mut out = HashMap::new();
        for (name, g) in &self.graphs {
            match self.graph_csr(g, &fetched) {
                Ok(m) => {
                    out.insert(name.clone(), m);
                }
                Err(e) => tracing::warn!("rds: skipping graph '{name}': {e}"),
            }
        }
        Ok(out)
    }

    fn graph_csr(&self, g: &Node, f: &Fetched) -> Result<SparseMatrixCSR> {
        let dim = ints(slot(g, "Dim")?, f)?;
        let n = self.cells.len();
        if dim != [n as i32, n as i32] {
            return Err(bad(format!("graph is {dim:?}, expected [{n}, {n}]")));
        }
        let dn = slot(g, "Dimnames")?.items();
        let names = |k: usize| match dn.get(k).map(|(_, v)| &v.kind) {
            Some(Kind::Str(v)) => Some(v.iter().map(|s| s.clone().unwrap_or_default()).collect()),
            _ => None,
        };
        let src_row = self.rows_for(names(0), n)?;
        let src_col = self.rows_for(names(1), n)?;
        let (i, p, x) = (
            ints(slot(g, "i")?, f)?,
            ints(slot(g, "p")?, f)?,
            reals(slot(g, "x")?, f)?,
        );
        if p.len() != n + 1 || i.len() != x.len() || *p.last().unwrap() as usize != i.len() {
            return Err(bad("graph @i / @p / @x do not fit together".into()));
        }
        let mut counts = vec![0u64; n + 1];
        for &r in &i {
            counts[src_row[r as usize] + 1] += 1;
        }
        for k in 0..n {
            counts[k + 1] += counts[k];
        }
        let indptr = counts.clone();
        let mut next = counts;
        let mut entries = vec![(0u32, 0.0f64); i.len()];
        for c in 0..n {
            for t in p[c] as usize..p[c + 1] as usize {
                let row = src_row[i[t] as usize];
                entries[next[row] as usize] = (src_col[c] as u32, x[t]);
                next[row] += 1;
            }
        }
        for r in 0..n {
            entries[indptr[r] as usize..indptr[r + 1] as usize].sort_unstable_by_key(|e| e.0);
        }
        let (indices, data) = entries.into_iter().unzip();
        Ok(SparseMatrixCSR {
            shape: (n, n),
            indptr,
            indices,
            data: TypedVec::F64(data),
        })
    }
}

/// An R numeric matrix.
struct RMatrix {
    nrow: usize,
    ncol: usize,
    /// Column-major, as R stores it.
    values: Vec<f64>,
    rownames: Option<Vec<String>>,
}

fn dense(n: &Node, f: &Fetched) -> Result<RMatrix> {
    let dim = match n.attr("dim").map(|d| &d.kind) {
        Some(Kind::Int(d)) if d.len() == 2 => (d[0] as usize, d[1] as usize),
        _ => return Err(bad("matrix without a 2-D dim".into())),
    };
    let v = match &n.kind {
        Kind::Int(_) | Kind::Lazy { num: Num::Int, .. } => {
            ints(n, f)?.into_iter().map(f64::from).collect()
        }
        _ => reals(n, f)?,
    };
    if v.len() != dim.0 * dim.1 {
        return Err(bad(format!("matrix of {} values is not {dim:?}", v.len())));
    }
    let rownames = n.attr("dimnames").and_then(|dn| match dn.items().first() {
        Some((
            _,
            Node {
                kind: Kind::Str(v), ..
            },
        )) => Some(v.iter().map(|s| s.clone().unwrap_or_default()).collect()),
        _ => None,
    });
    Ok(RMatrix {
        nrow: dim.0,
        ncol: dim.1,
        values: v,
        rownames,
    })
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
            bad(format!(
                "not a Seurat object; the file holds {}",
                describe(root)
            ))
        })
}

/// What a top-level object is, for the "not a Seurat object" error: its
/// class, or for a plain list its elements' names and classes.
fn describe(n: &Node) -> String {
    let class = |n: &Node| match n.class().join("/") {
        c if c.is_empty() => kind_name(n).to_string(),
        c => c,
    };
    if !n.class().is_empty() || n.items().is_empty() {
        return format!("a {}", class(n));
    }
    let names = n.names();
    let shown: Vec<String> = n
        .items()
        .iter()
        .enumerate()
        .take(8)
        .map(|(i, (tag, v))| {
            let name = tag
                .clone()
                .or_else(|| names.get(i).cloned())
                .unwrap_or_default();
            format!("{name}: {}", class(v))
        })
        .collect();
    let more = n.items().len().saturating_sub(shown.len());
    format!(
        "a list of {} ({}{})",
        n.items().len(),
        shown.join(", "),
        if more > 0 {
            format!(", +{more} more")
        } else {
            String::new()
        }
    )
}

fn kind_name(n: &Node) -> &'static str {
    match &n.kind {
        Kind::Null => "NULL",
        Kind::Sym(_) => "symbol",
        Kind::Str(_) => "character vector",
        Kind::Int(_) | Kind::Lazy { num: Num::Int, .. } => "integer vector",
        Kind::Lgl(_) | Kind::Lazy { num: Num::Lgl, .. } => "logical vector",
        Kind::Real(_) | Kind::Lazy { num: Num::Real, .. } => "numeric vector",
        Kind::List(_) => "list",
        Kind::S4 => "S4 object",
        Kind::Other => "R object",
    }
}

/// The dgCMatrix parts `picked` of one layer, holding `rows` (object cells,
/// per part), as one matrix over `n_vars` genes.
fn build_matrix(
    picked: &[(String, &Node)],
    rows: &[Vec<usize>],
    n_vars: usize,
    fetched: &Fetched,
) -> Result<Matrix> {
    let mut parts = Vec::new();
    let mut indptr = vec![0u64];
    for ((name, m), rows) in picked.iter().zip(rows) {
        let dim = ints(slot(m, "Dim")?, fetched)?;
        if dim.len() != 2 || dim[0] as usize != n_vars || dim[1] as usize != rows.len() {
            return Err(bad(format!(
                "layer '{name}' is {dim:?}, expected [{n_vars}, {}]",
                rows.len()
            )));
        }
        let p: Vec<u64> = ints(slot(m, "p")?, fetched)?
            .into_iter()
            .map(|v| v as u64)
            .collect();
        if p.len() != rows.len() + 1 {
            return Err(bad(format!("layer '{name}': @p has {} entries", p.len())));
        }
        let base = *indptr.last().unwrap();
        indptr.extend(p[1..].iter().map(|v| base + v));
        parts.push(Layer {
            i: slot(m, "i")?.clone(),
            x: slot(m, "x")?.clone(),
            p,
            row_offset: indptr.len() - 1 - rows.len(),
        });
    }
    Ok(Matrix { parts, indptr })
}

/// The assay's layers other than `x_layer`, by base name: a split layer
/// (`data.g1`, `data.g2`) is one name. Empty slots of a v3/v4 `Assay` (an
/// unscaled object's 0 x 0 `scale.data`) are not layers.
fn other_layers(a: &Node, x_layer: &str) -> Vec<String> {
    const KNOWN: [&str; 3] = ["counts", "scale.data", "data"];
    let names: Vec<String> = if a.has_class("Assay5") {
        a.attr("layers").map_or_else(Vec::new, |l| l.names())
    } else {
        KNOWN
            .iter()
            // An S4 dgCMatrix has no length of its own; a dense slot has
            // its values' length, 0 when empty.
            .filter(|n| {
                a.attr(n)
                    .is_some_and(|m| !m.is_null() && (m.has_class("dgCMatrix") || m.len() > 0))
            })
            .map(|n| n.to_string())
            .collect()
    };
    let mut out: Vec<String> = Vec::new();
    for n in names {
        let base = KNOWN
            .iter()
            .find(|b| n == **b || n.starts_with(&format!("{b}.")))
            .map_or(n.clone(), |b| b.to_string());
        if base != x_layer && !out.contains(&base) {
            out.push(base);
        }
    }
    out
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
            let what = match m.class() {
                c if c.is_empty() && m.attr("dim").is_some() => "a dense matrix".to_string(),
                c => format!("{c:?}"),
            };
            return Err(bad(format!(
                "layer '{name}' is {what}; only dgCMatrix layers are supported \
                 (on-disk BPCells layers live outside the .rds)"
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
// TODO: `i` costs 4 bytes per nonzero of the largest layer; spill it to a
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
        (self.x.indptr.len() - 1, self.n_vars)
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

    async fn obsm(&mut self) -> Result<Embeddings> {
        self.read_obsm()
    }

    async fn varm(&mut self) -> Result<Varm> {
        self.read_varm()
    }

    async fn obsp_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        if self.obsp.is_empty() && !self.graphs.is_empty() {
            self.obsp = self.read_graphs()?;
        }
        // Stream order, so the output does not depend on hashing.
        Ok(self
            .graphs
            .iter()
            .filter_map(|(name, _)| {
                let m = self.obsp.get(name)?;
                Some(SparseMatrixMeta {
                    name: name.clone(),
                    shape: m.shape,
                    indptr: m.indptr.clone(),
                })
            })
            .collect())
    }

    fn obsp_stream<'a>(&'a self, meta: &'a SparseMatrixMeta, chunk_size: usize) -> ChunkStream<'a> {
        let Some(m) = self.obsp.get(&meta.name) else {
            return crate::stream::error_stream(format!(
                "rds: obsp '{}' was not listed by obsp_metas",
                meta.name
            ));
        };
        row_chunks(m.shape.0, chunk_size, move |r| {
            let (lo, hi) = (m.indptr[r.start] as usize, m.indptr[r.end] as usize);
            let TypedVec::F64(x) = &m.data else {
                unreachable!("graphs are read as f64")
            };
            Ok(MatrixChunk {
                row_offset: r.start,
                nrows: r.len(),
                data: SparseMatrixCSR {
                    shape: (r.len(), m.shape.1),
                    indptr: m.indptr[r.start..=r.end]
                        .iter()
                        .map(|v| v - lo as u64)
                        .collect(),
                    indices: m.indices[lo..hi].to_vec(),
                    data: TypedVec::F64(x[lo..hi].to_vec()),
                },
            })
        })
    }

    fn x_indptr(&self) -> &[u64] {
        &self.x.indptr
    }

    fn x_stream(&mut self) -> ChunkStream<'_> {
        scan_stream(&self.path, &self.x.parts, self.n_vars, self.chunk_size)
    }

    async fn layer_metas(&mut self) -> Result<Vec<SparseMatrixMeta>> {
        let n_obs = self.x.indptr.len() - 1;
        Ok(self
            .layers
            .iter()
            .map(|(name, m)| SparseMatrixMeta {
                name: name.clone(),
                shape: (n_obs, self.n_vars),
                indptr: m.indptr.clone(),
            })
            .collect())
    }

    fn layer_stream<'a>(
        &'a self,
        meta: &'a SparseMatrixMeta,
        chunk_size: usize,
    ) -> ChunkStream<'a> {
        match self.layers.iter().find(|(n, _)| *n == meta.name) {
            Some((_, m)) => scan_stream(&self.path, &m.parts, self.n_vars, chunk_size),
            None => crate::stream::error_stream(format!(
                "rds: layer '{}' was not listed by layer_metas",
                meta.name
            )),
        }
    }
}

/// Row chunks of a matrix's `parts`, read forward through the stream.
fn scan_stream<'a>(
    path: &Path,
    parts: &'a [Layer],
    n_vars: usize,
    chunk: usize,
) -> ChunkStream<'a> {
    let mut scan = Scan {
        path: path.to_path_buf(),
        xdr: None,
        layer: 0,
        row: 0,
        started: false,
        i: Vec::new(),
        x_small: None,
    };
    let mut failed = false;
    Box::pin(stream::iter(std::iter::from_fn(move || {
        if failed {
            return None;
        }
        let next = scan.next_chunk(parts, n_vars, chunk);
        failed = matches!(next, Some(Err(_)));
        next
    })))
}
