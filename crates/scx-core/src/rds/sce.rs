//! A SummarizedExperiment saved with `saveRDS`: SingleCellExperiment and
//! SpatialExperiment (Bioconductor), read into the same [`RdsReader`] as a
//! Seurat object.
//!
//! - `@assays@data@listData`: the `layer` assay (default `counts`) is X, the
//!   other dgCMatrix assays are layers. Assays are genes x cells, so X
//!   streams by cell exactly as a Seurat layer does.
//! - `@colData` is obs; its `@rownames` are the cell names.
//! - rowData (`@rowRanges@elementMetadata`, or `@elementMetadata` without
//!   row ranges) is var; gene names are `@NAMES` or the row ranges' names.
//! - `reducedDims` become obsm `X_<name>`, `spatialCoords` obsm `spatial`.
//!
//! Not read: altExps, row/colPairs, `metadata`, and `imgData` (a Visium
//! object's images are usually only paths to PNGs on the saving machine).

use std::collections::HashMap;

use super::seurat::{bad, build_matrix, fetch, meta_columns, slot, Fetched, Matrix, RdsReader};
use super::xdr::{Kind, Node};
use crate::error::Result;
use crate::ir::{ObsTable, VarTable};

/// The SummarizedExperiment in `root`: the root itself, or (for a `.rda`)
/// the first saved one. Recognised by its slots, not its class, so every
/// subclass (SingleCellExperiment, SpatialExperiment, ...) matches.
pub(super) fn find_se(root: &Node) -> Option<&Node> {
    let is_se = |n: &Node| {
        matches!(n.kind, Kind::S4) && n.attr("assays").is_some() && n.attr("colData").is_some()
    };
    if is_se(root) {
        return Some(root);
    }
    root.items().iter().map(|(_, n)| n).find(|n| is_se(n))
}

/// Strings of a character vector node, or `None` for NULL / anything else.
fn strs(n: Option<&Node>) -> Option<Vec<String>> {
    match &n?.kind {
        Kind::Str(v) => Some(v.iter().map(|s| s.clone().unwrap_or_default()).collect()),
        _ => None,
    }
}

impl RdsReader {
    pub(super) fn from_se(
        src: super::Source,
        se: &Node,
        chunk_size: usize,
        assay: Option<&str>,
        layer: Option<&str>,
    ) -> Result<Self> {
        if let Some(a) = assay {
            return Err(bad(format!(
                "a {} has no Seurat assays (got --assay {a}); pick one of its assays with --layer",
                se.class().join("/")
            )));
        }
        let layer = layer.unwrap_or("counts");
        let assays = slot(se, "assays")?;
        let list = assays
            .attr("data")
            .and_then(|d| d.attr("listData"))
            .ok_or_else(|| {
                bad(format!(
                    "assays are a {:?}, not SimpleAssays; re-save the object after \
                     updateObject() in R",
                    assays.class()
                ))
            })?;
        let names = list.names();
        let all: Vec<(String, &Node)> = names
            .iter()
            .cloned()
            .zip(list.items().iter().map(|(_, n)| n))
            .collect();
        let x_node = all
            .iter()
            .find(|(n, _)| n == layer)
            .map(|(_, m)| *m)
            .ok_or_else(|| bad(format!("no assay '{layer}' (have: {})", names.join(", "))))?;
        if !x_node.has_class("dgCMatrix") {
            return Err(bad(format!(
                "assay '{layer}' is {:?}; only dgCMatrix assays are supported",
                x_node.class()
            )));
        }
        let others: Vec<(String, &Node)> = all
            .iter()
            .filter(|(n, m)| {
                let ok = m.has_class("dgCMatrix");
                if n != layer && !ok {
                    tracing::warn!(
                        "rds: assay '{n}' is {:?}, not dgCMatrix; skipped",
                        m.class()
                    );
                }
                n != layer && ok
            })
            .cloned()
            .collect();

        let col_data = slot(se, "colData")?;
        let obs_cols = col_data.attr("listData");
        // rowData: on the row ranges for a RangedSummarizedExperiment.
        let row_ranges = se.attr("rowRanges").filter(|n| !n.is_null());
        let row_data = row_ranges
            .and_then(|r| r.attr("elementMetadata"))
            .or_else(|| se.attr("elementMetadata"))
            .and_then(|m| m.attr("listData"));
        let int_col = se.attr("int_colData").and_then(|c| c.attr("listData"));
        // obsm sources: reducedDims, then spatialCoords.
        let mut obsm_src: Vec<(String, Node)> = Vec::new();
        if let Some(rd) = int_col
            .and_then(|c| c.get("reducedDims"))
            .and_then(|r| r.attr("listData"))
        {
            for (name, (_, m)) in rd.names().into_iter().zip(rd.items()) {
                obsm_src.push((format!("X_{}", name.to_lowercase()), m.clone()));
            }
        }
        if let Some(sc) = int_col.and_then(|c| c.get("spatialCoords")) {
            obsm_src.push(("spatial".into(), sc.clone()));
        }

        let mut want: Vec<&Node> = vec![slot(x_node, "p")?];
        for (_, m) in &others {
            want.push(slot(m, "p")?);
        }
        for m in obs_cols.iter().chain(row_data.iter()) {
            want.extend(m.items().iter().map(|(_, c)| c));
        }
        let fetched: Fetched = fetch(&src, &want)?;

        let dims = super::seurat::ints(slot(x_node, "Dim")?, &fetched)?;
        let (n_vars, n_obs) = match dims[..] {
            [g, c] => (g as usize, c as usize),
            _ => return Err(bad(format!("assay '{layer}' has Dim {dims:?}"))),
        };
        let dimnames = |k: usize| {
            strs(
                x_node
                    .attr("Dimnames")
                    .and_then(|d| d.items().get(k))
                    .map(|(_, n)| n),
            )
        };
        let numbered = |n: usize| (0..n).map(|i| i.to_string()).collect::<Vec<_>>();
        let cells = strs(col_data.attr("rownames"))
            .or_else(|| dimnames(1))
            .unwrap_or_else(|| numbered(n_obs));
        let genes = strs(se.attr("NAMES"))
            .or_else(|| {
                strs(
                    row_ranges
                        .and_then(|r| r.attr("partitioning"))
                        .and_then(|p| p.attr("NAMES")),
                )
            })
            .or_else(|| {
                strs(
                    row_ranges
                        .and_then(|r| r.attr("ranges"))
                        .and_then(|p| p.attr("NAMES")),
                )
            })
            .or_else(|| dimnames(0))
            .unwrap_or_else(|| numbered(n_vars));
        if cells.len() != n_obs || genes.len() != n_vars {
            return Err(bad(format!(
                "{} cell and {} gene names for a {n_vars} x {n_obs} assay",
                cells.len(),
                genes.len()
            )));
        }

        // One part, every cell in object order.
        let rows = vec![(0..n_obs).collect::<Vec<_>>()];
        let x = build_matrix(&[(layer.to_string(), x_node)], &rows, n_vars, &fetched)?;
        let mut layers: Vec<(String, Matrix)> = Vec::new();
        for (name, m) in &others {
            match build_matrix(&[(name.clone(), *m)], &rows, n_vars, &fetched) {
                Ok(mat) => layers.push((name.clone(), mat)),
                Err(e) => tracing::warn!("rds: assay '{name}' is not read: {e}"),
            }
        }
        let order: Vec<usize> = (0..n_obs).collect();
        let obs_columns = match obs_cols {
            Some(c) => meta_columns(c, &cells, &order, &fetched)?,
            None => Vec::new(),
        };
        let var_columns = match row_data {
            Some(c) => {
                let all: Vec<usize> = (0..n_vars).collect();
                meta_columns(c, &genes, &all, &fetched).unwrap_or_else(|e| {
                    tracing::warn!("rds: rowData is not read: {e}");
                    Vec::new()
                })
            }
            None => Vec::new(),
        };
        Ok(Self {
            src,
            chunk_size,
            n_vars,
            x,
            layers,
            obs: ObsTable {
                index: cells.clone(),
                columns: obs_columns,
            },
            var: VarTable {
                index: genes,
                columns: var_columns,
            },
            cells,
            row_of: Default::default(),
            obsm_src,
            varm_src: Vec::new(),
            graphs: Vec::new(),
            obsp: HashMap::new(),
        })
    }
}
