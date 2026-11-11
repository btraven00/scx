//! SingleCellExperiment / SpatialExperiment `.rds` (scripts/make_spe_fixtures.R)
//! and streaming an `.rds` by gene.
//!
//! - tiny_sce.rds / tiny_spe.rds: the tiny dataset (expected.json).
//! - visiumhd_spe.rds: 5000 bins x 30 genes of a real 10x Visium HD
//!   SpatialExperiment, checked against what R read back (visiumhd_spe.json).

#![allow(clippy::single_range_in_vec_init, clippy::needless_range_loop)]

use futures::executor::block_on;
use futures::StreamExt;
use scx_core::detect::{detect, Format};
use scx_core::dtype::TypedVec;
use scx_core::stream::DatasetReader;
use scx_core::{open, OpenOptions};
use serde_json::Value;
use std::ops::Range;
use std::path::PathBuf;

fn tiny(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn json(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(tiny(name)).unwrap()).unwrap()
}

fn reader(name: &str, chunk: usize) -> Box<dyn DatasetReader> {
    let p = tiny(name);
    assert_eq!(detect(&p), Some(Format::Rds));
    block_on(open(p.to_str().unwrap(), &OpenOptions::new(chunk))).unwrap()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

/// X densified (n_obs x n_vars) from the cell stream.
fn dense_by_cell(r: &mut dyn DatasetReader) -> Vec<Vec<f64>> {
    let (n_obs, n_vars) = r.shape();
    let mut d = vec![vec![0.0; n_vars]; n_obs];
    for c in block_on(r.x_stream().collect::<Vec<_>>()) {
        let c = c.unwrap();
        let TypedVec::F64(x) = &c.data.data else {
            panic!("f64")
        };
        for k in 0..c.nrows {
            for t in c.data.indptr[k] as usize..c.data.indptr[k + 1] as usize {
                d[c.row_offset + k][c.data.indices[t] as usize] = x[t];
            }
        }
    }
    d
}

/// X densified (n_obs x n_vars, zero outside `ranges`) from the gene stream,
/// checking each chunk's shape and that cells ascend within a gene.
fn dense_by_gene(r: &mut dyn DatasetReader, ranges: Vec<Range<usize>>) -> Vec<Vec<f64>> {
    let (n_obs, n_vars) = r.shape();
    let mut d = vec![vec![0.0; n_vars]; n_obs];
    for c in block_on(r.x_stream_cols(ranges).collect::<Vec<_>>()) {
        let c = c.unwrap();
        assert_eq!(c.data.shape, (c.nrows, n_obs));
        let TypedVec::F64(x) = &c.data.data else {
            panic!("f64")
        };
        for k in 0..c.nrows {
            let span = c.data.indptr[k] as usize..c.data.indptr[k + 1] as usize;
            assert!(c.data.indices[span.clone()].windows(2).all(|w| w[0] < w[1]));
            for t in span {
                d[c.data.indices[t] as usize][c.row_offset + k] = x[t];
            }
        }
    }
    d
}

fn check_tiny(name: &str) {
    let e = json("expected.json");
    let mut r = reader(name, 4);
    assert_eq!(r.shape(), (6, 5));
    let obs = block_on(r.obs()).unwrap();
    assert_eq!(obs.index, strs(&e["obs_names"]));
    let var = block_on(r.var()).unwrap();
    assert_eq!(var.index, strs(&e["var_names"]));
    let sym = var
        .columns
        .iter()
        .find(|c| c.name == "gene_symbol")
        .unwrap();
    let want = strs(&e["var"]["gene_symbol"]["values"]);
    assert!(
        (0..5).all(|g| sym.data.value_str(g) == want[g]),
        "{name}: rowData"
    );

    // colData keeps R types: factor, integer, logical, double, character.
    let types: Vec<(&str, &str)> = obs
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c.data.dtype_str()))
        .collect();
    for (col, t) in [
        ("cell_type", "categorical"),
        ("batch", "int32"),
        ("flag", "bool"),
        ("score", "float64"),
        ("donor", "string"),
    ] {
        assert!(
            types.contains(&(col, t)),
            "{name}: {col} not {t} in {types:?}"
        );
    }
    let na = |col: &str, row: usize| {
        obs.columns
            .iter()
            .find(|c| c.name == col)
            .unwrap()
            .is_na(row)
    };
    for (col, spec) in e["obs"].as_object().unwrap() {
        for row in 0..6 {
            assert_eq!(
                na(col, row),
                spec["values"][row].is_null(),
                "{name}: {col}[{row}] NA"
            );
        }
    }

    let counts: Vec<Vec<f64>> = e["counts"].as_array().unwrap().iter().map(f64s).collect();
    assert_eq!(dense_by_cell(r.as_mut()), counts, "{name}: X by cell");
    assert_eq!(
        dense_by_gene(r.as_mut(), vec![0..5]),
        counts,
        "{name}: X by gene"
    );

    // logcounts is a layer: expected.json's X.
    let metas = block_on(r.layer_metas()).unwrap();
    assert_eq!(
        metas.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
        ["logcounts"]
    );
    let mut got = vec![vec![0.0; 5]; 6];
    for c in block_on(r.layer_stream(&metas[0], 4).collect::<Vec<_>>()) {
        let c = c.unwrap();
        let TypedVec::F64(x) = &c.data.data else {
            panic!("f64")
        };
        for k in 0..c.nrows {
            for t in c.data.indptr[k] as usize..c.data.indptr[k + 1] as usize {
                got[c.row_offset + k][c.data.indices[t] as usize] = x[t];
            }
        }
    }
    let want: Vec<Vec<f64>> = e["X"].as_array().unwrap().iter().map(f64s).collect();
    for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
        assert!((g - w).abs() < 1e-6, "{name}: logcounts {g} vs {w}");
    }

    // reducedDim PCA -> obsm X_pca, row-major.
    let obsm = block_on(r.obsm()).unwrap();
    let pca = &obsm.map["X_pca"];
    let want: Vec<Vec<f64>> = e["obsm"]["X_pca"]
        .as_array()
        .unwrap()
        .iter()
        .map(f64s)
        .collect();
    assert_eq!(pca.shape, (6, want[0].len()));
    assert_eq!(pca.data, want.concat(), "{name}: X_pca");
}

#[test]
fn single_cell_experiment() {
    check_tiny("tiny_sce.rds");
    let obsm = block_on(reader("tiny_sce.rds", 4).obsm()).unwrap();
    assert!(!obsm.map.contains_key("spatial"));
}

#[test]
fn spatial_experiment() {
    check_tiny("tiny_spe.rds");
    let obsm = block_on(reader("tiny_spe.rds", 4).obsm()).unwrap();
    let sp = &obsm.map["spatial"];
    assert_eq!(sp.shape, (6, 2));
    let want: Vec<f64> = (0..6)
        .flat_map(|k| [10.0 * k as f64, 100.0 - k as f64])
        .collect();
    assert_eq!(sp.data, want);
}

#[test]
fn se_assay_is_picked_with_layer() {
    let p = tiny("tiny_sce.rds");
    let r = scx_core::rds::RdsReader::open(&p, 4, None, Some("logcounts")).unwrap();
    assert_eq!(r.shape(), (6, 5));
    let err = scx_core::rds::RdsReader::open(&p, 4, None, Some("spliced"))
        .err()
        .unwrap();
    assert!(
        err.to_string()
            .contains("no assay 'spliced' (have: counts, logcounts)"),
        "{err}"
    );
    let err = scx_core::rds::RdsReader::open(&p, 4, Some("RNA"), None)
        .err()
        .unwrap();
    assert!(err.to_string().contains("--layer"), "{err}");
}

/// A real Visium HD SpatialExperiment, past the lazy-read threshold.
#[test]
fn visium_hd_spatial_experiment() {
    let e = json("visiumhd_spe.json");
    let mut r = reader("visiumhd_spe.rds", 512);
    let (n_obs, n_vars) = r.shape();
    assert_eq!((n_obs, n_vars), (5000, 30));
    let obs = block_on(r.obs()).unwrap();
    assert_eq!(obs.index, strs(&e["obs_names"]));
    let var = block_on(r.var()).unwrap();
    assert_eq!(var.index, strs(&e["var_names"]));
    let col = |cols: &[scx_core::ir::Column], n: &str, i: usize| {
        cols.iter().find(|c| c.name == n).unwrap().data.value_str(i)
    };
    for g in 0..n_vars {
        assert_eq!(col(&var.columns, "ID", g), e["var_id"][g].as_str().unwrap());
        assert_eq!(
            col(&var.columns, "Type", g),
            e["var_type"][g].as_str().unwrap()
        );
    }
    for k in [0, 1234, 4999] {
        assert_eq!(
            col(&obs.columns, "in_tissue", k),
            e["in_tissue"][k].to_string()
        );
        assert_eq!(
            col(&obs.columns, "array_row", k),
            e["array_row"][k].to_string()
        );
    }

    let by_cell = dense_by_cell(r.as_mut());
    let gene_sums: Vec<f64> = (0..n_vars)
        .map(|g| by_cell.iter().map(|row| row[g]).sum())
        .collect();
    assert_eq!(gene_sums, f64s(&e["gene_sums"]));
    let cell_sums: Vec<f64> = by_cell.iter().map(|row| row.iter().sum()).collect();
    assert_eq!(cell_sums, f64s(&e["cell_sums"]));
    let nnz = by_cell.iter().flatten().filter(|v| **v != 0.0).count();
    assert_eq!(nnz as u64, e["nnz"][0].as_u64().unwrap()); // jsonlite boxes scalars

    // By gene: all of them, then two disjoint ranges (unsorted, one empty).
    assert_eq!(dense_by_gene(r.as_mut(), vec![0..n_vars]), by_cell);
    let part = dense_by_gene(r.as_mut(), vec![20..25, 3..9, 7..7]);
    for (k, row) in part.iter().enumerate() {
        for (g, &v) in row.iter().enumerate() {
            let kept = (3..9).contains(&g) || (20..25).contains(&g);
            assert_eq!(v, if kept { by_cell[k][g] } else { 0.0 }, "[{k}, {g}]");
        }
    }
    let err = block_on(r.x_stream_cols(vec![0..31]).next())
        .unwrap()
        .err()
        .unwrap();
    assert!(err.to_string().contains("beyond"), "{err}");

    let obsm = block_on(r.obsm()).unwrap();
    let sp = &obsm.map["spatial"];
    assert_eq!(sp.shape, (n_obs, 2));
    let want: Vec<f64> = e["spatial"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(f64s)
        .collect();
    // jsonlite writes 15 significant digits.
    assert_eq!(sp.data.len(), want.len());
    for (g, w) in sp.data.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-12 * w.abs(), "spatial {g} vs {w}");
    }
}

/// Gene streaming over a split Seurat object read lazily from the stream:
/// big_v5_split.rds holds X[k, g] = (7k + 3g) % 5 for cell "c<k>", in two
/// layers, so X rows are not in k order.
#[test]
fn seurat_split_layers_stream_by_gene() {
    let mut r = reader("big_v5_split.rds", 3);
    let obs = block_on(r.obs()).unwrap();
    let got = dense_by_gene(r.as_mut(), vec![1..3, 4..6]);
    for (row, name) in obs.index.iter().enumerate() {
        let k: usize = name[1..].parse().unwrap();
        for g in 0..6 {
            let want = if [1, 2, 4, 5].contains(&g) {
                ((7 * k + 3 * g) % 5) as f64
            } else {
                0.0
            };
            assert_eq!(got[row][g], want, "{name} gene {g}");
        }
    }
}
