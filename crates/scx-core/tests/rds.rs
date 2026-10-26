//! Seurat `.rds` reader against `tests/fixtures/tiny/expected.json`: the tiny
//! dataset saved as a v3/v4 `Assay`, a v5 `Assay5`, and an `Assay5` split
//! into one counts layer per donor (scripts/make_tiny_fixtures_rds.R).
//!
//! Split layers put X rows in layer order, so everything is compared by cell
//! name.

use futures::executor::block_on;
use futures::StreamExt;
use scx_core::detect::{detect, Format};
use scx_core::dtype::TypedVec;
use scx_core::{open, OpenOptions};
use serde_json::Value;
use std::path::PathBuf;

fn tiny(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(tiny("expected.json")).unwrap()).unwrap()
}

/// Open `name` through the factory (so detection is exercised too) with
/// chunks of 4 rows, which split the 6 cells across chunks and layers.
fn check(name: &str) {
    let path = tiny(name);
    assert_eq!(detect(&path), Some(Format::Rds));
    let e = expected();
    let mut r = block_on(open(path.to_str().unwrap(), &OpenOptions::new(4))).unwrap();
    let (n_obs, n_vars) = r.shape();
    assert_eq!((n_obs, n_vars), (6, 5));

    let var = block_on(r.var()).unwrap();
    let want_genes: Vec<&str> = e["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(var.index, want_genes);

    let obs = block_on(r.obs()).unwrap();
    let names: Vec<&str> = e["obs_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    // expected.json row for each X row
    let src: Vec<usize> = obs
        .index
        .iter()
        .map(|c| {
            names
                .iter()
                .position(|n| n == c)
                .unwrap_or_else(|| panic!("unknown cell {c}"))
        })
        .collect();

    // X: densify the stream and compare with the counts.
    let mut dense = vec![vec![0.0f64; n_vars]; n_obs];
    let chunks: Vec<_> = block_on(r.x_stream().collect::<Vec<_>>());
    let mut rows = 0;
    for c in chunks {
        let c = c.unwrap();
        let TypedVec::F64(data) = &c.data.data else {
            panic!("expected f64 data")
        };
        for k in 0..c.nrows {
            let (lo, hi) = (c.data.indptr[k] as usize, c.data.indptr[k + 1] as usize);
            for j in lo..hi {
                dense[c.row_offset + k][c.data.indices[j] as usize] = data[j];
            }
        }
        rows += c.nrows;
    }
    assert_eq!(rows, n_obs);
    assert_eq!(r.x_indptr().len(), n_obs + 1);
    for (row, &s) in src.iter().enumerate() {
        let want: Vec<f64> = e["counts"][s]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(dense[row], want, "{name}: X row for {}", names[s]);
    }

    // obs: every column of expected.json, NA where it has null.
    for (col_name, spec) in e["obs"].as_object().unwrap() {
        let col = obs
            .columns
            .iter()
            .find(|c| &c.name == col_name)
            .unwrap_or_else(|| panic!("{name}: obs column {col_name} missing"));
        for (row, &s) in src.iter().enumerate() {
            let want = &spec["values"][s];
            if want.is_null() {
                assert!(col.is_na(row), "{name}: {col_name}[{row}] should be NA");
                continue;
            }
            assert!(!col.is_na(row), "{name}: {col_name}[{row}] is NA");
            let got = col.data.value_str(row);
            let ok = match spec["type"].as_str().unwrap() {
                "float" => got.parse::<f64>().unwrap() == want.as_f64().unwrap(),
                "bool" => {
                    got == if want.as_bool().unwrap() {
                        "True"
                    } else {
                        "False"
                    }
                }
                "int" => got == want.as_i64().unwrap().to_string(),
                _ => got == want.as_str().unwrap(),
            };
            assert!(ok, "{name}: {col_name}[{row}] = {got}, want {want}");
        }
    }
}

#[test]
fn seurat_v4_assay() {
    check("tiny_v4.rds");
}

#[test]
fn seurat_v5_assay5() {
    check("tiny_v5.rds");
}

#[test]
fn seurat_v5_split_layers() {
    check("tiny_v5_split.rds");
    // Rows follow layer order (one layer per donor), not the object's order.
    let mut r = block_on(open(
        tiny("tiny_v5_split.rds").to_str().unwrap(),
        &OpenOptions::new(4),
    ))
    .unwrap();
    let obs = block_on(r.obs()).unwrap();
    let donor = obs.columns.iter().find(|c| c.name == "donor").unwrap();
    let donors: Vec<String> = (0..6).map(|i| donor.data.value_str(i)).collect();
    let first = &donors[0];
    let switch = donors.iter().position(|d| d != first).unwrap();
    assert!(
        donors[switch..].iter().all(|d| d != first),
        "rows not grouped by layer: {donors:?}"
    );
}

#[test]
fn missing_assay_and_layer_are_errors() {
    let p = tiny("tiny_v5.rds");
    let err = scx_core::rds::RdsSeuratReader::open(&p, 4, Some("ADT"), None)
        .err()
        .unwrap();
    assert!(err.to_string().contains("no assay 'ADT'"), "{err}");
    let err = scx_core::rds::RdsSeuratReader::open(&p, 4, None, Some("scale.data"))
        .err()
        .unwrap();
    assert!(err.to_string().contains("no 'scale.data' layer"), "{err}");
}

#[test]
fn non_seurat_rds_is_rejected() {
    let dir = std::env::temp_dir().join(format!("scx-rds-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // saveRDS(1:3, compress = FALSE): an ALTREP compact_intseq, no class.
    let mut b = b"X\n".to_vec();
    for v in [3, 0x040601, 0x030500, 5] {
        b.extend_from_slice(&i32::to_be_bytes(v));
    }
    b.extend_from_slice(b"UTF-8");
    for v in [13, 3, 1, 2, 3] {
        b.extend_from_slice(&i32::to_be_bytes(v)); // INTSXP, length 3, 1 2 3
    }
    let p = dir.join("ints.rds");
    std::fs::write(&p, b).unwrap();
    assert_eq!(detect(&p), Some(Format::Rds));
    let err = scx_core::rds::RdsSeuratReader::open(&p, 4, None, None)
        .err()
        .unwrap();
    assert!(err.to_string().contains("not a Seurat object"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
