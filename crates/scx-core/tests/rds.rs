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

/// An R ordered factor keeps `ordered`; a plain factor does not.
fn check_ordered(name: &str) {
    let mut r = block_on(open(tiny(name).to_str().unwrap(), &OpenOptions::new(4))).unwrap();
    let obs = block_on(r.obs()).unwrap();
    let ordered = |col: &str| match &obs.columns.iter().find(|c| c.name == col).unwrap().data {
        scx_core::ir::ColumnData::Categorical { ordered, .. } => *ordered,
        other => panic!("{name}: {col} is {}", other.dtype_str()),
    };
    assert!(ordered("cell_type_ordered"), "{name}");
    assert!(!ordered("cell_type"), "{name}");
}

#[test]
fn ordered_factors_stay_ordered() {
    for name in ["tiny_v4.rds", "tiny_v5.rds", "tiny_v5_split.rds"] {
        check_ordered(name);
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
    assert!(
        err.to_string()
            .contains("not a Seurat object; the file holds a integer vector"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A gzip around the (already gzipped) .rds, as GEO serves `x.rds.gz`, and a
/// gzip around an uncompressed one: both are detected and read.
#[test]
fn gzipped_rds_is_read() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let rds = std::fs::read(tiny("tiny_v4.rds")).unwrap();
    let mut plain = Vec::new();
    flate2::read::MultiGzDecoder::new(&rds[..])
        .read_to_end(&mut plain)
        .unwrap();
    for (name, inner) in [("double.rds.gz", &rds), ("single.rds.gz", &plain)] {
        let p = dir.path().join(name);
        let mut enc = flate2::write::GzEncoder::new(
            std::fs::File::create(&p).unwrap(),
            flate2::Compression::fast(),
        );
        enc.write_all(inner).unwrap();
        enc.finish().unwrap();
        assert_eq!(detect(&p), Some(Format::Rds), "{name}");
        let r = block_on(open(p.to_str().unwrap(), &OpenOptions::new(4))).unwrap();
        assert_eq!(r.shape(), (6, 5), "{name}");
    }
}

/// `rds::tree` shows the object's structure without reading its matrices.
#[test]
fn tree_shows_classes_slots_and_lengths() {
    let t = scx_core::rds::tree(&tiny("tiny_v5_split.rds"), 4).unwrap();
    assert!(t.starts_with("<root>: S4 <Seurat>"), "{t}");
    assert!(t.contains("@layers: list[2]"), "{t}");
    assert!(
        t.contains("$counts.d0: S4 <dgCMatrix>") || t.contains("$counts.d1: S4 <dgCMatrix>"),
        "{t}"
    );
    assert!(t.contains("@cells: lgl[") && t.contains("<LogMap>"), "{t}");
}

/// Without the `net` feature a URL says so, instead of "could not detect format".
#[cfg(not(feature = "net"))]
#[test]
fn url_without_net_feature_says_so() {
    let err = block_on(open("https://example.org/obj.rds", &OpenOptions::new(4)))
        .err()
        .unwrap();
    assert!(err.to_string().contains("without network support"), "{err}");
}

/// Base R objects a Seurat object can carry (closures in its commands,
/// environments, ALTREP vectors, complex and raw vectors, latin1 strings)
/// parse in every container R writes: format version 3, version 2 and `.rda`
/// (scripts/make_rds_zoo_fixtures.R). An ASCII save is refused by name.
#[test]
fn any_base_r_object_parses() {
    for (name, pad) in [("zoo.rds", ""), ("zoo_v2.rds", ""), ("zoo.rda", "  ")] {
        let t = scx_core::rds::tree(&tiny(name), 3).unwrap();
        for line in [
            "$intseq: int[5]",
            "$realseq: num[3] [2147483648.0, 2147483649.0, 2147483650.0]",
            "$wrap_real: num[3] [1.0, 2.0, 3.0]",
            "$wrap_int: int[3] [1, 2, 3]",
            "$latin1: chr[1] [\"café\"]",
            "$strs: chr[2] [\"a\", \"NA\"]",
            "$fn: list[2]",
            "... 6 more elements",
        ] {
            assert!(t.contains(&format!("  {pad}{line}")), "{name}: {line}\n{t}");
        }
    }
    let e = scx_core::rds::tree(&tiny("zoo_ascii.rds"), 3)
        .unwrap_err()
        .to_string();
    assert!(e.contains("ASCII"), "{e}");
}

/// big_*: past the parser's LAZY_MIN, so @i, @x and meta.data are fetched from
/// the stream after parsing, the path real files take. Every value follows a
/// formula in scripts/make_tiny_fixtures_rds.R, recomputed here per cell; the
/// v5 object is split by batch, so rows are matched by cell name. A complex
/// and a list column are skipped.
fn check_big(name: &str) {
    use scx_core::ir::ColumnData;
    let mut r = block_on(open(tiny(name).to_str().unwrap(), &OpenOptions::new(700))).unwrap();
    assert_eq!(r.shape(), (5000, 6), "{name}");
    let obs = block_on(r.obs()).unwrap();
    let ks: Vec<usize> = obs.index.iter().map(|c| c[1..].parse().unwrap()).collect();

    let mut rows = 0;
    for c in block_on(r.x_stream().collect::<Vec<_>>()) {
        let c = c.unwrap();
        let data = match &c.data.data {
            TypedVec::F64(v) => v,
            other => panic!("{name}: X is {:?}", other.dtype()),
        };
        for i in 0..c.nrows {
            let k = ks[c.row_offset + i];
            let (a, b) = (c.data.indptr[i] as usize, c.data.indptr[i + 1] as usize);
            let got: Vec<(u32, f64)> = (a..b).map(|j| (c.data.indices[j], data[j])).collect();
            let want: Vec<(u32, f64)> = (0..6u32)
                .map(|g| (g, ((7 * k + 3 * g as usize) % 5) as f64))
                .filter(|&(_, v)| v != 0.0)
                .collect();
            assert_eq!(got, want, "{name}: cell c{k}");
        }
        rows += c.nrows;
    }
    assert_eq!(rows, 5000, "{name}");

    let col = |n: &str| {
        obs.columns
            .iter()
            .find(|c| c.name == n)
            .unwrap_or_else(|| panic!("{name}: no column {n}"))
    };
    let missing = |n: &str, i: usize| col(n).mask.as_ref().is_some_and(|m| !m[i]);
    for skipped in ["cplx", "lst"] {
        assert!(
            obs.columns.iter().all(|c| c.name != skipped),
            "{name}: {skipped}"
        );
    }
    for (i, &k) in ks.iter().enumerate() {
        let at = format!("{name}: c{k}");
        assert_eq!(missing("grp", i), k % 50 == 0, "{at} grp");
        assert_eq!(missing("score", i), k % 40 == 0, "{at} score");
        assert_eq!(missing("n", i), k % 30 == 0, "{at} n");
        assert_eq!(missing("flag", i), k % 60 == 0, "{at} flag");
        assert_eq!(missing("label", i), k % 70 == 0, "{at} label");
        if let ColumnData::Categorical { codes, levels, .. } = &col("grp").data {
            if k % 50 != 0 {
                assert_eq!(levels[codes[i] as usize], ["a", "b", "c"][k % 3], "{at}");
            }
        } else {
            panic!("{at}: grp is not categorical");
        }
        match (
            &col("score").data,
            &col("n").data,
            &col("flag").data,
            &col("label").data,
        ) {
            (
                ColumnData::Float(s),
                ColumnData::Int(n),
                ColumnData::Bool(f),
                ColumnData::String(l),
            ) => {
                if k % 40 != 0 {
                    assert_eq!(s[i], k as f64 / 4.0, "{at}");
                }
                if k % 30 != 0 {
                    assert_eq!(n[i], k as i32, "{at}");
                }
                if k % 60 != 0 {
                    assert_eq!(f[i], k % 2 == 0, "{at}");
                }
                if k % 70 != 0 {
                    assert_eq!(l[i], format!("s{k}"), "{at}");
                }
            }
            other => panic!("{at}: column types {other:?}"),
        }
    }
}

#[test]
fn large_seurat_v4_read_from_the_stream() {
    check_big("big_v4.rds");
}

#[test]
fn large_seurat_v5_split_read_from_the_stream() {
    check_big("big_v5_split.rds");
}

/// A `.rda` holds a pairlist of saved variables; the Seurat object is found
/// among them.
#[test]
fn seurat_object_found_in_rda() {
    check_big("big_v4.rda");
}

/// What can't be read says what it is: a list that holds no Seurat object is
/// described by its elements, and a dense (non-dgCMatrix) layer is named.
#[test]
fn unreadable_objects_are_described() {
    let err = |name: &str| {
        scx_core::rds::RdsSeuratReader::open(&tiny(name), 4, None, None)
            .err()
            .unwrap()
            .to_string()
    };
    let e = err("zoo.rds");
    assert!(
        e.contains("not a Seurat object; the file holds a list of 18 (intseq: integer vector, realseq: numeric vector")
            && e.contains("+10 more"),
        "{e}"
    );
    let e = err("tiny_v5_dense.rds");
    assert!(
        e.contains("layer 'counts'") && e.contains("only dgCMatrix layers"),
        "{e}"
    );
}
