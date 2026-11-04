//! `scx convert --where`: the output holds only the matching cells, in every
//! output format, equal to the same rows of an unfiltered conversion.
//!
//! tiny.h5ad's cell_type is B T NaN B NK T, so `cell_type=B` + `cell_type=T`
//! keeps cells 0, 1, 3 and 5.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use futures::StreamExt;

const KEEP: [usize; 4] = [0, 1, 3, 5];
const B_OR_T: [&str; 4] = ["--where", "cell_type=B", "--where", "cell_type=T"];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

/// `scx convert` at info level; its log (stdout) on success.
fn convert(input: &Path, out: &Path, extra: &[&str]) -> String {
    let res = run(input, out, extra);
    assert!(
        res.status.success(),
        "scx convert failed:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
    String::from_utf8_lossy(&res.stdout).into_owned()
}

fn run(input: &Path, out: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .env("RUST_LOG", "info")
        .arg("convert")
        .arg(input)
        .arg(out)
        .args(extra)
        .output()
        .expect("failed to run scx convert")
}

fn open(path: &Path, layer: &str) -> Box<dyn scx_core::stream::DatasetReader + Send> {
    let opts = scx_core::OpenOptions {
        layer: Some(layer.to_string()),
        ..scx_core::OpenOptions::new(2)
    };
    futures::executor::block_on(scx_core::open(path.to_str().unwrap(), &opts)).unwrap()
}

fn dense(s: scx_core::stream::ChunkStream<'_>, n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    futures::executor::block_on(async {
        let mut out = vec![vec![0.0; n_cols]; n_rows];
        let mut s = s;
        while let Some(c) = s.next().await {
            let c = c.unwrap();
            let v = c.data.data.to_f64();
            for r in 0..c.nrows {
                for k in c.data.indptr[r] as usize..c.data.indptr[r + 1] as usize {
                    out[c.row_offset + r][c.data.indices[k] as usize] = v[k];
                }
            }
        }
        out
    })
}

fn x(path: &Path, layer: &str) -> Vec<Vec<f64>> {
    let mut r = open(path, layer);
    let (n, m) = r.shape();
    dense(r.x_stream(), n, m)
}

fn obs_names(path: &Path) -> Vec<String> {
    futures::executor::block_on(open(path, "counts").obs())
        .unwrap()
        .index
}

fn pick<T: Clone>(rows: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| rows[i].clone()).collect()
}

/// Each output format holds the subset's X and cell names. H5Seurat puts X
/// in `data` (tiny.h5ad has a `counts` layer); BPCells holds X alone.
#[test]
fn every_output_format_holds_the_subset() {
    let src = fixture("tiny.h5ad");
    let want = pick(&x(&src, "counts"), &KEEP);
    let names = pick(&obs_names(&src), &KEEP);
    let dir = tempfile::tempdir().unwrap();
    let cases: &[(&str, &[&str], &str)] = &[
        ("out.h5ad", &[], "counts"),
        ("out.zarr", &[], "counts"),
        ("bp.h5seurat", &[], "data"),
        ("dgc.h5seurat", &["--dgcmatrix"], "data"),
        ("out.bpcells", &[], "counts"),
    ];
    // Zarr output needs the optional `zarr` feature (off in the system-HDF5 CI job).
    let cases = cases.iter().filter(|c| cfg!(feature = "zarr") || !c.0.ends_with(".zarr"));
    for (name, extra, layer) in cases {
        let out = dir.path().join(name);
        let args: Vec<&str> = B_OR_T.iter().chain(extra.iter()).copied().collect();
        convert(&src, &out, &args);
        assert_eq!(x(&out, layer), want, "{name}: X");
        if !name.ends_with(".bpcells") {
            assert_eq!(obs_names(&out), names, "{name}: obs names");
        }
    }
}

/// h5ad output: layers and obsm by row, obsp by row and column, against
/// the same slots of an unfiltered conversion; var unchanged.
#[test]
fn h5ad_output_subsets_every_cell_axis_slot() {
    let src = fixture("tiny.h5ad");
    let dir = tempfile::tempdir().unwrap();
    let (full, sub) = (dir.path().join("full.h5ad"), dir.path().join("sub.h5ad"));
    convert(&src, &full, &[]);
    convert(&src, &sub, &B_OR_T);
    let (mut f, mut s) = (open(&full, "counts"), open(&sub, "counts"));
    let (fl, sl, fp, sp) = futures::executor::block_on(async {
        (
            f.layer_metas().await.unwrap(),
            s.layer_metas().await.unwrap(),
            f.obsp_metas().await.unwrap(),
            s.obsp_metas().await.unwrap(),
        )
    });
    assert_eq!(sl.len(), fl.len());
    assert_eq!(
        dense(s.layer_stream(&sl[0], 2), 4, 5),
        pick(&dense(f.layer_stream(&fl[0], 2), 6, 5), &KEEP)
    );
    let pf = dense(f.obsp_stream(&fp[0], 2), 6, 6);
    let want: Vec<Vec<f64>> = KEEP.iter().map(|&i| pick(&pf[i], &KEEP)).collect();
    assert_eq!(sp[0].shape, (4, 4));
    assert_eq!(dense(s.obsp_stream(&sp[0], 2), 4, 4), want);
    futures::executor::block_on(async {
        assert_eq!(s.shape(), (4, 5));
        assert_eq!(s.var().await.unwrap().index, f.var().await.unwrap().index);

        let (fm, sm) = (f.obsm().await.unwrap(), s.obsm().await.unwrap());
        let (a, b) = (&fm.map["X_pca"], &sm.map["X_pca"]);
        let k = a.shape.1;
        let rows: Vec<f64> = KEEP
            .iter()
            .flat_map(|&i| a.data[i * k..(i + 1) * k].to_vec())
            .collect();
        assert_eq!((b.shape, &b.data), ((4, k), &rows));

        let (fo, so) = (f.obs().await.unwrap(), s.obs().await.unwrap());
        assert_eq!(so.columns.len(), fo.columns.len());
        for (a, b) in fo.columns.iter().zip(&so.columns) {
            for (j, &i) in KEEP.iter().enumerate() {
                assert_eq!(a.is_na(i), b.is_na(j), "{}", a.name);
                assert_eq!(a.data.value_str(i), b.data.value_str(j), "{}", a.name);
            }
        }
    });
}

/// The filter is recorded in the sidecar and in uns["scx_provenance"].
#[test]
fn provenance_records_the_filter() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sub.h5ad");
    let args = [
        "--where",
        "cell_type=B",
        "--where",
        "cell_type=T",
        "--where",
        "donor=d1",
    ];
    convert(&fixture("tiny.h5ad"), &out, &args);
    let side: serde_json::Value =
        serde_json::from_slice(&std::fs::read(scx_core::provenance::sidecar_path(&out)).unwrap())
            .unwrap();
    // B or T, of donor d1: cells 1, 3 and 5.
    let want = serde_json::json!({ "cell_type": ["B", "T"], "donor": ["d1"] });
    assert_eq!(side["where"], want);
    assert_eq!(side["output"]["n_obs"], 3);
    let uns = futures::executor::block_on(open(&out, "counts").uns()).unwrap();
    let prov = &uns.raw["scx_provenance"];
    // uns stores a one-element list as a scalar.
    assert_eq!(prov["where"]["cell_type"], want["cell_type"]);
    assert_eq!(prov["where"]["donor"], "d1");
    assert_eq!(prov["n_obs"], 3);

    // Unfiltered: no `where` key.
    let plain = dir.path().join("plain.h5ad");
    convert(&fixture("tiny.h5ad"), &plain, &[]);
    let side: serde_json::Value =
        serde_json::from_slice(&std::fs::read(scx_core::provenance::sidecar_path(&plain)).unwrap())
            .unwrap();
    assert!(side.get("where").is_none(), "{side}");
}

/// tiny_raw_gzip.h5ad goes into a dgCMatrix h5seurat by copying X and
/// raw.X as stored. Under --where that copy would write every cell, so both
/// stream instead, and raw.X (Seurat's `counts`) keeps the matching rows.
#[test]
fn where_skips_the_chunk_copy_fast_path() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("raw_gzip.h5ad");
    std::fs::copy(fixture("tiny_raw_gzip.h5ad"), &src).unwrap();
    // The fixture has no obs columns; give it one. "a" keeps 0, 2, 3, 5.
    let keep = [0, 2, 3, 5];
    scx_core::h5ad::H5AdWriter::open_for_append(&src)
        .unwrap()
        .add_obs_column(&scx_core::ir::Column::new(
            "grp",
            scx_core::ir::ColumnData::Categorical {
                codes: vec![0, 1, 0, 0, 1, 0],
                levels: vec!["a".into(), "b".into()],
                ordered: false,
            },
        ))
        .unwrap();
    let log = convert(&src, &dir.path().join("all.h5seurat"), &["--dgcmatrix"]);
    assert!(
        log.contains("copied X into 'data' as stored"),
        "unfiltered copies: {log}"
    );

    let out = dir.path().join("sub.h5seurat");
    let log = convert(&src, &out, &["--dgcmatrix", "--where", "grp=a"]);
    assert!(!log.contains("as stored"), "--where must not copy: {log}");
    assert_eq!(x(&out, "data"), pick(&x(&src, "counts"), &keep));
    let mut r = open(&src, "counts");
    let meta = futures::executor::block_on(r.raw_meta()).unwrap().unwrap();
    let raw = dense(r.raw_stream(&meta, 2), 6, 5);
    assert_eq!(x(&out, "counts"), pick(&raw, &keep), "raw.X rows");
}

/// No matching cell writes a valid file with 0 cells, in every format.
#[test]
fn no_match_writes_an_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    for (name, extra) in [
        ("e.h5ad", &[][..]),
        ("e.zarr", &[]),
        ("e.h5seurat", &[]),
        ("dgc.h5seurat", &["--dgcmatrix"]),
    ]
    .into_iter()
    .filter(|(name, _)| cfg!(feature = "zarr") || !name.ends_with(".zarr"))
    {
        let out = dir.path().join(name);
        let args: Vec<&str> = ["--where", "cell_type=nope"]
            .into_iter()
            .chain(extra.iter().copied())
            .collect();
        convert(&fixture("tiny.h5ad"), &out, &args);
        let mut r = open(&out, "counts");
        assert_eq!(r.shape(), (0, 5), "{name}");
        assert!(dense(r.x_stream(), 0, 5).is_empty(), "{name}");
    }
}

/// A column obs doesn't have fails before anything is written, naming it.
#[test]
fn missing_column_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("x.h5ad");
    let res = run(&fixture("tiny.h5ad"), &out, &["--where", "celltype=B"]);
    assert!(!res.status.success());
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(
        err.contains("'celltype'") && err.contains("cell_type"),
        "{err}"
    );
    assert!(!out.exists());
}
