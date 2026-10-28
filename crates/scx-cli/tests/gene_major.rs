//! Integration coverage for `scx convert --gene-major` and for gene-major
//! (CSC) X as input. Uses the committed tiny fixtures.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use futures::StreamExt;
use scx_core::h5ad::H5AdReader;
use scx_core::stream::{ChunkStream, DatasetReader};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_scx"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn scx(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("failed to run scx")
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// The chunks of `stream` as a dense matrix of `n_rows` × `n_cols`, each
/// chunk's `row_offset` + local row indexing its rows.
fn dense(stream: ChunkStream<'_>, n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut m = vec![0.0; n_rows * n_cols];
    futures::executor::block_on(stream.for_each(|c| {
        let c = c.unwrap();
        let v = c.data.data.to_f64();
        for r in 0..c.nrows {
            for k in c.data.indptr[r] as usize..c.data.indptr[r + 1] as usize {
                m[(c.row_offset + r) * n_cols + c.data.indices[k] as usize] = v[k];
            }
        }
        futures::future::ready(())
    }));
    m
}

/// `m` (n_rows × n_cols) transposed.
fn t(m: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    (0..n_cols * n_rows)
        .map(|i| m[(i % n_rows) * n_cols + i / n_rows])
        .collect()
}

/// Regression: BPCells h5seurat declares f64 but streams u32 chunks, and the
/// transpose refused them with a dtype mismatch. Gene-major output from it
/// must hold the same matrix as a plain convert, here over many passes.
#[test]
fn gene_major_from_bpcells_h5seurat_matches_plain_convert() {
    let tmp = tempfile::tempdir().unwrap();
    let (plain, genes) = (tmp.path().join("plain.h5ad"), tmp.path().join("genes.h5ad"));
    let src = fixture("tiny_v5_bpcells.h5seurat");
    assert!(scx(&["convert", s(&src), s(&plain)]).status.success());
    let out = scx(&[
        "convert",
        s(&src),
        s(&genes),
        "--gene-major",
        "--transpose-mem",
        "1e-8",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mut p = H5AdReader::open(&plain, 3).unwrap();
    let mut g = H5AdReader::open(&genes, 3).unwrap();
    let (n_obs, n_vars) = p.shape();
    assert_eq!(g.shape(), (n_obs, n_vars));
    assert!(!g.x_col_ptr().is_empty(), "output is not gene-major");
    let want = dense(p.x_stream(), n_obs, n_vars);
    let by_gene = dense(
        g.x_stream_cols(std::iter::once(0..n_vars).collect()),
        n_vars,
        n_obs,
    );
    assert_eq!(t(&by_gene, n_vars, n_obs), want);
}

/// Regression: gene-major X used to fail only once X streamed, after the
/// output was partly written; concat then left a corrupt h5ad behind.
/// convert, concat and snapshot must refuse it up front, write nothing, and
/// say how to fix the input.
#[test]
fn gene_major_input_is_refused_before_anything_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny.h5ad");
    let genes = tmp.path().join("genes.h5ad");
    assert!(scx(&["convert", s(&src), s(&genes), "--gene-major"])
        .status
        .success());

    let back = tmp.path().join("back.h5ad");
    let cc = tmp.path().join("cc.h5ad");
    let snap = tmp.path().join("snap");
    let runs = [
        (
            "convert",
            scx(&["convert", s(&genes), s(&back)]),
            back.clone(),
        ),
        (
            "concat",
            scx(&["concat", s(&src), s(&genes), "-o", s(&cc)]),
            cc.clone(),
        ),
        (
            "snapshot",
            scx(&["snapshot", s(&genes), s(&snap)]),
            snap.clone(),
        ),
    ];
    for (cmd, out, path) in runs {
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{cmd} accepted gene-major X");
        assert!(
            err.contains("gene-major") && err.contains("tocsr"),
            "{cmd}: {err}"
        );
        assert!(!path.exists(), "{cmd} left {} behind", path.display());
    }
}

/// `--x-slot layers/<name>` with `--gene-major` puts the gene-major matrix
/// in that layer.
#[test]
fn gene_major_into_a_layer() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny.h5ad");
    let out_path = tmp.path().join("layer.h5ad");
    let out = scx(&[
        "convert",
        s(&src),
        s(&out_path),
        "--gene-major",
        "--x-slot",
        "layers/counts",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mut want = H5AdReader::open(&src, 3).unwrap();
    let mut got = H5AdReader::open_layer(&out_path, 3, Some("counts")).unwrap();
    let (n_obs, n_vars) = want.shape();
    assert!(!got.x_col_ptr().is_empty(), "layer is not gene-major");
    let by_gene = dense(
        got.x_stream_cols(std::iter::once(0..n_vars).collect()),
        n_vars,
        n_obs,
    );
    assert_eq!(
        t(&by_gene, n_vars, n_obs),
        dense(want.x_stream(), n_obs, n_vars)
    );
}
