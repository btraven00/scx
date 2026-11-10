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

/// Gene-major X as input. convert keeps it gene-major into h5ad (copied by
/// gene), and transposes it for outputs that take X by cell, with a warning:
/// in one read when it fits --transpose-mem, else in blocks of cells.
#[test]
fn gene_major_input_converts() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny.h5ad");
    let genes = tmp.path().join("genes.h5ad");
    assert!(scx(&["convert", s(&src), s(&genes), "--gene-major"])
        .status
        .success());
    let mut want = H5AdReader::open(&src, 3).unwrap();
    let (n_obs, n_vars) = want.shape();
    let want = dense(want.x_stream(), n_obs, n_vars);

    // h5ad: stays gene-major, no warning.
    let back = tmp.path().join("back.h5ad");
    let out = scx(&["convert", s(&genes), s(&back)]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!String::from_utf8_lossy(&out.stderr).contains("warning"));
    let mut got = H5AdReader::open(&back, 3).unwrap();
    assert!(!got.x_col_ptr().is_empty(), "h5ad output is not gene-major");
    let by_gene = dense(
        got.x_stream_cols(std::iter::once(0..n_vars).collect()),
        n_vars,
        n_obs,
    );
    assert_eq!(t(&by_gene, n_vars, n_obs), want);

    // zarr: by cell, after a warning.
    #[cfg(feature = "zarr")]
    {
        let z = tmp.path().join("cells.zarr");
        let out = scx(&["convert", s(&genes), s(&z)]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{err}");
        assert!(err.contains("warning: X is gene-major"), "{err}");
        let mut got = scx_core::zarr_ad::ZarrAdReader::open(&z, 3).unwrap();
        assert_eq!(dense(got.x_stream(), n_obs, n_vars), want);
        // diff reads the gene-major side by cell too.
        let out = scx(&["diff", s(&genes), s(&z)]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // Over budget: a block of cells per read (here one cell each), same X.
    let big = tmp.path().join("big.h5seurat");
    let out = scx(&["convert", s(&genes), s(&big), "--transpose-mem", "1e-12"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(err.contains("read once per block of cells"), "{err}");
    let out = scx(&["diff", s(&genes), s(&big)]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// concat and snapshot still take X by cell only: they refuse gene-major
/// input up front, write nothing, and say how to fix it.
#[test]
fn gene_major_input_is_refused_by_concat_and_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny.h5ad");
    let genes = tmp.path().join("genes.h5ad");
    assert!(scx(&["convert", s(&src), s(&genes), "--gene-major"])
        .status
        .success());

    let cc = tmp.path().join("cc.h5ad");
    let snap = tmp.path().join("snap");
    let runs = [
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

/// Bad gene-major arguments are refused before anything is read: a
/// non-positive `--transpose-mem`, and a format that can't store X by gene.
#[test]
fn gene_major_arguments_are_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny.h5ad");
    let cases = [
        (
            tmp.path().join("a.h5ad"),
            &["--gene-major", "--transpose-mem", "0"][..],
            "--transpose-mem must be a positive",
        ),
        (
            tmp.path().join("b.h5seurat"),
            &["--gene-major"][..],
            "--gene-major only applies to .h5ad output",
        ),
    ];
    for (dst, extra, want) in cases {
        let mut args = vec!["convert", s(&src), s(&dst)];
        args.extend_from_slice(extra);
        let out = scx(&args);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{extra:?} was accepted");
        assert!(err.contains(want), "{extra:?}: {err}");
        assert!(!dst.exists(), "{extra:?} left {} behind", dst.display());
    }
}

/// `scx inspect` says when X is gene-major and how to stream it; on
/// cell-major input it shows per-cell stats, including obsp's.
#[test]
fn inspect_shows_the_x_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("tiny.h5ad");
    let genes = tmp.path().join("genes.h5ad");
    assert!(scx(&["convert", s(&src), s(&genes), "--gene-major"])
        .status
        .success());

    let out = scx(&["inspect", s(&genes)]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("gene-major (CSC)"), "{text}");

    let out = scx(&["inspect", s(&src)]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(!text.contains("gene-major"), "{text}");
    assert!(text.contains("connectivities"), "{text}");
}

/// Gene-major X written compressed goes through the parallel-deflate
/// appender like row-major X does; it must read back as the same matrix.
#[test]
fn compressed_gene_major_matches_plain_convert() {
    let tmp = tempfile::tempdir().unwrap();
    let (plain, genes) = (tmp.path().join("plain.h5ad"), tmp.path().join("genes.h5ad"));
    let src = fixture("tiny.h5ad");
    assert!(scx(&["convert", s(&src), s(&plain)]).status.success());
    let out = scx(&[
        "convert",
        s(&src),
        s(&genes),
        "--gene-major",
        "--compress",
        "4",
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
    let want = dense(p.x_stream(), n_obs, n_vars);
    let by_gene = dense(
        g.x_stream_cols(std::iter::once(0..n_vars).collect()),
        n_vars,
        n_obs,
    );
    assert_eq!(t(&by_gene, n_vars, n_obs), want);
}
