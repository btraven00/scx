//! End-to-end: `scx inspect` and `scx convert` on a bixverse store, and the
//! error a build without the `bixverse` feature gives for one.

use std::process::{Command, Output};

fn scx(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(args)
        .output()
        .expect("failed to run scx")
}

#[cfg(feature = "bixverse")]
#[test]
fn inspects_and_converts_a_store() {
    use bixverse_sc_data::data_io::{CellGeneSparseWriter, CsrCellChunk};
    use futures::StreamExt;
    use scx_core::dtype::TypedVec;
    use scx_core::h5ad::H5AdReader;
    use scx_core::stream::DatasetReader;

    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("counts_cells.bin");
    let rows: [&[(u32, u32)]; 3] = [&[(0, 1), (2, 3)], &[(1, 2)], &[(0, 4), (3, 5)]];
    let mut w = CellGeneSparseWriter::new(&bin, true, rows.len(), 4, 1e4).unwrap();
    for (i, row) in rows.iter().enumerate() {
        let (idx, data): (Vec<u32>, Vec<u32>) = row.iter().copied().unzip();
        w.write_cell_chunk(CsrCellChunk::from_data(&data, &idx, i, 1e4, true))
            .unwrap();
    }
    w.finalise().unwrap();
    std::fs::write(
        dir.path().join("obs.tsv"),
        "cell_id\tbatch\na\tx\nb\tx\nc\ty\n",
    )
    .unwrap();

    let out = scx(&["inspect", dir.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Format : bixverse store"), "{stdout}");
    assert!(stdout.contains("3 obs × 4 vars"), "{stdout}");

    let h5ad = dir.path().join("out.h5ad");
    let out = scx(&["convert", bin.to_str().unwrap(), h5ad.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mut r = H5AdReader::open(&h5ad, 100).unwrap();
    assert_eq!(r.shape(), (3, 4));
    let obs = futures::executor::block_on(r.obs()).unwrap();
    assert_eq!(obs.index, ["a", "b", "c"]);
    let total: f64 = futures::executor::block_on(r.x_stream().collect::<Vec<_>>())
        .into_iter()
        .map(|c| match c.unwrap().data.data {
            TypedVec::F32(v) => v.iter().map(|&x| x as f64).sum::<f64>(),
            TypedVec::F64(v) => v.iter().sum(),
            _ => panic!("unexpected dtype"),
        })
        .sum();
    assert_eq!(total, 15.0);
}

#[cfg(not(feature = "bixverse"))]
#[test]
fn without_the_feature_a_store_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    // The magic alone is enough for detection.
    let bin = dir.path().join("counts_cells.bin");
    std::fs::write(&bin, b"SCRNASEQ").unwrap();
    let out_path = dir.path().join("out.h5ad");

    for args in [
        vec!["inspect", bin.to_str().unwrap()],
        vec!["convert", bin.to_str().unwrap(), out_path.to_str().unwrap()],
    ] {
        let out = scx(&args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{args:?} succeeded");
        assert!(stderr.contains("`bixverse` feature"), "{args:?}: {stderr}");
    }
    assert!(!out_path.exists());
}
