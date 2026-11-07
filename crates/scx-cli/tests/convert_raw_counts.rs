//! H5AD with the counts in `raw.X` (scanpy's usual layout) -> H5Seurat.
//!
//! Regression: scx ignored `raw`, so a normalised X went into Seurat's
//! `counts` slot and the real counts were dropped. `raw.X` now fills `counts`
//! and X goes to `data`, for both H5Seurat writers. A `raw` with a different
//! gene set than X can't share X's assay, so X stays the counts there.
//!
//! A gzip h5ad with sorted rows goes into a dgCMatrix h5seurat by copying
//! X and raw.X as stored; unsorted rows fall back to streaming.

use std::path::{Path, PathBuf};
use std::process::Command;

use futures::StreamExt;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn convert(input: &Path, out: &Path, extra: &[&str]) {
    let res = Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("convert")
        .arg(input)
        .arg(out)
        .args(extra)
        .output()
        .expect("failed to run scx convert");
    assert!(
        res.status.success(),
        "scx convert failed:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
}

/// The H5Seurat matrix in `slot`, dense, cells x genes.
fn slot(path: &Path, slot: &str) -> Vec<Vec<f64>> {
    futures::executor::block_on(async {
        let opts = scx_core::OpenOptions {
            source_hash: Default::default(),
            layer: Some(slot.to_string()),
            ..scx_core::OpenOptions::new(100)
        };
        let mut r = scx_core::open(path.to_str().unwrap(), &opts).await.unwrap();
        let (n_obs, n_vars) = r.shape();
        let mut out = vec![vec![0.0; n_vars]; n_obs];
        let mut stream = r.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            let values = chunk.data.data.to_f64();
            for row in 0..chunk.nrows {
                let span = chunk.data.indptr[row] as usize..chunk.data.indptr[row + 1] as usize;
                for k in span {
                    out[chunk.row_offset + row][chunk.data.indices[k] as usize] = values[k];
                }
            }
        }
        out
    })
}

// tests/fixtures/tiny: raw.X holds COUNTS, X holds COUNTS * 0.5.
const COUNTS: [[f64; 5]; 6] = [
    [1.0, 0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 4.0],
    [5.0, 0.0, 6.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 7.0],
    [8.0, 9.0, 0.0, 0.0, 0.0],
];

fn counts(n_genes: usize, scale: f64) -> Vec<Vec<f64>> {
    COUNTS
        .iter()
        .map(|row| row[..n_genes].iter().map(|v| v * scale).collect())
        .collect()
}

#[test]
fn raw_x_fills_counts_and_x_goes_to_data() {
    for extra in [&[][..], &["--dgcmatrix"][..]] {
        let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
        convert(&fixture("tiny_raw.h5ad"), out.path(), extra);
        assert_eq!(
            slot(out.path(), "counts"),
            counts(5, 1.0),
            "counts, {extra:?}"
        );
        assert_eq!(slot(out.path(), "data"), counts(5, 0.5), "data, {extra:?}");
    }
}

#[test]
fn raw_with_other_genes_leaves_x_as_counts() {
    let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    convert(
        &fixture("tiny_raw_subset.h5ad"),
        out.path(),
        &["--dgcmatrix"],
    );
    assert_eq!(slot(out.path(), "counts"), counts(4, 0.5));
}

/// `scx convert` at info level; its log, which names copied matrices.
fn convert_log(input: &Path, out: &Path, extra: &[&str]) -> String {
    let res = Command::new(env!("CARGO_BIN_EXE_scx"))
        .env("RUST_LOG", "info")
        .arg("convert")
        .arg(input)
        .arg(out)
        .args(extra)
        .output()
        .expect("failed to run scx convert");
    // The tracing log goes to stdout; errors to stderr.
    let log = String::from_utf8_lossy(&res.stdout).into_owned();
    assert!(
        res.status.success(),
        "scx convert failed:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
    log
}

/// gzip, f32, sorted rows: --dgcmatrix copies X and raw.X as stored.
#[test]
fn dgcmatrix_copies_gzip_x_and_raw_as_stored() {
    let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let log = convert_log(&fixture("tiny_raw_gzip.h5ad"), out.path(), &["--dgcmatrix"]);
    assert!(log.contains("copied raw/X into 'counts'"), "{log}");
    assert!(log.contains("copied X into 'data'"), "{log}");
    assert_eq!(slot(out.path(), "counts"), counts(5, 1.0));
    assert_eq!(slot(out.path(), "data"), counts(5, 0.5));
}

/// Unsorted rows can't be copied (dgCMatrix needs them sorted): they stream.
#[test]
fn dgcmatrix_streams_unsorted_rows() {
    let out = tempfile::NamedTempFile::with_suffix(".h5seurat").unwrap();
    let log = convert_log(
        &fixture("tiny_raw_unsorted_gzip.h5ad"),
        out.path(),
        &["--dgcmatrix"],
    );
    assert!(!log.contains("copied"), "{log}");
    assert_eq!(slot(out.path(), "counts"), counts(5, 1.0));
    assert_eq!(slot(out.path(), "data"), counts(5, 0.5));
}

/// --output-hash picks the sidecar's `output.hash` algorithm (blake3 by
/// default) or leaves it out; the source hash stays blake3 (its default).
#[test]
fn output_hash_flag_picks_sidecar_hash() {
    let dir = tempfile::tempdir().unwrap();
    for (extra, want) in [
        (&[][..], Some("blake3:")),
        (&["--output-hash", "sha256"][..], Some("sha256:")),
        (&["--output-hash", "none"][..], None),
    ] {
        let out = dir.path().join("o.h5seurat");
        let _ = std::fs::remove_file(&out);
        convert(&fixture("tiny_raw.h5ad"), &out, extra);
        let side = std::fs::read_to_string(dir.path().join("o.h5seurat.prov.json")).unwrap();
        let side: serde_json::Value = serde_json::from_str(&side).unwrap();
        let got = side["output"].get("hash").and_then(|h| h.as_str());
        assert_eq!(
            got.map(|h| &h[..h.find(':').unwrap() + 1]),
            want,
            "{extra:?}: {side}"
        );
        assert!(
            side["source"]["hash"]
                .as_str()
                .unwrap()
                .starts_with("blake3:"),
            "{extra:?}: {side}"
        );
        // The hash is of the output file's bytes, in lowercase hex.
        let want_hash = match want {
            Some("blake3:") => Some(format!(
                "blake3:{}",
                blake3::hash(&std::fs::read(&out).unwrap()).to_hex()
            )),
            Some(_) => Some(format!(
                "sha256:{}",
                scx_core::provenance::sha256_file(&out).unwrap()
            )),
            None => None,
        };
        assert_eq!(got.map(str::to_string), want_hash, "{extra:?}");
    }
}

/// uns["scx_provenance"] records only the source's file name, so the same
/// source converted from two different directories gives identical bytes.
#[test]
fn source_directory_does_not_change_the_output() {
    let dir = tempfile::tempdir().unwrap();
    let mut hashes = Vec::new();
    for sub in ["a", "b/deeper"] {
        let d = dir.path().join(sub);
        std::fs::create_dir_all(&d).unwrap();
        let src = d.join("tiny_raw.h5ad");
        std::fs::copy(fixture("tiny_raw.h5ad"), &src).unwrap();
        let out = d.join("o.h5ad");
        convert(&src, &out, &[]);
        let side = std::fs::read_to_string(d.join("o.h5ad.prov.json")).unwrap();
        let side: serde_json::Value = serde_json::from_str(&side).unwrap();
        hashes.push(side["output"]["hash"].as_str().unwrap().to_string());
    }
    assert_eq!(hashes[0], hashes[1]);
}

/// Input that no reader recognises fails with an error naming the URL or
/// stdin, not the temporary copy.
#[test]
fn unrecognised_url_and_stdin_input_name_the_source() {
    let dir = tempfile::tempdir().unwrap();
    let junk = dir.path().join("junk.bin");
    std::fs::write(&junk, b"not a single-cell file").unwrap();
    let out = dir.path().join("out.h5ad");
    let run = |input: &str| {
        Command::new(env!("CARGO_BIN_EXE_scx"))
            .args(["convert", input])
            .arg(&out)
            .stdin(std::fs::File::open(&junk).unwrap())
            .output()
            .unwrap()
    };
    let res = run("-");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(!res.status.success());
    assert!(
        err.contains("cannot open stdin") && err.contains("MatrixMarket"),
        "{err}"
    );
    #[cfg(feature = "net")]
    {
        let url = format!("file://{}", junk.display());
        let res = run(&url);
        let err = String::from_utf8_lossy(&res.stderr);
        assert!(!res.status.success());
        assert!(
            err.contains(&format!("could not detect the format of '{url}'")),
            "{err}"
        );
    }
}

/// A URL input other than Parquet is downloaded, detected by content and
/// converted like the local file: `scx diff` finds nothing (it skips the
/// provenance, which names the URL) and the source hash is the same.
#[cfg(feature = "net")]
#[test]
fn url_input_converts_like_the_local_file() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["tiny_raw.h5ad", "tiny_v5.h5seurat", "tiny_10x.h5"] {
        let run = |input: &str, out: &str| {
            let out = dir.path().join(out);
            let _ = std::fs::remove_file(&out);
            convert(Path::new(input), &out, &[]);
            let side = std::fs::read_to_string(format!("{}.prov.json", out.display())).unwrap();
            let side: serde_json::Value = serde_json::from_str(&side).unwrap();
            (out, side["source"]["hash"].clone())
        };
        let (local, local_hash) = run(fixture(name).to_str().unwrap(), "local.h5ad");
        let url = format!("file://{}", fixture(name).display());
        let (remote, remote_hash) = run(&url, "remote.h5ad");
        assert!(local_hash.is_string(), "{name}");
        assert_eq!(remote_hash, local_hash, "{name}");
        let diff = Command::new(env!("CARGO_BIN_EXE_scx"))
            .arg("diff")
            .args([&local, &remote])
            .output()
            .unwrap();
        assert!(
            diff.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&diff.stdout)
        );
    }
}

/// `scx convert -` reads stdin: piped in, a file converts as it does from
/// its path (`scx diff` finds nothing) and records the same source hash.
#[test]
fn stdin_input_converts_like_the_file() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["tiny_raw.h5ad", "tiny_v5.h5seurat", "tiny_v5.rds"] {
        let local = dir.path().join("local.h5ad");
        let piped = dir.path().join("piped.h5ad");
        for p in [&local, &piped] {
            let _ = std::fs::remove_file(p);
        }
        convert(&fixture(name), &local, &["--dtype", "f64"]);
        let res = Command::new(env!("CARGO_BIN_EXE_scx"))
            .args(["convert", "-"])
            .arg(&piped)
            .args(["--dtype", "f64"])
            .stdin(std::fs::File::open(fixture(name)).unwrap())
            .output()
            .unwrap();
        assert!(
            res.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&res.stderr)
        );
        let hash = |out: &Path| {
            let side = std::fs::read_to_string(format!("{}.prov.json", out.display())).unwrap();
            serde_json::from_str::<serde_json::Value>(&side).unwrap()["source"]["hash"].clone()
        };
        assert!(hash(&local).is_string(), "{name}");
        assert_eq!(hash(&piped), hash(&local), "{name}");
        let diff = Command::new(env!("CARGO_BIN_EXE_scx"))
            .arg("diff")
            .args([&local, &piped])
            .output()
            .unwrap();
        assert!(
            diff.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&diff.stdout)
        );
    }
}

/// raw.X of the AnnData at `path` (h5ad or Zarr), dense, cells x genes;
/// `None` when the reader finds no raw beside X.
fn raw(path: &Path) -> Option<Vec<Vec<f64>>> {
    futures::executor::block_on(async {
        let opts = scx_core::OpenOptions {
            source_hash: Default::default(),
            ..scx_core::OpenOptions::new(100)
        };
        let mut r = scx_core::open(path.to_str().unwrap(), &opts).await.unwrap();
        let meta = r.raw_meta().await.unwrap()?;
        let mut out = vec![vec![0.0; meta.shape.1]; meta.shape.0];
        let mut stream = r.raw_stream(&meta, 4);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            let values = chunk.data.data.to_f64();
            for row in 0..chunk.nrows {
                let span = chunk.data.indptr[row] as usize..chunk.data.indptr[row + 1] as usize;
                for k in span {
                    out[chunk.row_offset + row][chunk.data.indices[k] as usize] = values[k];
                }
            }
        }
        Some(out)
    })
}

/// Regression: h5ad -> h5ad and h5ad -> Zarr dropped `raw` without a word.
/// It now stays raw, values and genes intact, and reads back from both;
/// `--exclude raw` still drops it.
#[test]
fn h5ad_and_zarr_output_keep_raw() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture("tiny_raw.h5ad");
    let want = Some(counts(5, 1.0));
    let out = dir.path().join("out.h5ad");
    convert(&input, &out, &[]);
    assert_eq!(raw(&out), want, "h5ad");
    // The system-HDF5 CI job builds without Zarr.
    if cfg!(feature = "zarr") {
        let zarr = dir.path().join("out.zarr");
        convert(&input, &zarr, &[]);
        assert_eq!(raw(&zarr), want, "zarr");
        let back = dir.path().join("back.h5ad");
        convert(&zarr, &back, &[]);
        assert_eq!(raw(&back), want, "zarr -> h5ad");
    }

    let excluded = dir.path().join("excluded.h5ad");
    convert(&input, &excluded, &["--exclude", "raw"]);
    assert_eq!(raw(&excluded), None);
}
