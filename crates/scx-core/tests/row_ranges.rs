//! `x_stream_rows` returns exactly the requested rows, equal to those rows of
//! the full matrix, for every reader; `reads_row_ranges()` says which readers
//! touch only those rows on disk. Uses the committed tiny fixtures.

use futures::StreamExt;
use scx_core::stream::{normalize_ranges, ChunkStream};
use std::ops::Range;
use std::path::PathBuf;

fn tiny(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

/// Each streamed row as (row index, dense values).
fn rows(stream: ChunkStream<'_>, n_cols: usize) -> Vec<(usize, Vec<f64>)> {
    futures::executor::block_on(stream.map(|c| c.unwrap()).collect::<Vec<_>>())
        .into_iter()
        .flat_map(|c| {
            let v = c.data.data.to_f64();
            (0..c.nrows)
                .map(|r| {
                    let mut row = vec![0.0; n_cols];
                    for k in c.data.indptr[r] as usize..c.data.indptr[r + 1] as usize {
                        row[c.data.indices[k] as usize] = v[k];
                    }
                    (c.row_offset + r, row)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn x_stream_rows_returns_exactly_the_requested_rows() {
    // (fixture, reads only the requested rows on disk)
    let cases = [
        ("tiny.h5ad", true),
        ("tiny_edge.zarr", true),
        ("tiny_v4.h5seurat", true),
        ("tiny_v5.h5seurat", true),
        ("tiny_10x.h5", true),
        // BPCells decodes the whole matrix on open, then slices it.
        ("tiny_v5_bpcells.h5seurat", false),
        // MatrixMarket is unordered text: a range scans the file.
        ("tiny_mtx", false),
    ];
    let range_sets: Vec<Vec<Range<usize>>> = vec![
        vec![1..3, 4..6],
        vec![5..6, 0..2], // unsorted
        vec![2..4, 3..5], // overlapping
        vec![2..2, 3..4], // with an empty range
        vec![0..6],
        vec![],
    ];
    for (name, bounded) in cases {
        if name.ends_with(".zarr") && !cfg!(feature = "zarr") {
            continue;
        }
        for chunk in [1, 4] {
            let opts = scx_core::OpenOptions::new(chunk);
            let path = tiny(name);
            let mut r = scx_core::open(path.to_str().unwrap(), &opts)
                .await
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(r.reads_row_ranges(), bounded, "{name}: reads_row_ranges");
            let (n_obs, n_vars) = r.shape();
            assert_eq!(n_obs, 6, "{name}");
            let full: Vec<Vec<f64>> = rows(r.x_stream(), n_vars)
                .into_iter()
                .map(|(_, v)| v)
                .collect();
            for ranges in &range_sets {
                let want: Vec<usize> = normalize_ranges(ranges.clone())
                    .into_iter()
                    .flatten()
                    .collect();
                let got = rows(r.x_stream_rows(ranges.clone()), n_vars);
                let ctx = format!("{name}, chunk {chunk}, ranges {ranges:?}");
                assert_eq!(
                    got.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                    want,
                    "{ctx}: rows"
                );
                for (i, v) in &got {
                    assert_eq!(v, &full[*i], "{ctx}: row {i}");
                }
            }
            // A range past the end is an error, not a short read.
            let err = futures::executor::block_on(
                r.x_stream_rows(std::iter::once(4..7).collect()).next(),
            );
            assert!(
                matches!(err, Some(Err(_))),
                "{name}: out-of-range not refused"
            );
        }
    }
}
