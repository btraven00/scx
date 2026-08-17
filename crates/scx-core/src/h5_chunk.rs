//! Parallel inflate for deflate-chunked 1-D HDF5 datasets.
//!
//! libhdf5 decompresses a dataset's gzip chunks single-threaded, under the
//! hdf5-metno process-global lock — so a normal `read_slice` is decode-bound on
//! one core (the same wall anndata/h5py hit). For the CSR `X/data` and
//! `X/indices` arrays of an h5ad, each gzip chunk is independent, so we can read
//! the raw compressed chunks (cheap) and inflate them across all cores.
//!
//! This path is used only when it is provably equivalent to the normal read:
//! the dataset must be 1-D, chunked, and filtered by either deflate alone or
//! shuffle-then-deflate. Anything else (blosc, scale-offset, fletcher32, …)
//! falls back, because raw inflate would not reproduce the stored values.
//! Per-chunk we honour the filter mask: HDF5 may store an individual chunk
//! with a filter skipped when it didn't help.
//!
//! Shuffle matters in practice rather than in theory: `rhdf5` / `HDF5Array`
//! write `[Shuffle, Deflate]` by default, so every R-produced file — which is
//! most 10x files that did not come straight off a sequencer — used to miss
//! this path entirely and decode on one core.
//!
//! Safety: the raw `H5Dread_chunk` / `H5Dget_chunk_info_by_coord` FFI calls go
//! straight to `hdf5-sys`, bypassing hdf5-metno's wrappers. libhdf5 keeps global
//! state shared across every open file, so these calls must still be serialized
//! against all other HDF5 access in the process (e.g. concurrent readers on other
//! files) — we take the same process-global lock the safe API uses via
//! `hdf5::sync::sync` for the read stage. Only the parallel inflate stage, which
//! touches no HDF5 state, runs outside that lock.

use std::io::Read;

use flate2::read::ZlibDecoder;
use hdf5::filters::Filter;
use hdf5::Dataset;
use rayon::prelude::*;

use crate::error::{Result, ScxError};

/// How to decode one raw chunk of a dataset this module can handle.
#[derive(Debug, Clone, Copy)]
pub struct ChunkPlan {
    /// Chunk length in elements.
    pub chunk_len: usize,
    /// Filter-mask bit for deflate — its index in the pipeline.
    deflate_bit: u32,
    /// Filter-mask bit for shuffle, when the pipeline has one.
    shuffle_bit: Option<u32>,
}

/// Decode plan if `ds` is a 1-D chunked dataset this module can read directly;
/// otherwise `None`, and the caller must use the normal HDF5 read path.
///
/// Filters are applied in pipeline order on write and unwound in reverse on
/// read, so `[Shuffle, Deflate]` means inflate first, then un-shuffle. The
/// filter-mask bit for each filter is its index in that pipeline.
pub fn chunk_plan(ds: &Dataset) -> Option<ChunkPlan> {
    let (deflate_bit, shuffle_bit) = match ds.filters().as_slice() {
        [Filter::Deflate(_)] => (0, None),
        [Filter::Shuffle, Filter::Deflate(_)] => (1, Some(0)),
        _ => return None,
    };
    match ds.chunk() {
        Some(dims) if dims.len() == 1 => Some(ChunkPlan {
            chunk_len: dims[0],
            deflate_bit,
            shuffle_bit,
        }),
        _ => None,
    }
}

/// Invert the HDF5 shuffle filter.
///
/// Shuffle is a byte transpose: for `n` elements of `e` bytes it emits all the
/// first bytes, then all the second bytes, and so on. Mirrors `H5Zshuffle.c`
/// exactly, including its `e > 1 && n > 1` no-op guard and the `nbytes % e`
/// tail that the filter copies through untouched — get either wrong and the
/// values are silently garbage rather than an error.
fn unshuffle(src: &[u8], elem_size: usize) -> Vec<u8> {
    let nbytes = src.len();
    let n = nbytes / elem_size.max(1);
    if elem_size <= 1 || n <= 1 {
        return src.to_vec();
    }
    let mut out = vec![0u8; nbytes];
    // Gather rather than scatter: writes stay sequential, reads stride by n
    // within a chunk that is L2-resident at any realistic chunk_len.
    for j in 0..n {
        let dst = &mut out[j * elem_size..(j + 1) * elem_size];
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = src[i * n + j];
        }
    }
    let tail = n * elem_size;
    if tail < nbytes {
        out[tail..].copy_from_slice(&src[tail..]);
    }
    out
}

/// Inflate elements `[a, b)` of a 1-D deflate-chunked dataset in parallel and
/// return their raw little-endian bytes (`(b - a) * elem_size` of them).
///
/// `elem_size` must equal the dataset's stored element size; the caller
/// reinterprets the bytes. `plan` comes from [`chunk_plan`].
pub fn read_range_parallel(
    ds: &Dataset,
    a: usize,
    b: usize,
    elem_size: usize,
    plan: ChunkPlan,
) -> Result<Vec<u8>> {
    if b <= a {
        return Ok(Vec::new());
    }
    let chunk_len = plan.chunk_len;
    let dsid = ds.id();
    let ka = a / chunk_len;
    let kb = (b - 1) / chunk_len;

    // 1) Read the raw (still-filtered) chunks. These are raw `hdf5-sys` FFI calls,
    //    so they do NOT go through hdf5-metno's wrappers and would otherwise race
    //    with any other thread in the process that is inside libhdf5 (HDF5's
    //    global state is shared across all open files, not per-file). Hold the
    //    same process-global lock the safe API uses for the whole read stage; the
    //    parallel inflate below touches no HDF5 state and stays outside the lock.
    let raw: Vec<(usize, u32, Vec<u8>)> = hdf5::sync::sync(|| {
        let mut raw: Vec<(usize, u32, Vec<u8>)> = Vec::with_capacity(kb - ka + 1);
        for k in ka..=kb {
            let off = (k * chunk_len) as hdf5_sys::h5::hsize_t;
            let mut filter_mask: u32 = 0;
            let mut addr: hdf5_sys::h5::haddr_t = 0;
            let mut size: hdf5_sys::h5::hsize_t = 0;
            let rc = unsafe {
                hdf5_sys::h5d::H5Dget_chunk_info_by_coord(
                    dsid,
                    &off,
                    &mut filter_mask,
                    &mut addr,
                    &mut size,
                )
            };
            if rc < 0 || size == 0 {
                return Err(ScxError::InvalidFormat(format!(
                    "H5Dget_chunk_info_by_coord failed at element {}",
                    k * chunk_len
                )));
            }
            let mut buf = vec![0u8; size as usize];
            // The raw chunk-read symbol is versioned by the HDF5 ABI we link
            // against:
            //   - HDF5 < 2.0 exports the 5-arg `H5Dread_chunk` and
            //     hdf5-metno-sys binds it as `H5Dread_chunk1` (via a link_name
            //     override).
            //   - HDF5 >= 2.0 drops the deprecated `H5Dread_chunk1` symbol
            //     entirely and exports only the 6-arg `H5Dread_chunk2` (extra
            //     in/out `buf_size`); hdf5-metno-sys only declares the
            //     `H5Dread_chunk2` binding when it detects that version.
            // Gate on the actual linked version (the `hdf5_2_0` cfg set by
            // build.rs from hdf5-metno-sys's `DEP_HDF5_VERSION_2_0_0` metadata),
            // NOT on our `vendored-hdf5` feature — the feature only selects
            // static-vs-system linking and a system HDF5 can be either ABI.
            // `buf` is already sized exactly from H5Dget_chunk_info_by_coord
            // above, so the chunk fits without resizing.
            let rc = unsafe {
                #[cfg(hdf5_2_0)]
                {
                    let mut buf_size = buf.len();
                    hdf5_sys::h5d::H5Dread_chunk2(
                        dsid,
                        0, // H5P_DEFAULT
                        &off,
                        &mut filter_mask,
                        buf.as_mut_ptr().cast(),
                        &mut buf_size,
                    )
                }
                #[cfg(not(hdf5_2_0))]
                {
                    hdf5_sys::h5d::H5Dread_chunk1(
                        dsid,
                        0, // H5P_DEFAULT
                        &off,
                        &mut filter_mask,
                        buf.as_mut_ptr().cast(),
                    )
                }
            };
            if rc < 0 {
                return Err(ScxError::InvalidFormat(format!(
                    "H5Dread_chunk failed at element {}",
                    k * chunk_len
                )));
            }
            raw.push((k, filter_mask, buf));
        }
        Ok(raw)
    })?;

    // 2) Inflate, then un-shuffle, across cores. A set bit in the filter mask
    //    means that filter was skipped for that chunk, so each stage is applied
    //    only when its own bit is clear.
    let inflated: Vec<(usize, Vec<u8>)> = raw
        .into_par_iter()
        .map(|(k, filter_mask, cbuf)| -> Result<(usize, Vec<u8>)> {
            let bytes = if filter_mask & (1 << plan.deflate_bit) != 0 {
                cbuf
            } else {
                let mut d = ZlibDecoder::new(&cbuf[..]);
                let mut out = Vec::new();
                d.read_to_end(&mut out)
                    .map_err(|e| ScxError::InvalidFormat(format!("inflate failed: {e}")))?;
                out
            };
            let bytes = match plan.shuffle_bit {
                Some(bit) if filter_mask & (1 << bit) == 0 => unshuffle(&bytes, elem_size),
                _ => bytes,
            };
            Ok((k, bytes))
        })
        .collect::<Result<Vec<_>>>()?;

    // 3) Concatenate the requested byte sub-range (chunks are in ka..=kb order).
    let mut out = Vec::with_capacity((b - a) * elem_size);
    for (k, bytes) in &inflated {
        let chunk_first = k * chunk_len;
        let avail = bytes.len() / elem_size; // last dataset chunk may be short
        let lo = a.max(chunk_first);
        let hi = b.min(chunk_first + avail);
        if hi > lo {
            out.extend_from_slice(
                &bytes[(lo - chunk_first) * elem_size..(hi - chunk_first) * elem_size],
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scx_h5chunk_{name}_{}.h5", std::process::id()));
        p
    }

    /// f64 values with distinct bytes in every position — a byte-transpose that
    /// is off by one plane still produces plausible-looking floats, so the data
    /// has to be able to tell.
    fn values(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 1.000_000_123 + 0.5).collect()
    }

    fn write(path: &std::path::Path, vals: &[f64], shuffle: bool) {
        let f = File::create(path).unwrap();
        let mut b = f.new_dataset::<f64>().shape([vals.len()]).chunk([1024]);
        if shuffle {
            b = b.shuffle();
        }
        b.deflate(6).create("data").unwrap().write(vals).unwrap();
    }

    /// The whole point: shuffle+deflate must decode to exactly what libhdf5
    /// itself returns, for ranges that start and end mid-chunk.
    #[test]
    fn shuffle_deflate_matches_normal_read() {
        let path = tmp("shuffle");
        let vals = values(5000);
        write(&path, &vals, true);

        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        let plan = chunk_plan(&ds).expect("shuffle+deflate must be a supported plan");
        assert_eq!(plan.chunk_len, 1024);

        for (a, b) in [(0, 5000), (0, 1), (1023, 1025), (500, 3700), (4096, 5000)] {
            let raw = read_range_parallel(&ds, a, b, 8, plan).unwrap();
            let got: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(got, &vals[a..b], "range {a}..{b} decoded wrong");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Deflate alone must keep working — the filter-mask bit moves when shuffle
    /// is present, and getting that wrong breaks the pre-existing h5ad path.
    #[test]
    fn deflate_only_still_matches() {
        let path = tmp("deflate");
        let vals = values(3000);
        write(&path, &vals, false);

        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        let plan = chunk_plan(&ds).unwrap();
        let raw = read_range_parallel(&ds, 100, 2900, 8, plan).unwrap();
        let got: Vec<f64> = raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, &vals[100..2900]);
        let _ = std::fs::remove_file(&path);
    }

    /// Anything outside the two supported pipelines must fall back, not guess.
    #[test]
    fn unsupported_pipeline_declines() {
        let path = tmp("fletcher");
        {
            let f = File::create(&path).unwrap();
            f.new_dataset::<f64>()
                .shape([2048])
                .chunk([1024])
                .fletcher32()
                .deflate(6)
                .create("data")
                .unwrap()
                .write(&values(2048))
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        assert!(chunk_plan(&f.dataset("data").unwrap()).is_none());
        let _ = std::fs::remove_file(&path);
    }

    /// The transpose itself, against a hand-rolled shuffle.
    #[test]
    fn unshuffle_inverts_shuffle() {
        let e = 8usize;
        let n = 37usize;
        let orig: Vec<u8> = (0..n * e).map(|i| (i * 7 % 251) as u8).collect();
        let mut shuffled = vec![0u8; orig.len()];
        for i in 0..e {
            for j in 0..n {
                shuffled[i * n + j] = orig[j * e + i];
            }
        }
        assert_eq!(unshuffle(&shuffled, e), orig);
    }
}
