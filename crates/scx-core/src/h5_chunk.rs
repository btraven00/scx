//! Parallel inflate for deflate-chunked 1-D HDF5 datasets.
//!
//! libhdf5 decompresses a dataset's gzip chunks single-threaded, under the
//! hdf5-metno process-global lock — so a normal `read_slice` is decode-bound on
//! one core (the same wall anndata/h5py hit). For the CSR `X/data` and
//! `X/indices` arrays of an h5ad, each gzip chunk is independent, so we can read
//! the raw compressed chunks (cheap) and inflate them across all cores.
//!
//! This path is used only when it is provably equivalent to the normal read:
//! the dataset must be 1-D, chunked, and filtered by deflate **alone** (a
//! shuffle/blosc/scaleoffset filter would make raw inflate wrong → fall back).
//! Per-chunk we honour the filter mask: HDF5 may store an individual chunk
//! uncompressed (deflate skipped) when compression didn't help.
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

/// Chunk length (in elements) if `ds` is a 1-D dataset filtered by deflate
/// alone; otherwise `None` (caller must use the normal HDF5 read path).
pub fn deflate_chunk_len(ds: &Dataset) -> Option<usize> {
    let filters = ds.filters();
    let only_deflate = filters.len() == 1 && matches!(filters[0], Filter::Deflate(_));
    if !only_deflate {
        return None;
    }
    match ds.chunk() {
        Some(dims) if dims.len() == 1 => Some(dims[0]),
        _ => None,
    }
}

/// Inflate elements `[a, b)` of a 1-D deflate-chunked dataset in parallel and
/// return their raw little-endian bytes (`(b - a) * elem_size` of them).
///
/// `elem_size` must equal the dataset's stored element size; the caller
/// reinterprets the bytes. `chunk_len` comes from [`deflate_chunk_len`].
pub fn read_range_parallel(
    ds: &Dataset,
    a: usize,
    b: usize,
    elem_size: usize,
    chunk_len: usize,
) -> Result<Vec<u8>> {
    if b <= a {
        return Ok(Vec::new());
    }
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
            //   - System libhdf5 (pre-2.0, the `--no-default-features` path)
            //     exports the 5-arg `H5Dread_chunk` and hdf5-metno-sys binds it
            //     as `H5Dread_chunk1` (via a link_name override).
            //   - The vendored/bundled HDF5 2.0.0 (the default `vendored-hdf5`
            //     path) drops the deprecated `H5Dread_chunk1` symbol entirely and
            //     exports only the 6-arg `H5Dread_chunk2` (extra in/out
            //     `buf_size`).
            // Gate on the same feature that selects the bundled library so each
            // build calls a symbol that actually exists. `buf` is already sized
            // exactly from H5Dget_chunk_info_by_coord above, so the chunk fits
            // without resizing.
            let rc = unsafe {
                #[cfg(feature = "vendored-hdf5")]
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
                #[cfg(not(feature = "vendored-hdf5"))]
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

    // 2) Inflate across cores. A set bit 0 in the filter mask means deflate was
    //    skipped for that chunk (stored raw) — use the bytes as-is.
    let inflated: Vec<(usize, Vec<u8>)> = raw
        .into_par_iter()
        .map(|(k, filter_mask, cbuf)| -> Result<(usize, Vec<u8>)> {
            if filter_mask & 1 != 0 {
                Ok((k, cbuf))
            } else {
                let mut d = ZlibDecoder::new(&cbuf[..]);
                let mut out = Vec::new();
                d.read_to_end(&mut out)
                    .map_err(|e| ScxError::InvalidFormat(format!("inflate failed: {e}")))?;
                Ok((k, out))
            }
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
