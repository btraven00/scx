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
//! Safety: the raw `H5Dread_chunk` FFI bypasses the hdf5-metno global lock. That
//! is sound here only because the caller (the `open_stream` reader thread) is
//! the single thread touching this file; the parallel stage (inflate) never
//! calls into HDF5.

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

    // 1) Read the raw (still-filtered) chunks. Serial + cheap; single-threaded
    //    HDF5 access, so bypassing the global lock is sound.
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
        // Use the 5-arg H5Dread_chunk1 (no data_size out-param): it's present in
        // every hdf5-metno-sys binding, whereas the 6-arg H5Dread_chunk2 only
        // exists with the bundled HDF5 2.0.0 feature, not the system library.
        // `buf` is already sized exactly from H5Dget_chunk_info_by_coord above.
        let rc = unsafe {
            hdf5_sys::h5d::H5Dread_chunk1(
                dsid,
                0, // H5P_DEFAULT
                &off,
                &mut filter_mask,
                buf.as_mut_ptr().cast(),
            )
        };
        if rc < 0 {
            return Err(ScxError::InvalidFormat(format!(
                "H5Dread_chunk failed at element {}",
                k * chunk_len
            )));
        }
        raw.push((k, filter_mask, buf));
    }

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
            out.extend_from_slice(&bytes[(lo - chunk_first) * elem_size..(hi - chunk_first) * elem_size]);
        }
    }
    Ok(out)
}
