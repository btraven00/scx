//! Parallel inflate (and deflate, see [`DeflateAppender`]) for deflate-chunked
//! 1-D HDF5 datasets.
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

use std::borrow::Cow;
use std::io::{Read, Write};

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use flate2::{Decompress, FlushDecompress, Status};
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

/// Filter mask, file address and stored size of the chunk that starts at
/// element `start`. Raw `hdf5-sys` FFI: the caller must hold `hdf5::sync::sync`.
fn chunk_location(dsid: hdf5_sys::h5i::hid_t, start: usize) -> Result<(u32, u64, usize)> {
    let off = start as hdf5_sys::h5::hsize_t;
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
            start
        )));
    }
    Ok((filter_mask, addr, size as usize))
}

/// Read the stored (still-filtered) chunk that starts at element `start`, and
/// its filter mask. Raw `hdf5-sys` FFI: the caller must hold `hdf5::sync::sync`.
fn read_raw_chunk(dsid: hdf5_sys::h5i::hid_t, start: usize) -> Result<(u32, Vec<u8>)> {
    let off = start as hdf5_sys::h5::hsize_t;
    let (mut filter_mask, _, size) = chunk_location(dsid, start)?;
    let mut buf = vec![0u8; size];
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
            start
        )));
    }
    Ok((filter_mask, buf))
}

/// Sees a chunk's values, as raw little-endian bytes; false stops a copy.
pub type ChunkCheck<'a> = &'a mut dyn FnMut(&[u8]) -> bool;

/// Copy the 1-D chunked dataset `src` to `dst/name` as stored: same type,
/// chunking and filters, each chunk moved still compressed. The new dataset
/// carries no timestamp, unlike an `H5Ocopy`, which keeps the source's.
///
/// `check`, if given, sees each chunk's values in order, as raw little-endian
/// bytes: unfiltered chunks as stored, deflate ones (optionally after shuffle)
/// inflated in parallel a batch at a time. Once it returns false the copy
/// stops and this returns `Ok(false)`, leaving `dst/name` partly written for
/// the caller to unlink. Any other filter is an error when `check` is given.
///
/// Chunk bytes are read with `pread` at the address HDF5 reports, one lookup
/// per chunk instead of `H5Dread_chunk`'s two; that address is only the file
/// offset without a userblock, so a file with one is an error.
pub fn copy_chunks(
    src: &Dataset,
    dst: &hdf5::Group,
    name: &str,
    mut check: Option<ChunkCheck>,
) -> Result<bool> {
    const BATCH: usize = 256;

    let Some(chunk) = src.chunk().filter(|c| c.len() == 1).map(|c| c[0]) else {
        return Err(ScxError::InvalidFormat(format!(
            "{}: not a 1-D chunked dataset",
            src.name()
        )));
    };
    if src.file()?.fcpl()?.userblock() != 0 {
        return Err(ScxError::InvalidFormat(format!(
            "{}: chunk copy does not support a userblock",
            src.filename()
        )));
    }
    let filters = src.filters();
    let plan = chunk_plan(src);
    if check.is_some() && !filters.is_empty() && plan.is_none() {
        return Err(ScxError::InvalidFormat(format!(
            "{}: cannot decode {filters:?} to check it",
            src.name()
        )));
    }
    let dtype = src.dtype()?;
    let elem = dtype.size();
    let n = src.size();
    let out = dst
        .new_dataset_builder()
        .empty_as(&dtype)
        .chunk(chunk)
        .set_filters(&filters)
        .shape(n)
        .create(name)?;
    let file = std::fs::File::open(src.filename())?;
    let did = out.id();
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    let all_locs = chunk_locations(src, chunk, n)?;
    for (batch, locs) in starts.chunks(BATCH).zip(all_locs.chunks(BATCH)) {
        let raw = locs
            .iter()
            .map(|&(_, addr, size)| {
                let mut buf = vec![0u8; size];
                read_at(&file, &mut buf, addr)?;
                Ok(buf)
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(check) = check.as_mut() {
            let decoded = plan
                .map(|p| {
                    raw.par_iter()
                        .zip(locs)
                        .map(|(b, &(mask, ..))| decode_chunk(p, mask, b, elem))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            for (i, &start) in batch.iter().enumerate() {
                let bytes = decoded.as_ref().map_or(&raw[i], |d| &d[i]);
                // The last chunk is stored full size; only its first part is data.
                let valid = (chunk.min(n - start) * elem).min(bytes.len());
                if !check(&bytes[..valid]) {
                    return Ok(false);
                }
            }
        }
        hdf5::sync::sync(|| {
            for ((&start, &(mask, ..)), buf) in batch.iter().zip(locs).zip(&raw) {
                let off = start as hdf5_sys::h5::hsize_t;
                let rc = unsafe {
                    hdf5_sys::h5d::H5Dwrite_chunk(
                        did,
                        0,
                        mask,
                        &off,
                        buf.len(),
                        buf.as_ptr().cast(),
                    )
                };
                if rc < 0 {
                    return Err(ScxError::InvalidFormat(format!(
                        "H5Dwrite_chunk failed at element {start} of {}",
                        src.name()
                    )));
                }
            }
            Ok(())
        })?;
    }
    Ok(true)
}

/// (filter mask, file address, stored size) of each chunk of the 1-D dataset
/// `ds`, in element order. HDF5 1.14 lists them in one pass; older versions
/// look each up, about 50 us a chunk.
fn chunk_locations(ds: &Dataset, chunk: usize, n: usize) -> Result<Vec<(u32, u64, usize)>> {
    #[cfg(hdf5_1_14)]
    {
        let mut locs = vec![None; n.div_ceil(chunk)];
        ds.chunks_visit(|c| {
            if let Some(slot) = locs.get_mut(c.offset[0] as usize / chunk) {
                *slot = Some((c.filter_mask, c.addr, c.size as usize));
            }
            0
        })?;
        locs.into_iter()
            .enumerate()
            .map(|(k, loc)| {
                loc.filter(|l| l.2 > 0).ok_or_else(|| {
                    ScxError::InvalidFormat(format!(
                        "{}: chunk at element {} is not stored",
                        ds.name(),
                        k * chunk
                    ))
                })
            })
            .collect()
    }
    #[cfg(not(hdf5_1_14))]
    {
        let id = ds.id();
        hdf5::sync::sync(|| {
            (0..n)
                .step_by(chunk)
                .map(|s| chunk_location(id, s))
                .collect()
        })
    }
}

/// Fill `buf` from `file` at byte `offset`, leaving the file position alone.
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        let mut done = 0;
        while done < buf.len() {
            match std::os::windows::fs::FileExt::seek_read(
                file,
                &mut buf[done..],
                offset + done as u64,
            )? {
                0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                k => done += k,
            }
        }
        Ok(())
    }
}

/// The values (raw little-endian bytes) of one stored chunk: inflate, then
/// un-shuffle. A set bit in the filter mask means that filter was skipped for
/// this chunk, so each stage is applied only when its own bit is clear.
fn decode_chunk(
    plan: ChunkPlan,
    filter_mask: u32,
    cbuf: &[u8],
    elem_size: usize,
) -> Result<Vec<u8>> {
    let bytes = if filter_mask & (1 << plan.deflate_bit) != 0 {
        cbuf.to_vec()
    } else {
        let mut out = Vec::new();
        ZlibDecoder::new(cbuf)
            .read_to_end(&mut out)
            .map_err(|e| ScxError::InvalidFormat(format!("inflate failed: {e}")))?;
        out
    };
    Ok(match plan.shuffle_bit {
        Some(bit) if filter_mask & (1 << bit) == 0 => unshuffle(&bytes, elem_size),
        _ => bytes,
    })
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
    let mut out = vec![0u8; b.saturating_sub(a) * elem_size];
    read_range_into(ds, a, elem_size, plan, &mut out)?;
    Ok(out)
}

/// [`read_range_parallel`] into a typed `Vec`, decoded in place: no byte
/// buffer to copy out of. `T` must have the dataset's stored element size;
/// values are taken bit for bit, so signedness may differ.
pub(crate) fn read_range_typed<T: bytemuck::Pod>(
    ds: &Dataset,
    a: usize,
    b: usize,
    plan: ChunkPlan,
) -> Result<Vec<T>> {
    // ponytail: assumes a little-endian host, as `TypedVec::from_le_bytes` did.
    let mut out = zeroed_vec_huge::<T>(b.saturating_sub(a));
    read_range_into(
        ds,
        a,
        size_of::<T>(),
        plan,
        bytemuck::cast_slice_mut(&mut out),
    )?;
    Ok(out)
}

/// `n` zeroed values. On Linux a block of 4 MB or more is madvised to 2 MB
/// pages first, as numpy does for its arrays: with THP in `madvise` mode,
/// filling it then costs a page fault per 2 MB instead of per 4 KB.
///
/// zeroed_vec is calloc, so a large block arrives as fresh untouched pages
/// and the advice applies to every one of them.
pub(crate) fn zeroed_vec_huge<T: bytemuck::Pod>(n: usize) -> Vec<T> {
    let v = bytemuck::zeroed_vec::<T>(n);
    #[cfg(target_os = "linux")]
    {
        const PAGE: usize = 4096;
        let bytes = n * size_of::<T>();
        if bytes >= 4 << 20 {
            // madvise only reads the range, so a shared pointer will do; `v`
            // stays immutable and other targets see no unused `mut`.
            let p = v.as_ptr() as usize;
            let start = p.next_multiple_of(PAGE);
            let end = (p + bytes) / PAGE * PAGE;
            // A hint: on failure the pages are just small.
            unsafe { libc::madvise(start as *mut _, end - start, libc::MADV_HUGEPAGE) };
        }
    }
    v
}

/// Decode elements `[a, a + out.len() / elem_size)` into `out`.
///
/// Each chunk owns a disjoint slice of `out`. A chunk the range covers whole,
/// deflated and not shuffled, inflates straight into its slice. The rest (the
/// range's first and last chunks when partial, the dataset's padded edge
/// chunk, shuffled or deflate-skipped chunks) decode to a temporary and copy
/// their part across.
fn read_range_into(
    ds: &Dataset,
    a: usize,
    elem_size: usize,
    plan: ChunkPlan,
    out: &mut [u8],
) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let b = a + out.len() / elem_size;
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
    let raw: Vec<(u32, Vec<u8>)> = hdf5::sync::sync(|| {
        (ka..=kb)
            .map(|k| read_raw_chunk(dsid, k * chunk_len))
            .collect::<Result<_>>()
    })?;

    // 2) Split `out` into each chunk's slice, in ka..=kb order.
    let mut slices = Vec::with_capacity(raw.len());
    let mut rest = out;
    for k in ka..=kb {
        let lo = a.max(k * chunk_len);
        let hi = b.min((k + 1) * chunk_len);
        let (head, tail) = rest.split_at_mut((hi - lo) * elem_size);
        slices.push((k * chunk_len, lo, head));
        rest = tail;
    }

    // 3) Decode across cores. A set bit in the filter mask means that filter
    //    was skipped for that chunk, so each stage applies only when its own
    //    bit is clear.
    raw.into_par_iter()
        .zip(slices)
        .try_for_each(|((mask, cbuf), (first, lo, dst))| {
            let whole = dst.len() == chunk_len * elem_size;
            let deflated = mask & (1 << plan.deflate_bit) == 0;
            let shuffled = plan.shuffle_bit.is_some_and(|bit| mask & (1 << bit) == 0);
            if whole && deflated && !shuffled {
                return inflate_exact(&cbuf, dst);
            }
            let bytes = decode_chunk(plan, mask, &cbuf, elem_size)?;
            let from = (lo - first) * elem_size;
            let src = bytes.get(from..from + dst.len()).ok_or_else(|| {
                ScxError::InvalidFormat(format!(
                    "chunk at element {first} decodes to {} bytes, too few",
                    bytes.len()
                ))
            })?;
            dst.copy_from_slice(src);
            Ok(())
        })
}

/// Inflate the zlib stream `cbuf` into `dst`, which it must fill exactly.
fn inflate_exact(cbuf: &[u8], dst: &mut [u8]) -> Result<()> {
    let mut z = Decompress::new(true);
    let status = z
        .decompress(cbuf, dst, FlushDecompress::Finish)
        .map_err(|e| ScxError::InvalidFormat(format!("inflate failed: {e}")))?;
    if status != Status::StreamEnd || z.total_out() != dst.len() as u64 {
        return Err(ScxError::InvalidFormat(format!(
            "inflate gave {} bytes ({status:?}) for a {}-byte chunk",
            z.total_out(),
            dst.len()
        )));
    }
    Ok(())
}

/// Apply the HDF5 shuffle filter: the inverse of [`unshuffle`], with the same
/// `e > 1 && n > 1` guard and pass-through tail as `H5Zshuffle.c`.
fn shuffle(src: &[u8], elem_size: usize) -> Vec<u8> {
    let nbytes = src.len();
    let n = nbytes / elem_size.max(1);
    if elem_size <= 1 || n <= 1 {
        return src.to_vec();
    }
    let mut out = vec![0u8; nbytes];
    for (i, plane) in out.chunks_exact_mut(n).take(elem_size).enumerate() {
        for (j, slot) in plane.iter_mut().enumerate() {
            *slot = src[j * elem_size + i];
        }
    }
    let tail = n * elem_size;
    if tail < nbytes {
        out[tail..].copy_from_slice(&src[tail..]);
    }
    out
}

/// Appends to a resizable 1-D deflate-chunked dataset, compressing whole
/// chunks across cores and storing them with `H5Dwrite_chunk`.
///
/// libhdf5 deflates on one thread under its global lock, so a compressed
/// `write_slice` is compress-bound on one core. Here values are buffered until
/// whole chunks are available; each chunk is filtered (shuffle, then zlib at
/// the dataset's level — HDF5's deflate filter output is a zlib stream) in
/// parallel and written exactly once. [`finish`](Self::finish) writes the last
/// partial chunk zero-padded to full chunk size, which is how HDF5 itself
/// stores filtered edge chunks.
///
/// Like HDF5's optional deflate filter, a chunk that doesn't shrink is stored
/// raw with the deflate bit set in its filter mask.
///
/// `H5Dwrite_chunk` bypasses HDF5's chunk cache, so the dataset must be
/// written only through this appender: a `write_slice` to it as well could
/// leave a cached chunk that later overwrites one written here.
pub struct DeflateAppender {
    ds: Dataset,
    elem_size: usize,
    chunk_len: usize,
    level: u32,
    shuffle: bool,
    /// Bytes of the trailing partial chunk not yet written.
    pending: Vec<u8>,
    /// Elements written or pending; the dataset's current extent.
    len: usize,
}

impl DeflateAppender {
    /// An appender for `ds` if it is an empty 1-D chunked dataset filtered by
    /// deflate or shuffle+deflate, else `None` (use normal writes).
    pub fn new(ds: Dataset) -> Option<Self> {
        let (level, shuffle) = match ds.filters().as_slice() {
            [Filter::Deflate(l)] => (*l as u32, false),
            [Filter::Shuffle, Filter::Deflate(l)] => (*l as u32, true),
            _ => return None,
        };
        let chunk_len = match ds.chunk() {
            Some(dims) if dims.len() == 1 => dims[0],
            _ => return None,
        };
        if ds.shape() != [0] {
            return None;
        }
        let elem_size = ds.dtype().ok()?.size();
        Some(Self {
            ds,
            elem_size,
            chunk_len,
            level,
            shuffle,
            pending: Vec::new(),
            len: 0,
        })
    }

    /// Append `vals`, whose type must match the dataset's stored type.
    pub fn append<T: bytemuck::Pod>(&mut self, vals: &[T]) -> Result<()> {
        if size_of::<T>() != self.elem_size {
            return Err(ScxError::InvalidFormat(format!(
                "DeflateAppender: {}-byte values for a {}-byte dataset",
                size_of::<T>(),
                self.elem_size
            )));
        }
        if vals.is_empty() {
            return Ok(());
        }
        self.len += vals.len();
        self.ds.resize(self.len)?;
        let chunk_bytes = self.chunk_len * self.elem_size;
        let mut bytes: &[u8] = bytemuck::cast_slice(vals);
        // Top up the pending partial chunk first.
        if !self.pending.is_empty() {
            let take = (chunk_bytes - self.pending.len()).min(bytes.len());
            self.pending.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.pending.len() < chunk_bytes {
                return Ok(());
            }
            let first = (self.len - vals.len()) / self.chunk_len;
            let full = std::mem::take(&mut self.pending);
            self.write_chunks(first, &full)?;
        }
        // Whole chunks straight from the input; the remainder waits.
        let whole = bytes.len() / chunk_bytes * chunk_bytes;
        let first = (self.len - bytes.len() / self.elem_size) / self.chunk_len;
        self.write_chunks(first, &bytes[..whole])?;
        self.pending.extend_from_slice(&bytes[whole..]);
        Ok(())
    }

    /// Write the trailing partial chunk, if any.
    pub fn finish(mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut last = std::mem::take(&mut self.pending);
        last.resize(self.chunk_len * self.elem_size, 0);
        self.write_chunks(self.len / self.chunk_len, &last)
    }

    /// Filter and store whole chunks `bytes`, the first at chunk index `first`.
    fn write_chunks(&self, first: usize, bytes: &[u8]) -> Result<()> {
        let chunk_bytes = self.chunk_len * self.elem_size;
        let deflate_bit = u32::from(self.shuffle);
        let filtered: Vec<(u32, Vec<u8>)> = bytes
            .par_chunks(chunk_bytes)
            .map(|raw| {
                let src: Cow<[u8]> = if self.shuffle {
                    Cow::Owned(shuffle(raw, self.elem_size))
                } else {
                    Cow::Borrowed(raw)
                };
                let mut z = ZlibEncoder::new(
                    Vec::with_capacity(src.len() / 2),
                    Compression::new(self.level),
                );
                // Writing into a Vec can't fail.
                z.write_all(&src).expect("zlib into Vec");
                let out = z.finish().expect("zlib into Vec");
                if out.len() < src.len() {
                    (0, out)
                } else {
                    (1 << deflate_bit, src.into_owned())
                }
            })
            .collect();
        // Raw hdf5-sys FFI: hold the process-global HDF5 lock (see module docs).
        let dsid = self.ds.id();
        hdf5::sync::sync(|| {
            for (i, (mask, buf)) in filtered.iter().enumerate() {
                let off = ((first + i) * self.chunk_len) as hdf5_sys::h5::hsize_t;
                let rc = unsafe {
                    hdf5_sys::h5d::H5Dwrite_chunk(
                        dsid,
                        0, // H5P_DEFAULT
                        *mask,
                        &off,
                        buf.len(),
                        buf.as_ptr().cast(),
                    )
                };
                if rc < 0 {
                    return Err(ScxError::InvalidFormat(format!(
                        "H5Dwrite_chunk failed at element {off}"
                    )));
                }
            }
            Ok(())
        })
    }
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
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| f64::from_le_bytes(c))
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
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c))
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

    /// Append `vals` in uneven pieces through a `DeflateAppender` into a
    /// resizable dataset chunked at 1000, then read back with libhdf5's own
    /// filter pipeline: values, filters and chunking must all match.
    fn appender_roundtrip<T>(name: &str, vals: &[T], shuffle: bool)
    where
        T: hdf5::H5Type + bytemuck::Pod + PartialEq + std::fmt::Debug,
    {
        let path = tmp(name);
        {
            let f = File::create(&path).unwrap();
            let mut b = f.new_dataset::<T>().chunk(1000);
            if shuffle {
                b = b.shuffle();
            }
            let ds = b
                .deflate(4)
                .shape(hdf5::SimpleExtents::resizable([0usize]))
                .create("data")
                .unwrap();
            let mut app = DeflateAppender::new(ds).expect("deflate dataset");
            // Pieces below, at and across chunk boundaries.
            let mut rest = vals;
            for n in [1, 998, 1, 2500, 0, 1000].iter().cycle() {
                if rest.is_empty() {
                    break;
                }
                let (head, tail) = rest.split_at((*n).min(rest.len()));
                app.append(head).unwrap();
                rest = tail;
            }
            app.finish().unwrap();
        }
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.read_raw::<T>().unwrap(), vals);
        assert_eq!(ds.chunk(), Some(vec![1000]));
        let want = if shuffle {
            vec![Filter::Shuffle, Filter::Deflate(4)]
        } else {
            vec![Filter::Deflate(4)]
        };
        assert_eq!(ds.filters(), want);
        // The parallel read path agrees too.
        if !vals.is_empty() {
            let plan = chunk_plan(&ds).unwrap();
            let raw = read_range_parallel(&ds, 0, vals.len(), size_of::<T>(), plan).unwrap();
            assert_eq!(raw, bytemuck::cast_slice::<T, u8>(vals));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn appender_partial_last_chunk() {
        appender_roundtrip("app_partial", &values(4321), false);
    }

    #[test]
    fn appender_exact_multiple() {
        appender_roundtrip("app_exact", &values(5000), false);
    }

    #[test]
    fn appender_empty() {
        appender_roundtrip::<f64>("app_empty", &[], false);
    }

    #[test]
    fn appender_shuffle_deflate_i32() {
        let v: Vec<i32> = (0..7777).map(|i| i * 31 - 9000).collect();
        appender_roundtrip("app_shuffle", &v, true);
    }

    /// Random bytes don't deflate: those chunks must be stored raw with the
    /// deflate bit set in their filter mask, and still read back.
    #[test]
    fn appender_incompressible_chunk() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let v: Vec<u64> = (0..2500)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect();
        appender_roundtrip("app_random", &v, false);
    }

    /// In-place decode must give libhdf5's own bytes for every range shape:
    /// aligned whole chunks (inflated straight into the output), ranges that
    /// start and end mid-chunk, the padded edge chunk, empty ranges, and
    /// chunks stored raw with the deflate bit set, with and without shuffle.
    #[test]
    fn in_place_decode_matches_libhdf5() {
        // Compressible and random 1000-element runs, so some chunks are stored
        // deflated and others raw (deflate skipped).
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let vals: Vec<u64> = (0..5500u64)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                if (i / 1000) % 2 == 0 {
                    i
                } else {
                    x
                }
            })
            .collect();
        for shuffle in [false, true] {
            let path = tmp(&format!("inplace_{shuffle}"));
            {
                let f = File::create(&path).unwrap();
                let mut b = f.new_dataset::<u64>().chunk(1000);
                if shuffle {
                    b = b.shuffle();
                }
                let ds = b
                    .deflate(4)
                    .shape(hdf5::SimpleExtents::resizable([0usize]))
                    .create("data")
                    .unwrap();
                let mut app = DeflateAppender::new(ds).unwrap();
                app.append(&vals).unwrap();
                app.finish().unwrap();
            }
            let f = File::open(&path).unwrap();
            let ds = f.dataset("data").unwrap();
            let plan = chunk_plan(&ds).unwrap();
            let masks: Vec<u32> = chunk_locations(&ds, 1000, vals.len())
                .unwrap()
                .iter()
                .map(|l| l.0)
                .collect();
            let skip = 1 << u32::from(shuffle);
            assert!(
                masks.iter().any(|m| m & skip != 0),
                "no deflate-skipped chunk"
            );
            assert!(masks.iter().any(|m| m & skip == 0), "no deflated chunk");
            for (a, b) in [
                (0, 5500),
                (0, 1000),
                (1000, 3000),
                (999, 1001),
                (1500, 4321),
                (4000, 5500),
                (5000, 5500),
                (5499, 5500),
                (2000, 2000),
                (0, 0),
            ] {
                let want: Vec<u64> = ds.read_slice_1d(a..b).unwrap().to_vec();
                let got: Vec<u64> = read_range_typed(&ds, a, b, plan).unwrap();
                assert_eq!(got, want, "typed {a}..{b} shuffle={shuffle}");
                let bytes = read_range_parallel(&ds, a, b, 8, plan).unwrap();
                assert_eq!(
                    bytes,
                    bytemuck::cast_slice::<u64, u8>(&want),
                    "bytes {a}..{b}"
                );
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn shuffle_inverts_unshuffle() {
        let orig: Vec<u8> = (0..8 * 37 + 3).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(unshuffle(&shuffle(&orig, 8), 8), orig);
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
