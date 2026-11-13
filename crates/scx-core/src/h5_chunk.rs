//! Parallel reads (and deflate, see [`DeflateAppender`]) of chunked 1-D HDF5
//! datasets: unfiltered chunks are `pread` straight into place, deflate
//! chunks are inflated across cores.
//!
//! libhdf5 decompresses a dataset's gzip chunks single-threaded, under the
//! hdf5-metno process-global lock — so a normal `read_slice` is decode-bound on
//! one core (the same wall anndata/h5py hit). For the CSR `X/data` and
//! `X/indices` arrays of an h5ad, each gzip chunk is independent, so we can read
//! the raw compressed chunks (cheap) and inflate them across all cores.
//!
//! This path is used only when it is provably equivalent to the normal read:
//! the dataset must be 1-D, chunked, stored little-endian, and either
//! unfiltered or filtered by deflate alone or shuffle-then-deflate. Anything else (blosc, scale-offset, fletcher32, …)
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
//! `hdf5::sync::sync` for them. Reads look up chunk addresses under that lock
//! and then `pread` each chunk's bytes inside its parallel inflate task, which
//! touches no HDF5 state ([`ChunkReader`]).

use std::borrow::Cow;
use std::io::Write;

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
    /// Filter-mask bit for deflate — its index in the pipeline; `None` for
    /// an unfiltered dataset, whose chunks hold the values as stored.
    deflate_bit: Option<u32>,
    /// Filter-mask bit for shuffle, when the pipeline has one.
    shuffle_bit: Option<u32>,
}

impl ChunkPlan {
    /// A set bit in a chunk's filter mask means that filter was skipped for
    /// it, so a stage applies only when its own bit is clear.
    fn deflated(&self, mask: u32) -> bool {
        self.deflate_bit.is_some_and(|bit| mask & (1 << bit) == 0)
    }

    fn shuffled(&self, mask: u32) -> bool {
        self.shuffle_bit.is_some_and(|bit| mask & (1 << bit) == 0)
    }

    /// The chunk's stored bytes are its values, untouched by any filter.
    fn stored_raw(&self, mask: u32) -> bool {
        !self.deflated(mask) && !self.shuffled(mask)
    }
}

/// Decode plan if `ds` is a 1-D chunked dataset this module can read directly
/// (unfiltered, deflate or shuffle+deflate, stored little-endian); otherwise
/// `None`, and the caller must use the normal HDF5 read path.
///
/// Filters are applied in pipeline order on write and unwound in reverse on
/// read, so `[Shuffle, Deflate]` means inflate first, then un-shuffle. The
/// filter-mask bit for each filter is its index in that pipeline. An
/// unfiltered chunk is read straight into place with `pread`; `H5Dread` would
/// copy it through the chunk cache, one chunk at a time, under libhdf5's lock.
pub fn chunk_plan(ds: &Dataset) -> Option<ChunkPlan> {
    let (deflate_bit, shuffle_bit) = match ds.filters().as_slice() {
        [] => (None, None),
        [Filter::Deflate(_)] => (Some(0), None),
        [Filter::Shuffle, Filter::Deflate(_)] => (Some(1), Some(0)),
        _ => return None,
    };
    // Bytes are taken as stored, so only little-endian storage decodes here.
    if ds.dtype().ok()?.byte_order() == hdf5::datatype::ByteOrder::BigEndian {
        return None;
    }
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

/// Sees values of a dataset as raw little-endian bytes: `(index of the first
/// value, the element just before it or empty, the values)`. Called in
/// parallel once per chunk with an empty `prev`, then once per chunk
/// boundary with the next chunk's first element and the previous chunk's
/// last as `prev`. False fails the check.
pub type ChunkCheck<'a> = &'a (dyn Fn(usize, &[u8], &[u8]) -> bool + Sync);

const BATCH: usize = 64;

/// The chunk length of the 1-D chunked dataset `src`, or an error; and an
/// error if its file has a userblock: chunks are read with `pread` at the
/// address HDF5 reports, which is only the file offset without one.
fn chunk_len(src: &Dataset) -> Result<usize> {
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
    Ok(chunk)
}

/// `BATCH` chunks: the first element of each, their (filter mask, address,
/// stored size), their stored bytes.
type Batch<'a> = (&'a [usize], &'a [(u32, u64, usize)], Vec<Vec<u8>>);

/// Each stored chunk's bytes, `BATCH` chunks at a time.
fn stored_batches<'a>(
    src: &Dataset,
    starts: &'a [usize],
    locs: &'a [(u32, u64, usize)],
) -> Result<impl Iterator<Item = Result<Batch<'a>>> + 'a> {
    let file = std::fs::File::open(src.filename())?;
    Ok(starts
        .chunks(BATCH)
        .zip(locs.chunks(BATCH))
        .map(move |(batch, locs)| {
            let raw = locs
                .iter()
                .map(|&(_, addr, size)| {
                    let mut buf = vec![0u8; size];
                    read_at(&file, &mut buf, addr)?;
                    Ok(buf)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((batch, locs, raw))
        }))
}

/// Show `check` each chunk of the 1-D chunked dataset `src`, all chunks in
/// parallel, each read (`pread`) and inflated in its own task: unfiltered
/// chunks as stored, deflate ones (optionally after shuffle) inflated. Then
/// each chunk boundary. `Ok(false)` once a call fails. Any other filter is an
/// error. Touches no HDF5 state after listing the chunks, so it can run
/// beside a [`copy_chunks`].
pub fn check_chunks(src: &Dataset, check: ChunkCheck) -> Result<bool> {
    let chunk = chunk_len(src)?;
    let filters = src.filters();
    // Unfiltered chunks are checked as stored; a plan is for decoding.
    let plan = chunk_plan(src).filter(|_| !filters.is_empty());
    if !filters.is_empty() && plan.is_none() {
        return Err(ScxError::InvalidFormat(format!(
            "{}: cannot decode {filters:?} to check it",
            src.name()
        )));
    }
    let elem = src.dtype()?.size();
    let n = src.size();
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    let locs = chunk_locations(src, chunk, n)?;
    let file = std::fs::File::open(src.filename())?;
    // Each chunk's (first, last) element, or None once a chunk fails.
    let edges = starts
        .par_iter()
        .zip(&locs)
        .map(|(&start, &(mask, addr, size))| {
            let mut buf = vec![0u8; size];
            read_at(&file, &mut buf, addr)?;
            let vals = match plan {
                Some(p) => decode_chunk(p, mask, &buf, elem)?,
                None => buf,
            };
            // The last chunk is stored full size; only its first part is data.
            let v = &vals[..(chunk.min(n - start) * elem).min(vals.len())];
            if v.len() < elem || !check(start, &[], v) {
                return Ok(None);
            }
            Ok(Some((v[..elem].to_vec(), v[v.len() - elem..].to_vec())))
        })
        .collect::<Result<Option<Vec<_>>>>()?;
    let Some(edges) = edges else {
        return Ok(false);
    };
    Ok((1..edges.len()).all(|k| check(starts[k], &edges[k - 1].1, &edges[k].0)))
}

/// Copy the 1-D chunked dataset `src` to `dst/name` as stored: same type,
/// chunking and filters, each chunk moved still compressed. The new dataset
/// carries no timestamp, unlike an `H5Ocopy`, which keeps the source's.
/// Copying stored chunks this way follows scConvert
/// (<https://github.com/mianaz/scConvert>); this implementation is scx's own.
///
/// Once `stop` is set the copy stops between batches and this returns
/// `Ok(false)`, leaving `dst/name` partly written for the caller to unlink.
/// A thread writes each batch of chunks while the next batch is read.
pub fn copy_chunks(
    src: &Dataset,
    dst: &hdf5::Group,
    name: &str,
    stop: Option<&std::sync::atomic::AtomicBool>,
) -> Result<bool> {
    let chunk = chunk_len(src)?;
    let dtype = src.dtype()?;
    let n = src.size();
    // The source's extents, not a fixed `n`: anndata writes resizable
    // datasets, which may be smaller than one chunk (9 values in a 65536
    // chunk), and HDF5 refuses that chunk for a fixed-size dataset.
    let out = dst
        .new_dataset_builder()
        .empty_as(&dtype)
        .chunk(chunk)
        .set_filters(&src.filters())
        .shape(src.space()?.extents()?)
        .create(name)?;
    let did = out.id();
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    let all_locs = chunk_locations(src, chunk, n)?;
    let stopped = || stop.is_some_and(|s| s.load(std::sync::atomic::Ordering::Relaxed));
    let (tx, rx) = std::sync::mpsc::sync_channel::<Batch>(1);
    std::thread::scope(|s| {
        let writer = s.spawn(move || {
            for (batch, locs, raw) in rx {
                write_chunks(did, batch, locs, &raw, src)?;
            }
            Ok::<_, ScxError>(())
        });
        for item in stored_batches(src, &starts, &all_locs)? {
            if stopped() {
                return Ok(false);
            }
            if tx.send(item?).is_err() {
                break; // the writer failed; join reports why
            }
        }
        drop(tx);
        writer.join().expect("chunk writer panicked")?;
        Ok(!stopped())
    })
}

/// `H5Dwrite_chunk` each of a batch's stored chunks into the dataset `did`.
fn write_chunks(
    did: hdf5_sys::h5i::hid_t,
    batch: &[usize],
    locs: &[(u32, u64, usize)],
    raw: &[Vec<u8>],
    src: &Dataset,
) -> Result<()> {
    hdf5::sync::sync(|| {
        for ((&start, &(mask, ..)), buf) in batch.iter().zip(locs).zip(raw) {
            let off = start as hdf5_sys::h5::hsize_t;
            let rc = unsafe {
                hdf5_sys::h5d::H5Dwrite_chunk(did, 0, mask, &off, buf.len(), buf.as_ptr().cast())
            };
            if rc < 0 {
                return Err(ScxError::InvalidFormat(format!(
                    "H5Dwrite_chunk failed at element {start} of {}",
                    src.name()
                )));
            }
        }
        Ok(())
    })
}

/// Whether the 1-D chunked datasets `a` and `b` hold the same values, decided
/// without decoding either: same type, length, chunking and filters, and every
/// chunk stored with the same filter mask and the same (still compressed)
/// bytes. `false` when it can't tell (not 1-D chunked, a userblock), and for
/// equal values stored differently.
pub fn same_stored(a: &Dataset, b: &Dataset) -> Result<bool> {
    let chunk = match (a.chunk(), b.chunk()) {
        (Some(ca), Some(cb)) if ca == cb && ca.len() == 1 => ca[0],
        _ => return Ok(false),
    };
    let n = a.size();
    if n != b.size()
        || a.dtype()? != b.dtype()?
        || a.filters() != b.filters()
        || a.file()?.fcpl()?.userblock() != 0
        || b.file()?.fcpl()?.userblock() != 0
    {
        return Ok(false);
    }
    let (la, lb) = (chunk_locations(a, chunk, n)?, chunk_locations(b, chunk, n)?);
    if la.iter().zip(&lb).any(|(x, y)| (x.0, x.2) != (y.0, y.2)) {
        return Ok(false);
    }
    let (fa, fb) = (
        std::fs::File::open(a.filename())?,
        std::fs::File::open(b.filename())?,
    );
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    for (&(_, addr_a, size), &(_, addr_b, _)) in la.iter().zip(&lb) {
        ba.resize(size, 0);
        bb.resize(size, 0);
        read_at(&fa, &mut ba, addr_a)?;
        read_at(&fb, &mut bb, addr_b)?;
        if ba != bb {
            return Ok(false);
        }
    }
    Ok(true)
}

/// (filter mask, file address, stored size) of each chunk of the 1-D dataset
/// `ds`, in element order. HDF5 1.14 lists them in one pass; older versions
/// look each up, about 50 us a chunk.
fn chunk_locations(ds: &Dataset, chunk: usize, n: usize) -> Result<Vec<(u32, u64, usize)>> {
    #[cfg(hdf5_1_14)]
    {
        #[cfg(target_os = "linux")]
        if let Some((file, leaves, node)) = index_leaves(ds) {
            use std::os::fd::AsRawFd;
            for at in leaves {
                // A hint: on failure the walk below reads the node itself.
                unsafe {
                    libc::posix_fadvise(
                        file.as_raw_fd(),
                        at as i64,
                        node as i64,
                        libc::POSIX_FADV_WILLNEED,
                    )
                };
            }
        }
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

/// Where the leaf nodes of `ds`'s chunk index are: the file, each leaf's
/// address, and the node size. `None` whenever anything is not as expected.
///
/// libhdf5 walks a chunk index one node at a time, each a small read
/// somewhere else in the file. Cold, on a seeking disk or a network mount,
/// that is milliseconds per node and seconds per dataset, while the same
/// reads issued together take a tenth of the time. So [`chunk_locations`]
/// asks the kernel to read the leaves ahead of the walk. This only ever
/// produces read-ahead hints: libhdf5 still does the walk that counts.
///
/// Reads the two structures that lead to the leaves, as the HDF5 file format
/// specification lays them out: the dataset's object header, for
/// its version 3 chunked layout message (the root node's address), and the
/// version 1 B-tree's inner nodes. That is the index h5py, anndata and scx
/// write; files with the newer index types (`libver="latest"`) get no hint.
#[cfg(all(hdf5_1_14, target_os = "linux"))]
fn index_leaves(ds: &Dataset) -> Option<(std::fs::File, Vec<u64>, usize)> {
    let fcpl = ds.file().ok()?.fcpl().ok()?;
    let sizes = fcpl.sizes();
    if fcpl.userblock() != 0 || (sizes.sizeof_addr as usize, sizes.sizeof_size as usize) != (8, 8) {
        return None;
    }
    // The object header's address is the dataset's token in the native file
    // format (H5VLnative_token_to_addr copies it out the same way).
    let header = hdf5::sync::sync(|| unsafe {
        let mut info = std::mem::MaybeUninit::<hdf5_sys::h5o::H5O_info2_t>::zeroed();
        let rc =
            hdf5_sys::h5o::H5Oget_info3(ds.id(), info.as_mut_ptr(), hdf5_sys::h5o::H5O_INFO_BASIC);
        (rc >= 0).then(|| {
            let token: [u8; 16] = std::mem::transmute(info.assume_init().token);
            le_u64(&token, 0)
        })
    })??;
    let file = std::fs::File::open(ds.filename()).ok()?;
    let (root, rank) = index_root(&file, header)?;
    // A node holds up to 2K children and 2K + 1 keys. A key is the chunk's
    // size and filter mask (4 bytes each) and its offset along each dimension.
    let entry = 8 + 8 * rank + 8;
    let node = 24 + 2 * fcpl.istore_k() as usize * entry + entry - 8;
    if node > 1 << 20 {
        return None; // not a node size any writer uses
    }
    let (mut level, mut nodes) = index_node(&file, root, node, entry)?;
    if level == 0 {
        return Some((file, vec![root], node));
    }
    while level > 1 {
        level -= 1;
        nodes = nodes
            .par_iter()
            .map(|&at| {
                Some(
                    index_node(&file, at, node, entry)
                        .filter(|n| n.0 == level)?
                        .1,
                )
            })
            .collect::<Option<Vec<_>>>()?
            .concat();
    }
    Some((file, nodes, node))
}

/// The address of the root node of a chunk index and the rank of its keys
/// (the dataset's rank + 1), from the object header at `header`: version 1
/// (h5py's default) or version 2 ("OHDR", what hdf5-metno and so scx write).
#[cfg(all(hdf5_1_14, target_os = "linux"))]
fn index_root(file: &std::fs::File, header: u64) -> Option<(u64, usize)> {
    let mut pre = [0u8; 40];
    read_at(file, &mut pre, header).ok()?;
    // Where the first block of messages is, its size, and the size of a
    // message's own header.
    let (v2, first, head) = if pre[0] == 1 {
        (false, (header + 16, le_u64(&pre, 8)? & 0xffff_ffff), 8)
    } else if pre[..5] == *b"OHDR\x02" {
        // Flags say which optional fields precede the block's size (times,
        // attribute thresholds), how wide the size is, and whether each
        // message carries a creation order.
        let flags = pre[5] as usize;
        let at = 6 + (flags >> 5 & 1) * 16 + (flags >> 4 & 1) * 4;
        let width = 1 << (flags & 3);
        let mut len = [0u8; 8];
        len[..width].copy_from_slice(&pre[at..at + width]);
        let first = (header + (at + width) as u64, u64::from_le_bytes(len));
        (true, first, 4 + (flags >> 2 & 1) * 2)
    } else {
        return None;
    };
    // More blocks are named by continuation messages. A real header has a
    // handful.
    let mut blocks = vec![first];
    let mut k = 0;
    while let Some(&(at, len)) = blocks.get(k).filter(|b| k < 16 && b.1 <= 1 << 20) {
        k += 1;
        let mut buf = vec![0u8; len as usize];
        read_at(file, &mut buf, at).ok()?;
        let mut p = 0;
        while let Some(h) = buf.get(p..p + head) {
            let (ty, size) = if v2 {
                (h[0] as u16, u16::from_le_bytes([h[1], h[2]]))
            } else {
                (
                    u16::from_le_bytes([h[0], h[1]]),
                    u16::from_le_bytes([h[2], h[3]]),
                )
            };
            let Some(body) = buf.get(p + head..p + head + size as usize) else {
                break;
            };
            match ty {
                // Continuation: address and length of another block, which in
                // version 2 sits between "OCHK" and a checksum.
                0x10 => {
                    let (at, len) = (le_u64(body, 0)?, le_u64(body, 8)?);
                    blocks.push(if v2 {
                        (at + 4, len.checked_sub(8)?)
                    } else {
                        (at, len)
                    });
                }
                // Layout: version 3, class 2 (chunked), rank, address.
                0x08 if body.get(..2)? == [3, 2] => {
                    return Some((le_u64(body, 3)?, body[2] as usize));
                }
                0x08 => return None,
                _ => {}
            }
            p += head + size as usize;
        }
    }
    None
}

/// The level and children of the chunk index node at `at`: "TREE", node type
/// 1, level, entries used (2 bytes), two sibling addresses, then for each
/// entry a key and a child's address. Level 0 children are chunks.
#[cfg(all(hdf5_1_14, target_os = "linux"))]
fn index_node(file: &std::fs::File, at: u64, node: usize, entry: usize) -> Option<(u8, Vec<u64>)> {
    let mut buf = vec![0u8; node];
    read_at(file, &mut buf, at).ok()?;
    if &buf[..5] != b"TREE\x01" {
        return None;
    }
    let used = u16::from_le_bytes([buf[6], buf[7]]) as usize;
    let kids = (1..=used).map(|i| le_u64(&buf, 24 + i * entry - 8));
    Some((buf[5], kids.collect::<Option<_>>()?))
}

#[cfg(all(hdf5_1_14, target_os = "linux"))]
fn le_u64(buf: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(buf.get(at..at + 8)?.try_into().ok()?))
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
    let bytes = if !plan.deflated(filter_mask) {
        cbuf.to_vec()
    } else {
        // A chunk inflates to exactly its full size (the last one too), so
        // inflate in one call into a buffer of that size: ~12% faster than
        // growing one through `read_to_end`.
        let mut out = vec![0u8; plan.chunk_len * elem_size];
        let mut z = flate2::Decompress::new(true);
        let status = z
            .decompress(cbuf, &mut out, flate2::FlushDecompress::Finish)
            .map_err(|e| ScxError::InvalidFormat(format!("inflate failed: {e}")))?;
        if status != flate2::Status::StreamEnd || z.total_out() as usize != out.len() {
            return Err(ScxError::InvalidFormat(format!(
                "inflate failed: chunk gave {} bytes, expected {}",
                z.total_out(),
                out.len()
            )));
        }
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
    ChunkReader::with_plan(ds, plan, a == 0 && b >= ds.size())?
        .read_into(a, elem_size, &mut out)?;
    Ok(out)
}

/// (filter mask, file address, stored size) of one chunk.
type ChunkLoc = (u32, u64, usize);

/// Reads ranges of one deflate-chunked 1-D dataset, for as long as a stream
/// reads it: the dataset, a plain file handle and, for whole-dataset reads,
/// every chunk's location are looked up once, not once per block.
///
/// Chunk bytes are read with `pread` inside each inflate task, so reading and
/// inflating overlap across cores and only the location lookup takes the HDF5
/// lock. A file with a userblock (chunk addresses are not file offsets there)
/// reads chunks through `H5Dread_chunk` under the lock instead.
pub(crate) struct ChunkReader {
    ds: Dataset,
    plan: ChunkPlan,
    /// For `pread`; `None` when the file has a userblock.
    file: Option<std::fs::File>,
    /// Every chunk's location, listed up front when the whole dataset will be
    /// read. Otherwise each read looks up just its own chunks: listing them
    /// all costs time in the dataset's size, too much per block of a
    /// row-range read of an atlas.
    locs: Option<Vec<ChunkLoc>>,
}

impl ChunkReader {
    /// A reader for `ds`, or `None` when [`chunk_plan`] declines it.
    pub(crate) fn new(ds: &Dataset, whole: bool) -> Result<Option<Self>> {
        chunk_plan(ds)
            .map(|plan| Self::with_plan(ds, plan, whole))
            .transpose()
    }

    pub(crate) fn with_plan(ds: &Dataset, plan: ChunkPlan, whole: bool) -> Result<Self> {
        let file = if ds.file()?.fcpl()?.userblock() == 0 {
            Some(std::fs::File::open(ds.filename())?)
        } else {
            None
        };
        // An index that can't be listed (an unstored chunk, ...) just means
        // per-read lookups, which report the problem for the chunks read.
        let locs = (whole && file.is_some())
            .then(|| chunk_locations(ds, plan.chunk_len, ds.size()).ok())
            .flatten();
        Ok(Self {
            ds: ds.clone(),
            plan,
            file,
            locs,
        })
    }

    /// Decode elements `[a, a + out.len() / elem_size)` into `out`.
    /// `elem_size` must be the dataset's stored element size.
    ///
    /// Each chunk owns a disjoint slice of `out`. A chunk the range covers
    /// whole, deflated and not shuffled, inflates straight into its slice. The
    /// rest (the range's first and last chunks when partial, the dataset's
    /// padded edge chunk, shuffled or deflate-skipped chunks) decode to a
    /// temporary and copy their part across.
    pub(crate) fn read_into(&self, a: usize, elem_size: usize, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let b = a + out.len() / elem_size;
        let chunk_len = self.plan.chunk_len;
        let (ka, kb) = (a / chunk_len, (b - 1) / chunk_len);

        // Each chunk's slice of `out`, in ka..=kb order.
        let mut slices = Vec::with_capacity(kb - ka + 1);
        let mut rest = out;
        for k in ka..=kb {
            let lo = a.max(k * chunk_len);
            let hi = b.min((k + 1) * chunk_len);
            let (head, tail) = rest.split_at_mut((hi - lo) * elem_size);
            slices.push((k * chunk_len, lo, head));
            rest = tail;
        }
        let plan = self.plan;
        let dsid = self.ds.id();
        // Raw `hdf5-sys` FFI below bypasses hdf5-metno's wrappers and would
        // race any other thread inside libhdf5 (its state is global, not per
        // file), so it runs under the process-global lock the safe API uses.
        // `pread` and inflate touch no HDF5 state and run outside it.
        let Some(file) = &self.file else {
            let raw: Vec<(u32, Vec<u8>)> = hdf5::sync::sync(|| {
                (ka..=kb)
                    .map(|k| read_raw_chunk(dsid, k * chunk_len))
                    .collect::<Result<_>>()
            })?;
            return raw.into_par_iter().zip(slices).try_for_each(
                |((mask, cbuf), (first, lo, dst))| {
                    decode_into(plan, mask, &cbuf, first, lo, dst, elem_size)
                },
            );
        };
        let locs: Cow<[ChunkLoc]> = match &self.locs {
            Some(all) => Cow::Borrowed(&all[ka..=kb]),
            None => Cow::Owned(hdf5::sync::sync(|| {
                (ka..=kb)
                    .map(|k| chunk_location(dsid, k * chunk_len))
                    .collect::<Result<Vec<_>>>()
            })?),
        };
        slices.into_par_iter().zip(&*locs).try_for_each(
            |((first, lo, dst), &(mask, addr, size))| {
                let from = (lo - first) * elem_size;
                if plan.stored_raw(mask) {
                    // The values as stored: read the wanted part straight
                    // into place, with no chunk buffer and no copy.
                    if from + dst.len() > size {
                        return Err(ScxError::InvalidFormat(format!(
                            "chunk at element {first} is stored with {size} bytes, too few"
                        )));
                    }
                    return Ok(read_at(file, dst, addr + from as u64)?);
                }
                let mut cbuf = vec![0u8; size];
                read_at(file, &mut cbuf, addr)?;
                decode_into(plan, mask, &cbuf, first, lo, dst, elem_size)
            },
        )
    }
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
    // A whole-dataset read lists every chunk in one pass; a per-chunk lookup
    // walks HDF5's index each time.
    ChunkReader::with_plan(ds, plan, a == 0 && b >= ds.size())?.read_typed(a, b)
}

impl ChunkReader {
    /// Elements `[a, b)` as `T`, which must have the stored element size;
    /// values are taken bit for bit, so signedness may differ.
    pub(crate) fn read_typed<T: bytemuck::Pod>(&self, a: usize, b: usize) -> Result<Vec<T>> {
        // Assumes a little-endian host, as `TypedVec::from_le_bytes` did.
        let mut out = zeroed_vec_huge::<T>(b.saturating_sub(a));
        self.read_into(a, size_of::<T>(), bytemuck::cast_slice_mut(&mut out))?;
        Ok(out)
    }
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

/// Decode the stored chunk `cbuf` (filter mask `mask`), which starts at
/// element `first`, into `dst`: its elements from `lo` on. A set bit in the
/// filter mask means that filter was skipped for this chunk, so each stage
/// applies only when its own bit is clear.
fn decode_into(
    plan: ChunkPlan,
    mask: u32,
    cbuf: &[u8],
    first: usize,
    lo: usize,
    dst: &mut [u8],
    elem_size: usize,
) -> Result<()> {
    let whole = dst.len() == plan.chunk_len * elem_size;
    if whole && plan.deflated(mask) && !plan.shuffled(mask) {
        return inflate_exact(cbuf, dst);
    }
    let decoded;
    let bytes = if plan.stored_raw(mask) {
        cbuf
    } else {
        decoded = decode_chunk(plan, mask, cbuf, elem_size)?;
        &decoded
    };
    let from = (lo - first) * elem_size;
    let src = bytes.get(from..from + dst.len()).ok_or_else(|| {
        ScxError::InvalidFormat(format!(
            "chunk at element {first} decodes to {} bytes, too few",
            bytes.len()
        ))
    })?;
    dst.copy_from_slice(src);
    Ok(())
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

    /// Every chunk is checked once, whole and with no predecessor, and every
    /// boundary once, as the next chunk's first element after the previous
    /// chunk's last; one failing chunk fails the check.
    #[test]
    fn check_chunks_sees_each_chunk_and_boundary() {
        let p = tmp("check");
        let n = 1024 * 70 + 100; // a short last chunk
        let vals = values(n);
        write(&p, &vals, true);
        let ds = File::open(&p).unwrap().dataset("data").unwrap();
        let (chunks, edges) = (std::sync::Mutex::new(vec![]), std::sync::Mutex::new(vec![]));
        let f64_at = |b: &[u8]| f64::from_le_bytes(b[..8].try_into().unwrap());
        let check = |start: usize, prev: &[u8], b: &[u8]| {
            let first_ok = f64_at(b) == vals[start];
            if prev.is_empty() {
                chunks.lock().unwrap().push(start);
                first_ok && b.len() == 8 * 1024.min(n - start)
            } else {
                edges.lock().unwrap().push(start);
                first_ok && b.len() == 8 && f64_at(prev) == vals[start - 1]
            }
        };
        assert!(check_chunks(&ds, &check).unwrap());
        let mut chunks = chunks.into_inner().unwrap();
        chunks.sort();
        let starts: Vec<usize> = (0..n).step_by(1024).collect();
        assert_eq!(chunks, starts);
        assert_eq!(edges.into_inner().unwrap(), starts[1..]);
        let fail = |start: usize, _: &[u8], _: &[u8]| start != 1024 * 65;
        assert!(!check_chunks(&ds, &fail).unwrap());
        std::fs::remove_file(&p).ok();
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

    /// same_stored: chunks stored byte for byte alike are the same values; one
    /// changed value, other chunking or another deflate level are not.
    #[test]
    fn same_stored_compares_stored_bytes() {
        let path = tmp("same_stored");
        let vals = values(5000);
        let mut other = vals.clone();
        other[4321] += 1.0;
        {
            let f = File::create(&path).unwrap();
            for (name, v, chunk, level) in [
                ("a", &vals, 1024, 6),
                ("b", &vals, 1024, 6),
                ("c", &other, 1024, 6),
                ("d", &vals, 512, 6),
                ("e", &vals, 1024, 1),
            ] {
                f.new_dataset::<f64>()
                    .shape([v.len()])
                    .chunk([chunk])
                    .deflate(level)
                    .create(name)
                    .unwrap()
                    .write(v)
                    .unwrap();
            }
        }
        let f = File::open(&path).unwrap();
        let ds = |n: &str| f.dataset(n).unwrap();
        assert!(same_stored(&ds("a"), &ds("b")).unwrap());
        for n in ["c", "d", "e"] {
            assert!(!same_stored(&ds("a"), &ds(n)).unwrap(), "{n}");
        }
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
            let whole = ChunkReader::new(&ds, true).unwrap().unwrap();
            assert!(whole.file.is_some() && whole.locs.is_some());
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
                // A stream's reader, with all chunk locations listed up front.
                let got: Vec<u64> = whole.read_typed(a, b).unwrap();
                assert_eq!(got, want, "whole-index {a}..{b} shuffle={shuffle}");
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    /// With a userblock, chunk addresses are not file offsets: the reader must
    /// read chunks through libhdf5 and still give its bytes.
    #[test]
    fn userblock_file_reads_through_libhdf5() {
        let path = tmp("userblock");
        let vals = values(5000);
        for shuffle in [false, true] {
            {
                let f = File::with_options()
                    .with_fcpl(|p| p.userblock(512))
                    .create(&path)
                    .unwrap();
                let mut b = f.new_dataset::<f64>().shape([vals.len()]).chunk([1024]);
                if shuffle {
                    b = b.shuffle();
                }
                b.deflate(6).create("data").unwrap().write(&vals).unwrap();
            }
            let f = File::open(&path).unwrap();
            let ds = f.dataset("data").unwrap();
            for whole in [false, true] {
                let r = ChunkReader::new(&ds, whole).unwrap().unwrap();
                assert!(r.file.is_none(), "a userblock file must not be pread");
                for (a, b) in [(0, 5000), (1000, 3100), (4999, 5000), (7, 7)] {
                    let got: Vec<f64> = r.read_typed(a, b).unwrap();
                    assert_eq!(got, &vals[a..b], "{a}..{b} shuffle={shuffle}");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The leaves the read-ahead hint names must be the dataset's real index:
    /// read as nodes, they list exactly the chunks libhdf5 itself lists. Small
    /// (one node), two levels and three; a file without this index gets none.
    #[cfg(all(hdf5_1_14, target_os = "linux"))]
    #[test]
    fn index_leaves_list_every_chunk() {
        let path = tmp("index_leaves");
        for (n, depth, v1) in [
            (40usize, 0, false),
            (3000, 1, false),
            (9000, 2, false),
            (3000, 1, true),
        ] {
            {
                // hdf5-metno writes version 2 object headers; the earliest
                // format, h5py's default, version 1.
                let f = File::with_options()
                    .with_fapl(|p| if v1 { p.libver_earliest() } else { p })
                    .create(&path)
                    .unwrap();
                let vals: Vec<u32> = (0..n as u32).collect();
                let ds = f.new_dataset_builder().with_data(&vals).chunk(2);
                ds.create("data").unwrap();
                // Attributes push the layout message's neighbours around, as
                // anndata's encoding attributes do.
                let ds = f.dataset("data").unwrap();
                for k in 0..20 {
                    let a = ds.new_attr::<u64>().create(format!("a{k}").as_str());
                    a.unwrap().write_scalar(&7u64).unwrap();
                }
            }
            let f = File::open(&path).unwrap();
            let ds = f.dataset("data").unwrap();
            let (file, leaves, node) = index_leaves(&ds).expect("a v1 chunk index");
            let entry = 32; // a 1-D key (24 bytes) and an address
            let mut got = vec![];
            for &at in &leaves {
                let (level, chunks) = index_node(&file, at, node, entry).unwrap();
                assert_eq!(level, 0);
                got.extend(chunks);
            }
            let want: Vec<u64> = chunk_locations(&ds, 2, n)
                .unwrap()
                .iter()
                .map(|l| l.1)
                .collect();
            assert_eq!(got, want, "{n} values");
            assert_eq!(leaves.len() > 1, depth > 0, "{n} values");
            assert_eq!(leaves.len() > 64, depth > 1, "{n} values");
        }
        // The newer file format indexes chunks another way: no hint, no harm.
        {
            let f = File::with_options()
                .with_fapl(|p| p.libver_latest())
                .create(&path)
                .unwrap();
            let vals: Vec<u32> = (0..3000).collect();
            let ds = f.new_dataset_builder().with_data(&vals).chunk(2);
            ds.create("data").unwrap();
        }
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert!(index_leaves(&ds).is_none());
        assert_eq!(chunk_locations(&ds, 2, 3000).unwrap().len(), 1500);
        let _ = std::fs::remove_file(&path);
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

    /// An unfiltered chunked dataset is read with `pread` straight into
    /// place. Every kind of range must match the values written: whole,
    /// chunk-aligned, inside one chunk, across a boundary, the short last
    /// chunk, and empty.
    #[test]
    fn unfiltered_chunked_reads_match_the_values_written() {
        let path = tmp("unfiltered");
        let n = 10_000usize;
        let u: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let f64s: Vec<f64> = u.iter().map(|&v| v as f64 * 0.5).collect();
        {
            let f = File::create(&path).unwrap();
            f.new_dataset_builder()
                .with_data(&u)
                .chunk(1024)
                .create("u")
                .unwrap();
            f.new_dataset_builder()
                .with_data(&f64s)
                .chunk(1024)
                .create("f")
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        let ranges = [
            (0, n),
            (0, 1024),
            (1000, 1100),
            (1023, 1025),
            (9_000, n),
            (9_999, n),
            (3_000, 3_000),
        ];
        for (name, want) in [
            ("u", bytemuck::cast_slice::<u32, u8>(&u)),
            ("f", bytemuck::cast_slice::<f64, u8>(&f64s)),
        ] {
            let ds = f.dataset(name).unwrap();
            let plan = chunk_plan(&ds).expect("an unfiltered chunked dataset has a plan");
            let elem = ds.dtype().unwrap().size();
            for (a, b) in ranges {
                let got = read_range_parallel(&ds, a, b, elem, plan).unwrap();
                assert_eq!(got, &want[a * elem..b * elem], "{name}[{a}..{b}]");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Bytes are taken as stored, so big-endian storage gets no plan and
    /// libhdf5 converts it on read, as before.
    #[test]
    fn big_endian_storage_has_no_plan_and_still_reads() {
        use hdf5_sys::h5t::{H5T_order_t, H5Tset_order};
        let path = tmp("bigendian");
        let vals: Vec<i32> = (0..3000).collect();
        {
            let f = File::create(&path).unwrap();
            let dt = hdf5::Datatype::from_type::<i32>().unwrap();
            hdf5::sync::sync(|| unsafe { H5Tset_order(dt.id(), H5T_order_t::H5T_ORDER_BE) });
            let ds = f
                .new_dataset_builder()
                .empty_as(&dt)
                .chunk(1024)
                .shape(vals.len())
                .create("be")
                .unwrap();
            ds.write(&vals).unwrap();
        }
        let f = File::open(&path).unwrap();
        let ds = f.dataset("be").unwrap();
        assert_eq!(
            ds.dtype().unwrap().byte_order(),
            hdf5::datatype::ByteOrder::BigEndian
        );
        assert!(
            chunk_plan(&ds).is_none(),
            "big-endian storage must not take the pread path"
        );
        assert_eq!(
            crate::h5::read_u32_range(&ds, 10..20).unwrap(),
            (10..20).collect::<Vec<u32>>()
        );
        let _ = std::fs::remove_file(&path);
    }
}
