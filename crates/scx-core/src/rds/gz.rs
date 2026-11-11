//! Random access into a gzip file, after zlib's `examples/zran.c`.
//!
//! An `.rds` is one gzip stream, and its large vectors sit wherever R wrote
//! them, so reading `@x` of the last assay used to mean inflating the whole
//! file again, once per pass. [`IndexingGz`] inflates the file once (the
//! skeleton pass the reader makes anyway) and records an access point every
//! [`SPAN`] bytes of output: the input bit position at a deflate block
//! boundary and the 32 KB of output before it. From those points
//! [`GzIndex`] inflates any byte range of the stream directly, and splits a
//! long range across points to inflate its pieces in parallel.
//!
//! Only the first gzip member is indexed; later members (concatenated gzip)
//! and nested gzip are read sequentially as before.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use libz_rs_sys as z;
use rayon::prelude::*;

/// Bytes of history a deflate block can refer back to.
const WIN: usize = 32 * 1024;
/// Output bytes between access points: a random read inflates up to this
/// much before its first wanted byte, and each point holds a 32 KB window.
const SPAN: u64 = 4 << 20;
/// Output a [`ParReader`] inflates per refill, in up to `WINDOW / SPAN`
/// pieces at once.
const WINDOW: u64 = 64 << 20;
const IN_BUF: usize = 256 * 1024;

struct Point {
    /// Offset in the decompressed stream.
    out: u64,
    /// Offset in the file of the first byte holding unread bits.
    inp: u64,
    /// Bits of the byte before `inp` still to be read (0 = byte aligned).
    bits: u8,
    /// Up to 32 KB of output before `out`.
    window: Vec<u8>,
}

/// Access points into the first gzip member of a file.
pub(crate) struct GzIndex {
    path: PathBuf,
    points: Vec<Point>,
    /// Decompressed length of the indexed member.
    len: u64,
}

fn corrupt(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A raw-deflate inflate stream. Boxed: zlib keeps a pointer to it.
struct Inflater(Box<z::z_stream>);

impl Inflater {
    fn new() -> io::Result<Self> {
        // SAFETY: z_stream is plain data (pointers, integers, optional fn
        // pointers); all zero is the documented initial state.
        let mut s: Box<z::z_stream> = Box::new(unsafe { std::mem::zeroed() });
        // SAFETY: `s` is a zeroed z_stream; -15 selects raw deflate.
        let rc = unsafe {
            z::inflateInit2_(
                &mut *s,
                -15,
                z::zlibVersion(),
                std::mem::size_of::<z::z_stream>() as i32,
            )
        };
        if rc != z::Z_OK {
            return Err(io::Error::other(format!("inflateInit2 failed ({rc})")));
        }
        Ok(Self(s))
    }

    /// Inflate from `input` into `output`; returns (zlib code, bytes read, bytes written).
    fn run(&mut self, input: &[u8], output: &mut [u8], flush: i32) -> (i32, usize, usize) {
        let s = &mut *self.0;
        s.next_in = input.as_ptr();
        s.avail_in = input.len() as u32;
        s.next_out = output.as_mut_ptr();
        s.avail_out = output.len() as u32;
        // SAFETY: next_in/next_out point at live slices of the lengths given.
        let rc = unsafe { z::inflate(s, flush) };
        let read = input.len() - s.avail_in as usize;
        let written = output.len() - s.avail_out as usize;
        (rc, read, written)
    }
}

// SAFETY: the stream and its zlib state are owned by this value alone and
// used through `&mut self`, so moving it to another thread is sound; the
// raw pointers inside are only set for the duration of one `run` call.
unsafe impl Send for Inflater {}

impl Drop for Inflater {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`, ended once.
        unsafe { z::inflateEnd(&mut *self.0) };
    }
}

fn crc32(crc: u32, data: &[u8]) -> u32 {
    // SAFETY: a valid slice.
    unsafe { z::crc32_z(crc as _, data.as_ptr(), data.len()) as u32 }
}

/// A file read through a buffer, tracking the file offset of the next byte.
struct Input {
    file: File,
    buf: Vec<u8>,
    pos: usize,
    len: usize,
    /// File offset of `buf[pos]`.
    offset: u64,
}

impl Input {
    fn refill(&mut self) -> io::Result<bool> {
        if self.pos == self.len {
            self.len = self.file.read(&mut self.buf)?;
            self.pos = 0;
        }
        Ok(self.len > self.pos)
    }

    fn byte(&mut self) -> io::Result<u8> {
        if !self.refill()? {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        self.pos += 1;
        self.offset += 1;
        Ok(self.buf[self.pos - 1])
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
        self.offset += n as u64;
    }

    /// Skip a gzip member header (RFC 1952).
    fn gzip_header(&mut self) -> io::Result<()> {
        if (self.byte()?, self.byte()?, self.byte()?) != (0x1f, 0x8b, 8) {
            return Err(corrupt("not a gzip (deflate) member"));
        }
        let flags = self.byte()?;
        for _ in 0..6 {
            self.byte()?; // mtime, xfl, os
        }
        if flags & 4 != 0 {
            let n = self.byte()? as usize | (self.byte()? as usize) << 8;
            for _ in 0..n {
                self.byte()?;
            }
        }
        for bit in [8, 16] {
            if flags & bit != 0 {
                while self.byte()? != 0 {} // name, comment
            }
        }
        if flags & 2 != 0 {
            self.byte()?;
            self.byte()?;
        }
        Ok(())
    }
}

/// Sequential gzip decoder that records access points into the first
/// member while it inflates. Checks each member's CRC and length.
pub(crate) struct IndexingGz {
    input: Input,
    z: Inflater,
    /// Output; `out[..out_len]` ends with the latest history, and
    /// `out[out_pos..out_len]` is not yet read.
    out: Vec<u8>,
    out_pos: usize,
    out_len: usize,
    /// Output of the current member, and over all members.
    member_out: u64,
    total_out: u64,
    crc: u32,
    first_member: bool,
    eof: bool,
    points: Vec<Point>,
    /// Filled when the first member ends.
    done: Arc<Mutex<Option<GzIndex>>>,
    path: PathBuf,
}

impl IndexingGz {
    /// Open `path`, a gzip file; the index appears in the returned slot once
    /// the first member has been read to its end.
    pub(crate) fn open(path: &Path) -> io::Result<(Self, Arc<Mutex<Option<GzIndex>>>)> {
        let mut input = Input {
            file: File::open(path)?,
            buf: vec![0; IN_BUF],
            pos: 0,
            len: 0,
            offset: 0,
        };
        input.gzip_header()?;
        let done = Arc::new(Mutex::new(None));
        let first = Point {
            out: 0,
            inp: input.offset,
            bits: 0,
            window: Vec::new(),
        };
        Ok((
            Self {
                input,
                z: Inflater::new()?,
                out: vec![0; WIN + IN_BUF],
                out_pos: 0,
                out_len: 0,
                member_out: 0,
                total_out: 0,
                crc: 0,
                first_member: true,
                eof: false,
                points: vec![first],
                done: done.clone(),
                path: path.to_path_buf(),
            },
            done,
        ))
    }

    /// Inflate more output into `out`, keeping the last 32 KB as history.
    fn fill(&mut self) -> io::Result<()> {
        if self.out_len > WIN {
            self.out.copy_within(self.out_len - WIN..self.out_len, 0);
            self.out_len = WIN;
            self.out_pos = WIN;
        }
        loop {
            if !self.input.refill()? {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "gzip stream is truncated",
                ));
            }
            let (pos, len, out_len) = (self.input.pos, self.input.len, self.out_len);
            // Stop at block boundaries (Z_BLOCK) only once a point is due:
            // it costs a return per deflate block.
            let due = self.first_member && self.total_out - self.points.last().unwrap().out >= SPAN;
            let (rc, read, written) = self.z.run(
                &self.input.buf[pos..len],
                &mut self.out[out_len..],
                if due { z::Z_BLOCK } else { z::Z_NO_FLUSH },
            );
            self.input.consume(read);
            self.crc = crc32(self.crc, &self.out[out_len..out_len + written]);
            self.out_len += written;
            self.member_out += written as u64;
            self.total_out += written as u64;
            match rc {
                z::Z_STREAM_END => return self.end_member(),
                z::Z_OK | z::Z_BUF_ERROR => {}
                _ => return Err(corrupt(format!("corrupt gzip data (zlib {rc})"))),
            }
            let dt = self.z.0.data_type;
            let boundary = dt & 128 != 0 && dt & 64 == 0;
            if due && boundary {
                self.points.push(Point {
                    out: self.total_out,
                    inp: self.input.offset,
                    bits: (dt & 7) as u8,
                    window: self.out[self.out_len.saturating_sub(WIN)..self.out_len].to_vec(),
                });
            }
            if self.out_len > self.out_pos || self.out_len == self.out.len() {
                return Ok(());
            }
        }
    }

    /// Check the member trailer; publish the index after the first member;
    /// start the next member, if any.
    fn end_member(&mut self) -> io::Result<()> {
        let mut t = [0u8; 8];
        for b in &mut t {
            *b = self
                .input
                .byte()
                .map_err(|_| corrupt("gzip trailer is truncated"))?;
        }
        let crc = u32::from_le_bytes(t[..4].try_into().unwrap());
        let size = u32::from_le_bytes(t[4..].try_into().unwrap());
        if crc != self.crc || size != self.member_out as u32 {
            return Err(corrupt("gzip CRC or length mismatch: the file is corrupt"));
        }
        if self.first_member {
            self.first_member = false;
            *self.done.lock().unwrap() = Some(GzIndex {
                path: std::mem::take(&mut self.path),
                points: std::mem::take(&mut self.points),
                len: self.total_out,
            });
        }
        if !self.input.refill()? {
            self.eof = true;
            return Ok(());
        }
        self.input.gzip_header()?;
        // SAFETY: an initialised stream.
        if unsafe { z::inflateReset(&mut *self.z.0) } != z::Z_OK {
            return Err(io::Error::other("inflateReset failed"));
        }
        self.member_out = 0;
        self.crc = 0;
        Ok(())
    }
}

impl Read for IndexingGz {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.out_pos == self.out_len {
            if self.eof {
                return Ok(0);
            }
            self.fill()?;
        }
        let n = buf.len().min(self.out_len - self.out_pos);
        buf[..n].copy_from_slice(&self.out[self.out_pos..self.out_pos + n]);
        self.out_pos += n;
        Ok(n)
    }
}

impl GzIndex {
    /// Whether `offset..offset + len` lies in the indexed member.
    pub(crate) fn covers(&self, offset: u64, len: u64) -> bool {
        offset.checked_add(len).is_some_and(|end| end <= self.len)
    }

    /// Inflate `buf.len()` bytes of the stream from `offset`, starting at the
    /// access point before it.
    fn read_into(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let p = &self.points[self.points.partition_point(|p| p.out <= offset) - 1];
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(p.inp - u64::from(p.bits > 0)))?;
        let mut z = Inflater::new()?;
        if p.bits > 0 {
            let mut b = [0u8];
            file.read_exact(&mut b)?;
            // SAFETY: an initialised stream.
            let rc =
                unsafe { z::inflatePrime(&mut *z.0, p.bits.into(), (b[0] >> (8 - p.bits)).into()) };
            if rc != z::Z_OK {
                return Err(io::Error::other("inflatePrime failed"));
            }
        }
        if !p.window.is_empty() {
            // SAFETY: a valid slice.
            let rc = unsafe {
                z::inflateSetDictionary(&mut *z.0, p.window.as_ptr(), p.window.len() as u32)
            };
            if rc != z::Z_OK {
                return Err(io::Error::other("inflateSetDictionary failed"));
            }
        }
        let mut skip = (offset - p.out) as usize;
        let mut scratch = vec![0u8; if skip > 0 { 64 * 1024 } else { 0 }];
        let mut input = vec![0u8; IN_BUF];
        let (mut in_pos, mut in_len, mut filled) = (0, 0, 0);
        while filled < buf.len() {
            if in_pos == in_len {
                in_len = file.read(&mut input)?;
                in_pos = 0;
                if in_len == 0 {
                    return Err(corrupt("gzip stream ends before the requested bytes"));
                }
            }
            let out: &mut [u8] = if skip > 0 {
                let n = skip.min(scratch.len());
                &mut scratch[..n]
            } else {
                &mut buf[filled..]
            };
            let (rc, read, written) = z.run(&input[in_pos..in_len], out, z::Z_NO_FLUSH);
            in_pos += read;
            if skip > 0 {
                skip -= written;
            } else {
                filled += written;
            }
            match rc {
                z::Z_OK | z::Z_BUF_ERROR => {}
                z::Z_STREAM_END if skip == 0 && filled == buf.len() => {}
                z::Z_STREAM_END => return Err(corrupt("read past the end of the gzip member")),
                _ => return Err(corrupt(format!("corrupt gzip data (zlib {rc})"))),
            }
        }
        Ok(())
    }

    /// `len` bytes of the stream from `offset`, inflated in parallel: one
    /// piece per access point the range crosses.
    pub(crate) fn read(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let end = offset + len as u64;
        let mut out = vec![0u8; len];
        let mut cuts: Vec<u64> = vec![offset];
        cuts.extend(
            self.points
                .iter()
                .map(|p| p.out)
                .filter(|&o| o > offset && o < end),
        );
        cuts.push(end);
        let mut pieces: Vec<(u64, &mut [u8])> = Vec::with_capacity(cuts.len());
        let mut rest = out.as_mut_slice();
        for w in cuts.windows(2) {
            let (head, tail) = rest.split_at_mut((w[1] - w[0]) as usize);
            pieces.push((w[0], head));
            rest = tail;
        }
        pool().install(|| {
            pieces
                .into_par_iter()
                .try_for_each(|(at, piece)| self.read_into(at, piece))
        })?;
        Ok(out)
    }
}

/// The stream from `offset` to the end of the indexed member, read ahead
/// [`WINDOW`] bytes at a time with [`GzIndex::read`].
pub(crate) struct ParReader {
    index: Arc<GzIndex>,
    pos: u64,
    buf: Vec<u8>,
    at: usize,
}

impl ParReader {
    pub(crate) fn new(index: Arc<GzIndex>, offset: u64) -> Self {
        Self {
            index,
            pos: offset,
            buf: Vec::new(),
            at: 0,
        }
    }
}

impl Read for ParReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.at == self.buf.len() {
            let ix = &self.index;
            if self.pos >= ix.len {
                return Ok(0);
            }
            // End on an access point, so the next refill starts on one.
            let want = self.pos + WINDOW;
            let stop = ix
                .points
                .iter()
                .map(|p| p.out)
                .find(|&o| o >= want)
                .unwrap_or(ix.len);
            self.buf = ix.read(self.pos, (stop - self.pos) as usize)?;
            self.at = 0;
            self.pos = stop;
        }
        let n = out.len().min(self.buf.len() - self.at);
        out[..n].copy_from_slice(&self.buf[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

/// The thread pool for this process's parallel reads.
///
/// Not rayon's global pool: a process forked after using it (R's
/// `parallel::mclapply`) inherits the pool without its threads, and its
/// first parallel read waits forever. The pool is rebuilt when the process
/// id changes.
pub(crate) fn pool() -> Arc<rayon::ThreadPool> {
    static POOL: Mutex<Option<(u32, Arc<rayon::ThreadPool>)>> = Mutex::new(None);
    let pid = std::process::id();
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((owner, p)) = slot.as_ref() {
        if *owner == pid {
            return p.clone();
        }
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .build()
            .expect("rds: cannot start reader threads"),
    );
    // The parent's pool has no threads here; dropping it would signal them.
    if let Some(old) = slot.replace((pid, p.clone())) {
        std::mem::forget(old);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use std::io::Write;

    /// Pseudo-random but compressible bytes, long enough for many points.
    fn data(n: usize) -> Vec<u8> {
        let mut x: u64 = 1;
        (0..n)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if i % 7 == 0 {
                    (x >> 59) as u8
                } else {
                    (i / 1000) as u8
                }
            })
            .collect()
    }

    fn gz(parts: &[&[u8]]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for p in parts {
            let mut e = GzEncoder::new(Vec::new(), flate2::Compression::default());
            e.write_all(p).unwrap();
            f.write_all(&e.finish().unwrap()).unwrap();
        }
        f
    }

    #[test]
    fn index_reads_any_range_like_a_sequential_pass() {
        let d = data(30 << 20);
        let f = gz(&[&d]);
        let (mut r, slot) = IndexingGz::open(f.path()).unwrap();
        let mut all = Vec::new();
        r.read_to_end(&mut all).unwrap();
        assert_eq!(all, d);
        let ix = Arc::new(slot.lock().unwrap().take().unwrap());
        assert!(ix.points.len() > 5, "{} points", ix.points.len());
        assert!(ix.points.iter().any(|p| p.bits > 0), "no unaligned point");
        for (off, len) in [
            (0, 10),
            (1, 5 << 20),
            (SPAN - 3, 7),
            (13 << 20, 9 << 20),
            ((30 << 20) - 5, 5),
        ] {
            assert_eq!(
                ix.read(off, len).unwrap(),
                &d[off as usize..off as usize + len],
                "{off}+{len}"
            );
        }
        let mut tail = Vec::new();
        ParReader::new(ix.clone(), 1234)
            .read_to_end(&mut tail)
            .unwrap();
        assert_eq!(tail, &d[1234..]);
        assert!(!ix.covers(30 << 20, 1));
    }

    /// Concatenated members read whole; only the first is indexed.
    #[test]
    fn later_members_are_read_but_not_indexed() {
        let (a, b) = (data(5 << 20), data(1 << 20));
        let f = gz(&[&a, &b]);
        let (mut r, slot) = IndexingGz::open(f.path()).unwrap();
        let mut all = Vec::new();
        r.read_to_end(&mut all).unwrap();
        assert_eq!(all, [a.clone(), b].concat());
        let ix = slot.lock().unwrap().take().unwrap();
        assert!(ix.covers(0, a.len() as u64) && !ix.covers(0, a.len() as u64 + 1));
    }

    #[test]
    fn corruption_is_an_error() {
        let d = data(1 << 20);
        let f = gz(&[&d]);
        let mut bytes = std::fs::read(f.path()).unwrap();
        let n = bytes.len();
        bytes[n - 6] ^= 1; // the length in the trailer
        std::fs::write(f.path(), &bytes).unwrap();
        let (mut r, _) = IndexingGz::open(f.path()).unwrap();
        let err = r.read_to_end(&mut Vec::new()).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
        bytes.truncate(n / 2);
        std::fs::write(f.path(), &bytes).unwrap();
        let (mut r, _) = IndexingGz::open(f.path()).unwrap();
        assert!(r.read_to_end(&mut Vec::new()).is_err());
    }
}
