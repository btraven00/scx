//! R serialized objects: `saveRDS` (`.rds`) and `save` (`.rda`) files holding
//! a Seurat object or a SummarizedExperiment (SingleCellExperiment,
//! SpatialExperiment). See [`RdsReader`] for what is read.
//!
//! Only gzip-compressed (the `saveRDS` default) and uncompressed XDR streams
//! are handled; bzip2/xz files get an error saying how to re-save them.
//!
//! With the `net` feature an `.rds` URL is downloaded to a temporary file
//! first: the reader makes several forward passes over the stream.

mod gz;
mod sce;
mod seurat;
mod xdr;

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub use seurat::RdsReader;
/// The reader's name from when it read Seurat objects only.
pub type RdsSeuratReader = RdsReader;

use crate::error::{Result, ScxError};

/// The decompressed stream of `path`, positioned at its start. Gzip layers
/// are peeled until the R stream shows: GEO's `x.rds.gz` is often a gzip
/// around `saveRDS`'s own gzip.
pub(crate) fn open_stream(path: &Path) -> Result<xdr::Xdr<Box<dyn Read + Send>>> {
    let mut r: Box<dyn BufRead + Send> = Box::new(BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(path)?,
    ));
    for _ in 0..3 {
        match r.fill_buf()? {
            [0x1f, 0x8b, ..] => {
                r = Box::new(BufReader::with_capacity(
                    1 << 20,
                    flate2::read::MultiGzDecoder::new(r),
                ))
            }
            [b'B', b'Z', b'h', ..] | [0xfd, b'7', b'z', b'X', b'Z', 0, ..] => {
                return Err(ScxError::InvalidFormat(format!(
                    "rds: '{}' is bzip2/xz-compressed; re-save it in R with \
                     saveRDS(obj, file, compress = TRUE) (gzip)",
                    path.display()
                )));
            }
            _ => break,
        }
    }
    Ok(xdr::Xdr::new(Box::new(r)))
}

/// Where a reader goes back to the decompressed stream for its lazy
/// vectors: through the gzip index built by the first pass when there is
/// one, else by inflating the file again from its start.
#[derive(Clone)]
pub(crate) struct Source {
    pub(crate) path: PathBuf,
    index: Option<Arc<gz::GzIndex>>,
}

type Stream = xdr::Xdr<Box<dyn Read + Send>>;

impl Source {
    /// Whether reads skip straight to their offset (and run in parallel).
    pub(crate) fn indexed(&self) -> bool {
        self.index.is_some()
    }

    /// The stream positioned at `offset`.
    pub(crate) fn stream_at(&self, offset: u64) -> Result<Stream> {
        match &self.index {
            Some(ix) if ix.covers(offset, 0) => Ok(xdr::Xdr::at(
                Box::new(gz::ParReader::new(ix.clone(), offset)),
                offset,
            )),
            _ => {
                let mut x = open_stream(&self.path)?;
                x.seek_to(offset)?;
                Ok(x)
            }
        }
    }

    /// `len` bytes of the stream at `offset`.
    pub(crate) fn bytes_at(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        match &self.index {
            Some(ix) if ix.covers(offset, len as u64) => Ok(ix
                .read(offset, len)
                .map_err(|e| ScxError::InvalidFormat(format!("rds: unreadable stream ({e})")))?),
            _ => self.stream_at(offset)?.bytes(len),
        }
    }
}

/// The stream of `path` from its start, for the reader's first pass. A
/// plain gzip file (what `saveRDS` writes) is indexed on the way: read the
/// stream to its end, then [`indexed_source`] makes later passes cheap.
pub(crate) fn open_indexing(path: &Path) -> Result<(Stream, Arc<Mutex<Option<gz::GzIndex>>>)> {
    let none = || Arc::new(Mutex::new(None));
    let mut magic = [0u8; 2];
    let is_gzip =
        std::fs::File::open(path)?.read_exact(&mut magic).is_ok() && magic == [0x1f, 0x8b];
    if !is_gzip {
        return Ok((open_stream(path)?, none()));
    }
    let (r, slot) = gz::IndexingGz::open(path)?;
    let mut r = BufReader::with_capacity(1 << 20, r);
    // gzip in gzip (GEO's x.rds.gz): the outer layer's offsets are not the
    // stream's, so read it the plain way.
    if r.fill_buf()?.starts_with(&[0x1f, 0x8b]) {
        return Ok((open_stream(path)?, none()));
    }
    Ok((xdr::Xdr::new(Box::new(r)), slot))
}

/// The [`Source`] for `path` once the stream from [`open_indexing`] was read
/// to its end.
pub(crate) fn indexed_source(path: &Path, slot: &Mutex<Option<gz::GzIndex>>) -> Source {
    Source {
        path: path.to_path_buf(),
        index: slot.lock().unwrap().take().map(Arc::new),
    }
}

/// Whether `path` is an R serialization stream (gzip or plain, `.rds` or `.rda`).
pub fn sniff(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let Ok(mut x) = open_stream(path) else {
        // bzip2/xz: still R data if the name says so, so open() can explain.
        return is_rds_name(&path.to_string_lossy());
    };
    x.header().is_ok()
}

/// The object tree of an R data file, as text, without reading its large
/// vectors: classes, S4 slots (`@name`), list element names, lengths and
/// `dim`s, to `max_depth` levels and up to 12 children per node.
pub fn tree(path: &Path, max_depth: usize) -> Result<String> {
    let mut x = open_stream(path)?;
    x.header()?;
    let root = xdr::Parser::new(&mut x).item()?;
    let mut out = String::new();
    xdr::write_tree(&root, "<root>", 0, max_depth, &mut out);
    Ok(out)
}

/// Whether a URL or path names an R data file by its extension (`.rds`,
/// `.rda`, `.rdata`, each optionally `.gz`). Percent-escapes are decoded
/// first: GEO links spell `x.rds.gz` as `x%2Erds%2Egz`.
pub fn is_rds_name(location: &str) -> bool {
    let path = location.split(['?', '#']).next().unwrap_or_default();
    let name = percent_decode(path).to_ascii_lowercase();
    let name = name.strip_suffix(".gz").unwrap_or(&name);
    [".rds", ".rda", ".rdata"].iter().any(|e| name.ends_with(e))
}

/// `%XX` escapes decoded; anything malformed is kept as is.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        match (
            b[i],
            b.get(i + 1).copied().and_then(hex),
            b.get(i + 2).copied().and_then(hex),
        ) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            (c, ..) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// Kept here for callers from before the spool moved to its own module.
#[cfg(feature = "net")]
pub use crate::spool::spool;
pub use crate::spool::Spool;

#[cfg(test)]
mod tests {
    #[test]
    fn rds_names() {
        assert!(super::is_rds_name("https://x.org/a/pbmc.rds"));
        assert!(super::is_rds_name("s3://b/k/obj.RDS?versionId=1"));
        assert!(super::is_rds_name("data.RData"));
        assert!(!super::is_rds_name("https://x.org/a.parquet"));
        assert!(super::is_rds_name(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138826/suppl/GSE138826%5Fregen%5Fdata%2Erds%2Egz"
        ));
        assert!(super::is_rds_name("obj.rda.gz"));
        assert!(!super::is_rds_name("matrix.mtx.gz"));
    }
}
