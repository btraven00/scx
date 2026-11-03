//! R serialized objects: `saveRDS` (`.rds`) and `save` (`.rda`) files holding
//! a Seurat object. See [`RdsSeuratReader`] for what is read.
//!
//! Only gzip-compressed (the `saveRDS` default) and uncompressed XDR streams
//! are handled; bzip2/xz files get an error saying how to re-save them.
//!
//! With the `net` feature an `.rds` URL is downloaded to a temporary file
//! first: the reader makes several forward passes over the stream.

mod seurat;
mod xdr;

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

pub use seurat::RdsSeuratReader;

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

/// A downloaded copy of a remote file, deleted when dropped.
pub struct Spool {
    path: std::path::PathBuf,
    /// `"sha256:<hex>"` of the downloaded bytes, hashed as they arrived.
    hash: String,
}

impl Spool {
    /// Where the copy is on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `"sha256:<hex>"` of the downloaded bytes.
    pub fn hash(&self) -> &str {
        &self.hash
    }
}

impl Drop for Spool {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Download `url` (any object-store location) to a temporary file, deleted
/// when the returned [`Spool`] is dropped.
// ponytail: one full download before parsing; stream it straight into the
// skeleton pass if remote .rds files get big enough for the wait to matter.
#[cfg(feature = "net")]
pub async fn spool(url: &str) -> Result<Spool> {
    use futures::StreamExt;
    use object_store::ObjectStoreExt;
    use sha2::Digest;
    use std::io::Write;

    let (store, path) = crate::net::resolve_store(url)?;
    let name = path.filename().unwrap_or("download.rds").to_string();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut spool = Spool {
        path: std::env::temp_dir().join(format!("scx-{}-{nanos}-{name}", std::process::id())),
        hash: String::new(),
    };
    let net = |e: object_store::Error| crate::parquet::net_err(format!("download '{url}': {e}"));
    let mut body = store.get(&path).await.map_err(net)?.into_stream();
    let mut f = std::io::BufWriter::new(std::fs::File::create(&spool.path)?);
    let mut hasher = sha2::Sha256::new();
    while let Some(bytes) = body.next().await {
        let bytes = bytes.map_err(net)?;
        hasher.update(&bytes);
        f.write_all(&bytes)?;
    }
    f.flush()?;
    spool.hash = format!("sha256:{}", crate::provenance::hex(hasher));
    Ok(spool)
}

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
