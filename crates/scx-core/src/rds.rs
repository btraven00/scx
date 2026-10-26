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

use std::io::{BufReader, Read};
use std::path::Path;

pub use seurat::RdsSeuratReader;

use crate::error::{Result, ScxError};

/// The decompressed stream of `path`, positioned at its start.
pub(crate) fn open_stream(path: &Path) -> Result<xdr::Xdr<Box<dyn Read + Send>>> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 6];
    let n = f.read(&mut magic)?;
    drop(f);
    let file = BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
    let r: Box<dyn Read + Send> = match &magic[..n] {
        [0x1f, 0x8b, ..] => Box::new(BufReader::with_capacity(
            1 << 20,
            flate2::read::MultiGzDecoder::new(file),
        )),
        [b'B', b'Z', b'h', ..] | [0xfd, b'7', b'z', b'X', b'Z', 0] => {
            return Err(ScxError::InvalidFormat(format!(
                "rds: '{}' is bzip2/xz-compressed; re-save it in R with \
                 saveRDS(obj, file, compress = TRUE) (gzip)",
                path.display()
            )));
        }
        _ => Box::new(file),
    };
    Ok(xdr::Xdr::new(r))
}

/// Whether `path` is an R serialization stream (gzip or plain, `.rds` or `.rda`).
pub fn sniff(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let Ok(mut x) = open_stream(path) else {
        // bzip2/xz: still R data if the extension says so, so open() can explain.
        return matches!(
            path.extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("rds" | "rda" | "rdata")
        );
    };
    x.header().is_ok()
}

/// Whether a URL or path names an R data file by its extension.
pub fn is_rds_name(location: &str) -> bool {
    let path = location
        .split(['?', '#'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    [".rds", ".rda", ".rdata"].iter().any(|e| path.ends_with(e))
}

/// A downloaded copy of a remote file, deleted when dropped.
pub(crate) struct Spool(pub(crate) std::path::PathBuf);

impl Drop for Spool {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Download `url` (any object-store location) to a temporary file.
// ponytail: one full download before parsing; stream it straight into the
// skeleton pass if remote .rds files get big enough for the wait to matter.
#[cfg(feature = "net")]
pub(crate) async fn spool(url: &str) -> Result<Spool> {
    use futures::StreamExt;
    use object_store::ObjectStoreExt;
    use std::io::Write;

    let (store, path) = crate::net::resolve_store(url)?;
    let name = path.filename().unwrap_or("download.rds").to_string();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let spool =
        Spool(std::env::temp_dir().join(format!("scx-{}-{nanos}-{name}", std::process::id())));
    let net = |e: object_store::Error| crate::parquet::net_err(format!("download '{url}': {e}"));
    let mut body = store.get(&path).await.map_err(net)?.into_stream();
    let mut f = std::io::BufWriter::new(std::fs::File::create(&spool.0)?);
    while let Some(bytes) = body.next().await {
        f.write_all(&bytes.map_err(net)?)?;
    }
    f.flush()?;
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
    }
}
