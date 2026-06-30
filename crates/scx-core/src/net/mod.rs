//! Network transport for object-store-backed readers.
//!
//! This is the reusable plumbing layer that every network reader sits on top of
//! (Parquet today; Zarr / remote h5ad / ranged HDF5 on the roadmap). It turns a
//! user-facing location string into a concrete [`ObjectStore`] handle + key:
//!
//! - `s3://`, `gs://`, `http(s)://`, `memory://`, `file://`, … → dispatched by
//!   [`object_store::parse_url_opts`] to the matching backend.
//! - a bare local path (`/data/x.parquet`, `out.h5ad`) → a [`LocalFileSystem`]
//!   store, so a local file flows through the same readers as a remote one.
//!
//! The split is decided by [`is_network_url`], which is also the predicate the
//! CLI uses to route a `convert` input away from the local HDF5 sniffers.
//!
//! Credentials are not wired up yet (v1 targets the public HTTPS Tahoe-100M
//! benchmark); private S3/GCS via env/config is a follow-up — see `// TODO creds`.

use std::sync::Arc;

use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::{parse_url_opts, ObjectStore};
use url::Url;

use crate::error::{Result, ScxError};

/// Schemes we route through the object-store transport. A string that parses as
/// a URL with one of these schemes is treated as network/object-store input;
/// anything else (bare relative or absolute paths) stays on the local readers.
const NET_SCHEMES: &[&str] = &[
    "s3", "gs", "gcs", "az", "azure", "http", "https", "memory", "file",
];

/// Returns true if `s` is an object-store location (vs. a bare local path).
///
/// Bare paths (`out.h5ad`, `/data/x.parquet`) fail `Url::parse` or carry an
/// unrecognized scheme and return false, so they keep flowing through the
/// existing filesystem readers.
pub fn is_network_url(s: &str) -> bool {
    match Url::parse(s) {
        Ok(url) => NET_SCHEMES.contains(&url.scheme()),
        Err(_) => false,
    }
}

/// Resolve a location string to an object-store handle and the key within it.
///
/// Reused by every network reader: construct the store here, hand the
/// `(store, path)` pair to the reader's `open()`.
pub fn resolve_store(location: &str) -> Result<(Arc<dyn ObjectStore>, StorePath)> {
    if is_network_url(location) {
        // object_store percent-decodes the path, turning a `%2F` into a real
        // `/`. That silently corrupts refs that contain slashes — notably
        // HuggingFace's auto-converted parquet ref `refs%2Fconvert%2Fparquet`,
        // which becomes `refs/convert/parquet` and 404s ("Invalid rev id"). Fail
        // early with guidance rather than emit a confusing not-found.
        if location.to_ascii_lowercase().contains("%2f") {
            return Err(ScxError::Net(format!(
                "URL '{location}' contains a percent-encoded slash (%2F), which object_store \
                 decodes to '/' and corrupts the ref. Use a ref without a slash — e.g. the \
                 dataset's main branch: .../resolve/main/<path>"
            )));
        }
        let url = Url::parse(location)
            .map_err(|e| ScxError::Net(format!("invalid URL '{location}': {e}")))?;
        // TODO creds: thread AWS_*/GOOGLE_* env (or explicit config) into the
        // options iterator for private buckets. Empty == anonymous/public.
        let (store, path) = parse_url_opts(&url, std::iter::empty::<(&str, &str)>())
            .map_err(|e| ScxError::Net(format!("cannot open '{location}': {e}")))?;
        Ok((Arc::from(store), path))
    } else {
        // Bare local path: an absolute filesystem key over a root LocalFileSystem.
        let abs = std::fs::canonicalize(location)
            .map_err(|e| ScxError::Net(format!("cannot resolve local path '{location}': {e}")))?;
        let path = StorePath::from_filesystem_path(&abs)
            .map_err(|e| ScxError::Net(format!("invalid local path '{location}': {e}")))?;
        let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
        Ok((store, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_network_vs_local() {
        assert!(is_network_url("s3://bucket/key.parquet"));
        assert!(is_network_url("gs://bucket/key.parquet"));
        assert!(is_network_url("https://example.com/data.parquet"));
        assert!(is_network_url("memory:///k"));
        assert!(is_network_url("file:///abs/x.parquet"));

        assert!(!is_network_url("out.h5ad"));
        assert!(!is_network_url("/data/x.parquet"));
        assert!(!is_network_url("./rel/path.parquet"));
        // Unsupported scheme is not routed through the transport.
        assert!(!is_network_url("ftp://host/x"));
    }

    #[test]
    fn resolves_memory_scheme_to_store_and_key() {
        // parse_url for memory:// yields a fresh empty InMemory — we only assert
        // routing (Ok + the key), not a round-trip.
        let (_store, path) = resolve_store("memory:///some/key.parquet").expect("resolve memory");
        assert_eq!(path.as_ref(), "some/key.parquet");
    }

    #[test]
    fn rejects_percent_encoded_slash_in_ref() {
        // HuggingFace auto-convert URLs use refs%2Fconvert%2Fparquet — object_store
        // would decode the %2F and 404. We reject early with guidance.
        let err = resolve_store(
            "https://huggingface.co/datasets/x/y/resolve/refs%2Fconvert%2Fparquet/a/0.parquet",
        )
        .unwrap_err();
        match err {
            ScxError::Net(msg) => assert!(msg.contains("%2F"), "message should mention %2F: {msg}"),
            other => panic!("expected ScxError::Net, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_scheme_is_net_error() {
        // ftp:// isn't a net scheme, so it falls to the local-path branch and
        // fails to canonicalize → ScxError::Net (not a panic).
        let err = resolve_store("ftp://host/x").unwrap_err();
        assert!(matches!(err, ScxError::Net(_)));
    }
}
