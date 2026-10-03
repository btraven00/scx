use std::io::{BufReader, Read};
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceInfo {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputInfo {
    pub path: String,
    /// Absent for directory outputs (e.g. a BPCells matrix dir).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub n_obs: usize,
    pub n_vars: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub scx_version: String,
    pub converted_at: String,
    pub source: SourceInfo,
    pub output: OutputInfo,
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    hash_file(&mut hasher, path)?;
    Ok(hex(hasher))
}

/// SHA-256 of a file, or of a directory-backed dataset (Zarr, BPCells,
/// MatrixMarket): every file under it in sorted relative-path order, each as
/// its path, a NUL, then its bytes. Stable across machines and listings.
pub fn sha256_path(path: &Path) -> std::io::Result<String> {
    if !path.is_dir() {
        return sha256_file(path);
    }
    fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if p.is_dir() {
                files(&p, out)?;
            } else {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut all = Vec::new();
    files(path, &mut all)?;
    all.sort();
    let mut hasher = Sha256::new();
    for f in &all {
        let rel = f.strip_prefix(path).unwrap_or(f);
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0u8]);
        hash_file(&mut hasher, f)?;
    }
    Ok(hex(hasher))
}

fn hash_file(hasher: &mut Sha256, path: &Path) -> std::io::Result<()> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hasher.update(&buf[..n]);
    }
}

fn hex(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn utc_now_rfc3339() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Deterministic provenance value — safe to bake into the artifact's uns.
/// Contains only inputs + version; no timestamp so the output is reproducible.
pub fn det_record(
    source_path: &str,
    source_url: Option<&str>,
    source_sha256: Option<&str>,
    n_obs: usize,
    n_vars: usize,
) -> serde_json::Value {
    let mut src = serde_json::json!({ "path": source_path });
    if let Some(url) = source_url {
        src["url"] = serde_json::Value::String(url.to_string());
    }
    if let Some(sha) = source_sha256 {
        src["sha256"] = serde_json::Value::String(sha.to_string());
    }
    serde_json::json!({
        "scx_version": env!("CARGO_PKG_VERSION"),
        "source": src,
        "n_obs": n_obs,
        "n_vars": n_vars,
    })
}

pub fn write_sidecar(record: &ProvenanceRecord, output: &Path) -> std::io::Result<()> {
    let mut s = output.as_os_str().to_owned();
    s.push(".prov.json");
    let json = serde_json::to_string_pretty(record).map_err(std::io::Error::other)?;
    std::fs::write(Path::new(&s), json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_path_hashes_directories_by_content() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        for d in [a.path(), b.path()] {
            std::fs::create_dir(d.join("X")).unwrap();
            std::fs::write(d.join("X/data"), b"123").unwrap();
            std::fs::write(d.join(".zattrs"), b"{}").unwrap();
        }
        let ha = sha256_path(a.path()).unwrap();
        assert_eq!(
            ha,
            sha256_path(b.path()).unwrap(),
            "same content, same hash"
        );
        std::fs::write(b.path().join("X/data"), b"124").unwrap();
        assert_ne!(ha, sha256_path(b.path()).unwrap(), "content change shows");
        std::fs::write(a.path().join("f"), b"x").unwrap();
        assert_eq!(
            sha256_path(&a.path().join("f")).unwrap(),
            sha256_file(&a.path().join("f")).unwrap()
        );
    }
}
