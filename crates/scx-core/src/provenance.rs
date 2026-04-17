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
    pub sha256: String,
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
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
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
    let json = serde_json::to_string_pretty(record)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(Path::new(&s), json)
}
