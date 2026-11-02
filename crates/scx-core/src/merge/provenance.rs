use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provenance::utc_now_rfc3339;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseAnchor {
    pub path: String,
    pub sha256: String,
    pub n_obs: usize,
    pub n_vars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotEntry {
    pub source_path: String,
    pub sha256: String,
    /// When the slot was patched in. Lives in the sidecar, never in uns: a
    /// wall-clock time would make the merged file differ on every run. Read
    /// here only from files written before that, to carry it over.
    #[serde(default, skip_serializing)]
    pub added_at: Option<String>,
}

/// `<output>.prov.json` of a merge: the non-deterministic part of its
/// provenance, kept out of the file so the file is byte-reproducible.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MergeSidecar {
    pub scx_version: String,
    pub slots: BTreeMap<String, SidecarSlot>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SidecarSlot {
    pub added_at: String,
}

/// Provenance record stored in `uns["scx_provenance"]` of a merged h5ad.
///
/// Schema:
/// ```json
/// {
///   "scx_version": "0.2.0",
///   "base": { "path": "...", "sha256": "...", "n_obs": N, "n_vars": M },
///   "slots": {
///     "layers/norm": { "source_path": "...", "sha256": "..." }
///   },
///   "tags": { "pipeline_version": "0.4.1" }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotProvenance {
    pub scx_version: String,
    pub base: BaseAnchor,
    pub slots: BTreeMap<String, SlotEntry>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<String, String>,
}

impl SlotProvenance {
    pub fn new(base: BaseAnchor) -> Self {
        Self {
            scx_version: env!("CARGO_PKG_VERSION").to_string(),
            base,
            slots: BTreeMap::new(),
            tags: BTreeMap::new(),
        }
    }

    pub fn add_slot(
        &mut self,
        slot_key: impl Into<String>,
        source_path: impl AsRef<Path>,
        sha256: impl Into<String>,
    ) {
        self.slots.insert(
            slot_key.into(),
            SlotEntry {
                source_path: source_path.as_ref().to_string_lossy().into_owned(),
                sha256: sha256.into(),
                added_at: Some(utc_now_rfc3339()),
            },
        );
    }

    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Record this run's `added_at` times in `<output>.prov.json`. With
    /// `append`, times from earlier runs in an existing sidecar are kept.
    pub fn write_sidecar(&self, output: &Path, append: bool) -> crate::error::Result<()> {
        let path = crate::provenance::sidecar_path(output);
        let mut sidecar = match std::fs::read(&path) {
            Ok(bytes) if append => serde_json::from_slice(&bytes).map_err(|e| {
                crate::error::ScxError::InvalidFormat(format!("{}: {e}", path.display()))
            })?,
            _ => MergeSidecar::default(),
        };
        sidecar.scx_version = env!("CARGO_PKG_VERSION").to_string();
        for (key, entry) in &self.slots {
            if let Some(at) = &entry.added_at {
                sidecar.slots.insert(
                    key.clone(),
                    SidecarSlot {
                        added_at: at.clone(),
                    },
                );
            }
        }
        let json = serde_json::to_string_pretty(&sidecar)
            .map_err(|e| crate::error::ScxError::InvalidFormat(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn to_json(&self) -> serde_json::Result<Value> {
        serde_json::to_value(self)
    }

    pub fn from_json(v: &Value) -> serde_json::Result<Self> {
        serde_json::from_value(v.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file merged before the sidecar existed has `added_at` in uns: the
    /// next append moves it to the sidecar and drops it from uns.
    #[test]
    fn legacy_added_at_moves_to_sidecar() {
        let legacy = serde_json::json!({
            "scx_version": "0.2.0",
            "base": {"path": "b.h5ad", "sha256": "x", "n_obs": 1, "n_vars": 1},
            "slots": {"layers/n": {"source_path": "p", "sha256": "y", "added_at": "2026-04-25T10:04:12Z"}}
        });
        let prov = SlotProvenance::from_json(&legacy).unwrap();
        assert!(prov.to_json().unwrap()["slots"]["layers/n"]
            .get("added_at")
            .is_none());

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("m.h5ad");
        prov.write_sidecar(&out, true).unwrap();
        let path = crate::provenance::sidecar_path(&out);
        let sc: MergeSidecar = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(sc.slots["layers/n"].added_at, "2026-04-25T10:04:12Z");
    }

    #[test]
    fn roundtrip_json() {
        let mut prov = SlotProvenance::new(BaseAnchor {
            path: "source.h5ad".to_string(),
            sha256: "abc123".to_string(),
            n_obs: 100,
            n_vars: 500,
        });
        prov.add_slot("layers/norm", Path::new("norm.h5ad"), "def456");
        prov.set_tag("pipeline_version", "0.4.1");

        let json = prov.to_json().unwrap();
        let back = SlotProvenance::from_json(&json).unwrap();
        assert_eq!(back.base.n_obs, 100);
        assert_eq!(back.base.n_vars, 500);
        assert_eq!(back.base.sha256, "abc123");
        assert!(back.slots.contains_key("layers/norm"));
        assert_eq!(back.slots["layers/norm"].sha256, "def456");
        assert_eq!(back.tags["pipeline_version"], "0.4.1");
    }

    #[test]
    fn tags_omitted_when_empty() {
        let prov = SlotProvenance::new(BaseAnchor {
            path: "f.h5ad".to_string(),
            sha256: "x".to_string(),
            n_obs: 10,
            n_vars: 20,
        });
        let json = prov.to_json().unwrap();
        assert!(json.get("tags").is_none());
    }
}
