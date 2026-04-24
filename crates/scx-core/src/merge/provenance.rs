use std::collections::HashMap;
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
///     "layers/norm": { "source_path": "...", "sha256": "...", "added_at": "..." }
///   },
///   "tags": { "pipeline_version": "0.4.1" }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotProvenance {
    pub scx_version: String,
    pub base: BaseAnchor,
    pub slots: HashMap<String, SlotEntry>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
}

impl SlotProvenance {
    pub fn new(base: BaseAnchor) -> Self {
        Self {
            scx_version: env!("CARGO_PKG_VERSION").to_string(),
            base,
            slots: HashMap::new(),
            tags: HashMap::new(),
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
                added_at: utc_now_rfc3339(),
            },
        );
    }

    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
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
