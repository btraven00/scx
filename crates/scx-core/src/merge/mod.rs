use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{Result, ScxError};

pub mod align;
pub mod provenance;

// ---------------------------------------------------------------------------
// SlotSelector + parser (task 3)
// ---------------------------------------------------------------------------

/// Identifies one named slot within an h5ad file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotSelector {
    Layer(String),
    ObsColumn(String),
    VarColumn(String),
    Obsm(String),
    Varm(String),
}

impl SlotSelector {
    /// Parse a slot string: `"layers/name"`, `"obs/col"`, `"var/col"`,
    /// `"obsm/X_pca"`, or `"varm/PCs"`.
    pub fn parse(s: &str) -> Result<Self> {
        let (prefix, name) = s.split_once('/').ok_or_else(|| {
            ScxError::InvalidFormat(format!(
                "slot selector '{s}' must be 'group/name' (e.g. 'layers/norm')"
            ))
        })?;
        if name.is_empty() {
            return Err(ScxError::InvalidFormat(format!(
                "slot name is empty in '{s}'"
            )));
        }
        match prefix {
            "layers" => Ok(Self::Layer(name.to_string())),
            "obs" => Ok(Self::ObsColumn(name.to_string())),
            "var" => Ok(Self::VarColumn(name.to_string())),
            "obsm" => Ok(Self::Obsm(name.to_string())),
            "varm" => Ok(Self::Varm(name.to_string())),
            other => Err(ScxError::InvalidFormat(format!(
                "unknown slot group '{other}'; expected layers, obs, var, obsm, or varm"
            ))),
        }
    }

    /// Canonical HDF5 path for this slot (e.g. `"layers/normalized_log1p"`).
    pub fn hdf5_path(&self) -> String {
        match self {
            Self::Layer(n) => format!("layers/{n}"),
            Self::ObsColumn(n) => format!("obs/{n}"),
            Self::VarColumn(n) => format!("var/{n}"),
            Self::Obsm(n) => format!("obsm/{n}"),
            Self::Varm(n) => format!("varm/{n}"),
        }
    }
}

/// Parse a CLI patch spec: `"path/to/file.h5ad:slot1,slot2"`.
///
/// Returns `(source_path, slots)`.
pub fn parse_patch_spec(s: &str) -> Result<(PathBuf, Vec<SlotSelector>)> {
    let (path_str, slots_str) = s.split_once(':').ok_or_else(|| {
        ScxError::InvalidFormat(format!(
            "patch spec '{s}' must be 'file.h5ad:slot/name[,slot/name...]'"
        ))
    })?;
    if path_str.is_empty() {
        return Err(ScxError::InvalidFormat(
            "patch spec has empty file path".into(),
        ));
    }
    if slots_str.is_empty() {
        return Err(ScxError::InvalidFormat(format!(
            "patch spec '{s}' has no slots after ':'"
        )));
    }
    let path = PathBuf::from(path_str);
    let slots = slots_str
        .split(',')
        .map(|t| SlotSelector::parse(t.trim()))
        .collect::<Result<Vec<_>>>()?;
    Ok((path, slots))
}

// ---------------------------------------------------------------------------
// Core data structures (task 2)
// ---------------------------------------------------------------------------

/// How to handle a slot that already exists in the output file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictPolicy {
    #[default]
    Error,
    Skip,
    Overwrite,
}

/// One source file + which of its slots to copy.
#[derive(Debug, Clone)]
pub struct PatchSpec {
    pub source: PathBuf,
    /// SHA-256 of the source file; computed lazily on first use.
    pub source_sha256: Option<String>,
    pub slots: Vec<SlotSelector>,
    pub conflict: ConflictPolicy,
}

impl PatchSpec {
    pub fn new(source: PathBuf, slots: Vec<SlotSelector>, conflict: ConflictPolicy) -> Self {
        Self {
            source,
            source_sha256: None,
            slots,
            conflict,
        }
    }
}

/// First-derivation vs. additive-append merge.
#[derive(Debug, Clone)]
pub enum MergeMode {
    /// Copy base → output, then apply patches into the copy.
    Create { base: PathBuf, output: PathBuf },
    /// Open an existing merged file R/W and append new slots.
    Append { into: PathBuf },
}

/// Top-level manager for a merge operation.
#[derive(Debug)]
pub struct SlotPatchManager {
    pub mode: MergeMode,
    pub patches: Vec<PatchSpec>,
    pub tags: HashMap<String, String>,
}

impl SlotPatchManager {
    pub fn new(mode: MergeMode) -> Self {
        Self {
            mode,
            patches: Vec::new(),
            tags: HashMap::new(),
        }
    }

    pub fn add_patch(&mut self, patch: PatchSpec) {
        self.patches.push(patch);
    }

    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_layer() {
        let (path, slots) = parse_patch_spec("norm.h5ad:layers/normalized_log1p").unwrap();
        assert_eq!(path, PathBuf::from("norm.h5ad"));
        assert_eq!(slots, vec![SlotSelector::Layer("normalized_log1p".into())]);
    }

    #[test]
    fn parse_multi_slot() {
        let (path, slots) =
            parse_patch_spec("hvg.h5ad:var/highly_variable,var/dispersions").unwrap();
        assert_eq!(path, PathBuf::from("hvg.h5ad"));
        assert_eq!(
            slots,
            vec![
                SlotSelector::VarColumn("highly_variable".into()),
                SlotSelector::VarColumn("dispersions".into()),
            ]
        );
    }

    #[test]
    fn parse_mixed_slots() {
        let (_, slots) = parse_patch_spec("pca.h5ad:obsm/X_pca,varm/PCs").unwrap();
        assert_eq!(
            slots,
            vec![
                SlotSelector::Obsm("X_pca".into()),
                SlotSelector::Varm("PCs".into()),
            ]
        );
    }

    #[test]
    fn parse_obs_column() {
        let (_, slots) = parse_patch_spec("clusters.h5ad:obs/leiden").unwrap();
        assert_eq!(slots, vec![SlotSelector::ObsColumn("leiden".into())]);
    }

    #[test]
    fn error_missing_colon() {
        assert!(parse_patch_spec("norm.h5ad").is_err());
    }

    #[test]
    fn error_empty_path() {
        assert!(parse_patch_spec(":layers/norm").is_err());
    }

    #[test]
    fn error_empty_slots() {
        assert!(parse_patch_spec("norm.h5ad:").is_err());
    }

    #[test]
    fn error_unknown_group() {
        assert!(parse_patch_spec("f.h5ad:obsp/nn").is_err());
    }

    #[test]
    fn error_missing_name() {
        assert!(parse_patch_spec("f.h5ad:layers/").is_err());
    }

    #[test]
    fn hdf5_paths() {
        assert_eq!(SlotSelector::Layer("x".into()).hdf5_path(), "layers/x");
        assert_eq!(SlotSelector::ObsColumn("c".into()).hdf5_path(), "obs/c");
        assert_eq!(SlotSelector::Obsm("X_pca".into()).hdf5_path(), "obsm/X_pca");
    }
}
