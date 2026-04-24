use std::collections::HashMap;
use std::path::{Path, PathBuf};

use futures::StreamExt;

use crate::{
    bpcells::BpcellsDatasetReader,
    detect::{sniff, sniff_dir, Format},
    error::{Result, ScxError},
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::open_h5seurat,
    stream::{DatasetReader, DatasetWriter},
};

pub mod align;
pub mod provenance;

use provenance::{BaseAnchor, SlotProvenance};

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
    pub chunk_size: usize,
}

/// Obs/var axis metadata loaded from the base file before patching.
#[derive(Debug, Clone)]
pub struct BaseMeta {
    pub n_obs: usize,
    pub n_vars: usize,
    pub obs_index: Vec<String>,
    pub var_index: Vec<String>,
}

impl SlotPatchManager {
    pub fn new(mode: MergeMode) -> Self {
        Self {
            mode,
            patches: Vec::new(),
            tags: HashMap::new(),
            chunk_size: 5000,
        }
    }

    pub fn add_patch(&mut self, patch: PatchSpec) {
        self.patches.push(patch);
    }

    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Execute the merge operation.
    pub async fn run(&mut self) -> Result<()> {
        match self.mode.clone() {
            MergeMode::Create { base, output } => self.run_create(&base, &output).await,
            MergeMode::Append { into: _ } => Err(ScxError::InvalidFormat(
                "append mode is not yet implemented (coming in v0.2.0)".into(),
            )),
        }
    }

    async fn run_create(&mut self, base: &Path, output: &Path) -> Result<()> {
        let chunk_size = self.chunk_size;
        let base_sha256 = crate::provenance::sha256_file(base)?;
        let base_meta = read_base_meta(base)?;

        std::fs::copy(base, output)?;

        let mut writer = H5AdWriter::open_for_append(output)?;

        let mut prov = SlotProvenance::new(BaseAnchor {
            path: base.to_string_lossy().into_owned(),
            sha256: base_sha256,
            n_obs: base_meta.n_obs,
            n_vars: base_meta.n_vars,
        });
        for (k, v) in &self.tags {
            prov.set_tag(k, v);
        }

        for patch in &mut self.patches {
            let patch_sha256 = match &patch.source_sha256 {
                Some(s) => s.clone(),
                None => {
                    let sha = crate::provenance::sha256_file(&patch.source)?;
                    patch.source_sha256 = Some(sha.clone());
                    sha
                }
            };

            let mut reader = open_patch_reader(&patch.source, chunk_size)?;

            for slot in patch.slots.clone() {
                match &slot {
                    SlotSelector::Layer(name) => {
                        apply_layer_patch(
                            &mut writer,
                            name,
                            reader.as_mut(),
                            &base_meta,
                            patch.conflict,
                            chunk_size,
                        )
                        .await?;
                        prov.add_slot(slot.hdf5_path(), &patch.source, &patch_sha256);
                    }
                    other => {
                        return Err(ScxError::InvalidFormat(format!(
                            "slot type '{}' not yet supported (coming in v0.2.0 chunk 3)",
                            other.hdf5_path()
                        )));
                    }
                }
            }
        }

        let prov_json = prov
            .to_json()
            .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
        writer.upsert_uns_provenance(&prov_json)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Read obs/var index and shape from an h5ad file without streaming X.
fn read_base_meta(path: &Path) -> Result<BaseMeta> {
    let mut reader = H5AdReader::open(path, 1)?;
    let (n_obs, n_vars) = reader.shape();
    // Use a tiny runtime just for the metadata reads (obs/var are cheap).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
    let (obs, var) = rt.block_on(async {
        let obs = reader.obs().await?;
        let var = reader.var().await?;
        Ok::<_, ScxError>((obs, var))
    })?;
    Ok(BaseMeta {
        n_obs,
        n_vars,
        obs_index: obs.index,
        var_index: var.index,
    })
}

/// Open a reader for any supported format using content-based detection.
fn open_patch_reader(
    path: &Path,
    chunk_size: usize,
) -> Result<Box<dyn DatasetReader + Send>> {
    let fmt = sniff_dir(path).or_else(|| sniff(path));
    Ok(match fmt {
        Some(Format::BPCells) => Box::new(BpcellsDatasetReader::open(path, chunk_size)?),
        Some(Format::H5Seurat) => open_h5seurat(path, chunk_size, None, None)?,
        _ => Box::new(H5AdReader::open(path, chunk_size)?),
    })
}

/// Stream one named layer from `reader` into `writer`, with conflict handling.
///
/// For Chunk 2 the patch must have the same shape as the base; obs-order
/// reindexing for mis-ordered patches comes in Chunk 3.
async fn apply_layer_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
    chunk_size: usize,
) -> Result<()> {
    let (patch_n_obs, patch_n_vars) = reader.shape();
    if patch_n_obs != base_meta.n_obs || patch_n_vars != base_meta.n_vars {
        return Err(ScxError::InvalidFormat(format!(
            "shape mismatch for layer '{name}': patch is {patch_n_obs}×{patch_n_vars}, \
             base is {}×{}",
            base_meta.n_obs, base_meta.n_vars
        )));
    }

    let slot_path = format!("layers/{name}");
    if writer.group_exists(&slot_path) {
        match conflict {
            ConflictPolicy::Error => {
                return Err(ScxError::InvalidFormat(format!(
                    "slot '{slot_path}' already exists (use --on-conflict skip|overwrite)"
                )));
            }
            ConflictPolicy::Skip => {
                tracing::info!(slot = slot_path, "skipping existing slot (conflict=skip)");
                return Ok(());
            }
            ConflictPolicy::Overwrite => {
                writer.unlink_child("layers", name)?;
            }
        }
    }

    let layer_metas = reader.layer_metas().await?;
    let meta = layer_metas
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| {
            ScxError::InvalidFormat(format!("layer '{name}' not found in patch source"))
        })?
        .clone();

    writer.begin_sparse("layers", name, &meta).await?;
    let mut stream = Box::pin(reader.layer_stream(&meta, chunk_size));
    while let Some(chunk) = stream.next().await {
        writer.write_sparse_chunk(&chunk?).await?;
    }
    writer.end_sparse().await?;

    Ok(())
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
