use std::collections::HashMap;
use std::path::{Path, PathBuf};

use futures::StreamExt;

use crate::{
    bpcells::BpcellsDatasetReader,
    detect::{sniff, sniff_dir, Format},
    error::{Result, ScxError},
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::open_h5seurat,
    ir::DenseMatrix,
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
    Obsp(String),
    Uns(String),
}

impl SlotSelector {
    /// Parse a slot string: `"layers/name"`, `"obs/col"`, `"var/col"`,
    /// `"obsm/X_pca"`, `"varm/PCs"`, `"obsp/connectivities"`, or `"uns/key"`.
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
            "obsp" => Ok(Self::Obsp(name.to_string())),
            "uns" => Ok(Self::Uns(name.to_string())),
            "varp" => Err(ScxError::InvalidFormat(
                "varp slots are not supported by merge: varp is not carried by the \
                 streaming reader/writer pipeline (it exists only in the npy snapshot \
                 path). Supporting it requires extending DatasetReader across all \
                 readers."
                    .into(),
            )),
            other => Err(ScxError::InvalidFormat(format!(
                "unknown slot group '{other}'; expected layers, obs, var, obsm, varm, obsp, or uns"
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
            Self::Obsp(n) => format!("obsp/{n}"),
            Self::Uns(n) => format!("uns/{n}"),
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
            MergeMode::Append { into } => self.run_append(&into).await,
        }
    }

    async fn run_create(&mut self, base: &Path, output: &Path) -> Result<()> {
        let chunk_size = self.chunk_size;
        let base_sha256 = crate::provenance::sha256_file(base)?;
        let base_meta = read_base_meta(base).await?;

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

        apply_patches(
            &mut writer,
            &mut self.patches,
            &base_meta,
            &mut prov,
            chunk_size,
        )
        .await?;

        let prov_json = prov
            .to_json()
            .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
        writer.upsert_uns_provenance(&prov_json)?;

        Ok(())
    }

    async fn run_append(&mut self, into: &Path) -> Result<()> {
        let chunk_size = self.chunk_size;

        // Read existing provenance + obs/var index before opening R/W.
        // The reader handle is dropped when read_append_context returns, so
        // open_for_append below gets an exclusive handle with no concurrent reader.
        let (base_meta, mut prov) = read_append_context(into).await?;

        let mut writer = H5AdWriter::open_for_append(into)?;

        apply_patches(
            &mut writer,
            &mut self.patches,
            &base_meta,
            &mut prov,
            chunk_size,
        )
        .await?;

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
async fn read_base_meta(path: &Path) -> Result<BaseMeta> {
    let mut reader = H5AdReader::open(path, 1)?;
    let (n_obs, n_vars) = reader.shape();
    let obs = reader.obs().await?;
    let var = reader.var().await?;
    Ok(BaseMeta {
        n_obs,
        n_vars,
        obs_index: obs.index,
        var_index: var.index,
    })
}

/// Read existing provenance + obs/var index from a file that was previously
/// created by `run_create`.  The reader is dropped before `open_for_append`
/// is called so HDF5 never has two concurrent handles to the same file.
async fn read_append_context(path: &Path) -> Result<(BaseMeta, SlotProvenance)> {
    let mut reader = H5AdReader::open(path, 1)?;
    let (n_obs, n_vars) = reader.shape();
    let obs = reader.obs().await?;
    let var = reader.var().await?;
    let uns = reader.uns().await?;

    let base_meta = BaseMeta {
        n_obs,
        n_vars,
        obs_index: obs.index,
        var_index: var.index,
    };

    let prov = if let Some(prov_val) = uns.raw.get("scx_provenance") {
        SlotProvenance::from_json(prov_val)
            .map_err(|e| ScxError::InvalidFormat(format!("cannot parse scx_provenance: {e}")))?
    } else {
        return Err(ScxError::InvalidFormat(
            "target file has no scx_provenance — it was not created by `scx merge`; \
             use --base + --output to create a fresh merged file"
                .into(),
        ));
    };

    Ok((base_meta, prov))
}

/// Apply all patches in `patches` to `writer`, recording provenance for each
/// written slot into `prov`.
async fn apply_patches(
    writer: &mut H5AdWriter,
    patches: &mut [PatchSpec],
    base_meta: &BaseMeta,
    prov: &mut SlotProvenance,
    chunk_size: usize,
) -> Result<()> {
    for patch in patches.iter_mut() {
        let sha256 = match &patch.source_sha256 {
            Some(s) => s.clone(),
            None => {
                let s = crate::provenance::sha256_file(&patch.source)?;
                patch.source_sha256 = Some(s.clone());
                s
            }
        };

        let mut reader = open_patch_reader(&patch.source, chunk_size)?;
        let conflict = patch.conflict;
        let source_str = patch.source.to_string_lossy().into_owned();

        for slot in &patch.slots {
            let applied = match slot {
                SlotSelector::Layer(name) => {
                    apply_layer_patch(
                        writer,
                        name,
                        reader.as_mut(),
                        base_meta,
                        conflict,
                        chunk_size,
                    )
                    .await?
                }
                SlotSelector::ObsColumn(name) => {
                    apply_obs_column_patch(writer, name, reader.as_mut(), base_meta, conflict)
                        .await?
                }
                SlotSelector::VarColumn(name) => {
                    apply_var_column_patch(writer, name, reader.as_mut(), base_meta, conflict)
                        .await?
                }
                SlotSelector::Obsm(name) => {
                    apply_obsm_patch(writer, name, reader.as_mut(), base_meta, conflict).await?
                }
                SlotSelector::Varm(name) => {
                    apply_varm_patch(writer, name, reader.as_mut(), base_meta, conflict).await?
                }
                SlotSelector::Obsp(name) => {
                    apply_obsp_patch(
                        writer,
                        name,
                        reader.as_mut(),
                        base_meta,
                        conflict,
                        chunk_size,
                    )
                    .await?
                }
                SlotSelector::Uns(name) => {
                    apply_uns_patch(writer, name, reader.as_mut(), conflict).await?
                }
            };
            if applied {
                prov.add_slot(slot.hdf5_path(), source_str.clone(), sha256.clone());
            }
        }
    }
    Ok(())
}

/// Open a reader for any supported format using content-based detection.
fn open_patch_reader(path: &Path, chunk_size: usize) -> Result<Box<dyn DatasetReader + Send>> {
    let fmt = sniff_dir(path).or_else(|| sniff(path));
    Ok(match fmt {
        Some(Format::BPCells) => Box::new(BpcellsDatasetReader::open(path, chunk_size)?),
        Some(Format::H5Seurat) => open_h5seurat(path, chunk_size, None, None)?,
        _ => Box::new(H5AdReader::open(path, chunk_size)?),
    })
}

/// Stream one named layer from `reader` into `writer`, with conflict handling.
/// Returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_layer_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
    chunk_size: usize,
) -> Result<bool> {
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
                return Ok(false);
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

    Ok(true)
}

/// Patch a single obs column; returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_obs_column_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
) -> Result<bool> {
    if !check_column_conflict(writer, "obs", name, conflict)? {
        return Ok(false);
    }
    let patch_obs = reader.obs().await?;
    let col = patch_obs
        .columns
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| {
            ScxError::InvalidFormat(format!("column '{name}' not found in patch obs"))
        })?;
    let reindex = align::build_obs_reindex(&base_meta.obs_index, &patch_obs.index)?;
    writer.add_obs_column(name, &align::reindex_column(&col.data, &reindex))?;
    Ok(true)
}

/// Patch a single var column; returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_var_column_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
) -> Result<bool> {
    if !check_column_conflict(writer, "var", name, conflict)? {
        return Ok(false);
    }
    let patch_var = reader.var().await?;
    let col = patch_var
        .columns
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| {
            ScxError::InvalidFormat(format!("column '{name}' not found in patch var"))
        })?;
    let reindex = align::build_var_reindex(&base_meta.var_index, &patch_var.index)?;
    writer.add_var_column(name, &align::reindex_column(&col.data, &reindex))?;
    Ok(true)
}

/// Patch a single obsm entry; returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_obsm_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
) -> Result<bool> {
    if !check_dict_conflict(writer, "obsm", name, conflict)? {
        return Ok(false);
    }
    let patch_obs = reader.obs().await?;
    let obsm = reader.obsm().await?;
    let mat = obsm.map.get(name).ok_or_else(|| {
        ScxError::InvalidFormat(format!("obsm entry '{name}' not found in patch"))
    })?;
    let reindex = align::build_obs_reindex(&base_meta.obs_index, &patch_obs.index)?;
    writer.add_obsm_entry(name, &reindex_dense_rows(mat, &reindex))?;
    Ok(true)
}

/// Patch a single varm entry; returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_varm_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
) -> Result<bool> {
    if !check_dict_conflict(writer, "varm", name, conflict)? {
        return Ok(false);
    }
    let patch_var = reader.var().await?;
    let varm = reader.varm().await?;
    let mat = varm.map.get(name).ok_or_else(|| {
        ScxError::InvalidFormat(format!("varm entry '{name}' not found in patch"))
    })?;
    let reindex = align::build_var_reindex(&base_meta.var_index, &patch_var.index)?;
    writer.add_varm_entry(name, &reindex_dense_rows(mat, &reindex))?;
    Ok(true)
}

/// Stream one named obsp (cell×cell) matrix from `reader` into `writer`.
/// Mirrors the layer patch: requires the patch obsp to be `n_obs × n_obs` and
/// streams it directly (no reindex — assumes matching obs order, as layers do).
/// Returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_obsp_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    base_meta: &BaseMeta,
    conflict: ConflictPolicy,
    chunk_size: usize,
) -> Result<bool> {
    let slot_path = format!("obsp/{name}");
    if writer.group_exists(&slot_path) {
        match conflict {
            ConflictPolicy::Error => {
                return Err(ScxError::InvalidFormat(format!(
                    "slot '{slot_path}' already exists (use --on-conflict skip|overwrite)"
                )));
            }
            ConflictPolicy::Skip => {
                tracing::info!(slot = slot_path, "skipping existing slot (conflict=skip)");
                return Ok(false);
            }
            ConflictPolicy::Overwrite => {
                writer.unlink_child("obsp", name)?;
            }
        }
    }

    let obsp_metas = reader.obsp_metas().await?;
    let meta = obsp_metas
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| ScxError::InvalidFormat(format!("obsp '{name}' not found in patch source")))?
        .clone();

    if meta.shape.0 != base_meta.n_obs || meta.shape.1 != base_meta.n_obs {
        return Err(ScxError::InvalidFormat(format!(
            "shape mismatch for obsp '{name}': patch is {}×{}, base obsp must be {}×{}",
            meta.shape.0, meta.shape.1, base_meta.n_obs, base_meta.n_obs
        )));
    }

    writer.begin_sparse("obsp", name, &meta).await?;
    let mut stream = Box::pin(reader.obsp_stream(&meta, chunk_size));
    while let Some(chunk) = stream.next().await {
        writer.write_sparse_chunk(&chunk?).await?;
    }
    writer.end_sparse().await?;

    Ok(true)
}

/// Copy a single top-level `uns` entry from the patch into `writer`.
/// uns is global metadata (no obs/var alignment). Scalar and nested-dict values
/// are written natively; arrays/nulls are skipped (same as the conversion path's
/// uns encoding). Returns `Ok(true)` if written, `Ok(false)` if skipped.
async fn apply_uns_patch(
    writer: &mut H5AdWriter,
    name: &str,
    reader: &mut dyn DatasetReader,
    conflict: ConflictPolicy,
) -> Result<bool> {
    if !check_dict_conflict(writer, "uns", name, conflict)? {
        return Ok(false);
    }
    let uns = reader.uns().await?;
    let value = uns.raw.get(name).ok_or_else(|| {
        ScxError::InvalidFormat(format!("uns entry '{name}' not found in patch source"))
    })?;
    writer.add_uns_entry(name, value)?;
    Ok(true)
}

/// Conflict-policy gating for obs/var column slots.
/// Returns `Ok(true)` to proceed, `Ok(false)` to skip, `Err` on conflict=error.
fn check_column_conflict(
    writer: &mut H5AdWriter,
    group: &str,
    name: &str,
    conflict: ConflictPolicy,
) -> Result<bool> {
    if !writer.child_exists(group, name) {
        return Ok(true);
    }
    match conflict {
        ConflictPolicy::Error => Err(ScxError::InvalidFormat(format!(
            "slot '{group}/{name}' already exists (use --on-conflict skip|overwrite)"
        ))),
        ConflictPolicy::Skip => {
            tracing::info!(
                slot = format!("{group}/{name}"),
                "skipping existing slot (conflict=skip)"
            );
            Ok(false)
        }
        ConflictPolicy::Overwrite => {
            writer.unlink_child(group, name)?;
            Ok(true)
        }
    }
}

/// Conflict-policy gating for dict entries (obsm/varm datasets).
/// Returns `Ok(true)` to proceed, `Ok(false)` to skip, `Err` on conflict=error.
fn check_dict_conflict(
    writer: &mut H5AdWriter,
    group: &str,
    name: &str,
    conflict: ConflictPolicy,
) -> Result<bool> {
    let exists = writer.child_exists(group, name);
    if !exists {
        return Ok(true);
    }
    let slot_path = format!("{group}/{name}");
    match conflict {
        ConflictPolicy::Error => Err(ScxError::InvalidFormat(format!(
            "slot '{slot_path}' already exists (use --on-conflict skip|overwrite)"
        ))),
        ConflictPolicy::Skip => {
            tracing::info!(slot = slot_path, "skipping existing slot (conflict=skip)");
            Ok(false)
        }
        ConflictPolicy::Overwrite => {
            writer.unlink_child(group, name)?;
            Ok(true)
        }
    }
}

/// Reindex rows of a dense matrix according to a row-reindex map.
/// Rows absent from the patch (None entries) are zero-filled.
fn reindex_dense_rows(mat: &DenseMatrix, reindex: &[Option<usize>]) -> DenseMatrix {
    let (_, ncols) = mat.shape;
    let nrows = reindex.len();
    let mut data = vec![0.0f64; nrows * ncols];
    for (out_row, src_row) in reindex.iter().enumerate() {
        if let Some(i) = src_row {
            let src = i * ncols;
            let dst = out_row * ncols;
            data[dst..dst + ncols].copy_from_slice(&mat.data[src..src + ncols]);
        }
    }
    DenseMatrix {
        shape: (nrows, ncols),
        data,
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
