use std::path::PathBuf;

use serde::Deserialize;
use scx_core::merge::{ConflictPolicy, MergeMode, PatchSpec, SlotPatchManager, parse_patch_spec};

#[derive(Debug, Deserialize)]
struct MergeConfig {
    base: Option<String>,
    output: Option<String>,
    into: Option<String>,
    #[serde(default)]
    patches: Vec<PatchConfigEntry>,
    #[serde(default)]
    on_conflict: ConflictPolicyStr,
    #[serde(default)]
    tags: std::collections::HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct PatchConfigEntry {
    source: String,
    slots: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum ConflictPolicyStr {
    #[default]
    Error,
    Skip,
    Overwrite,
}

impl From<ConflictPolicyStr> for ConflictPolicy {
    fn from(s: ConflictPolicyStr) -> Self {
        match s {
            ConflictPolicyStr::Error => ConflictPolicy::Error,
            ConflictPolicyStr::Skip => ConflictPolicy::Skip,
            ConflictPolicyStr::Overwrite => ConflictPolicy::Overwrite,
        }
    }
}

pub struct MergeArgs {
    /// Source base file (create mode).
    pub base: Option<String>,
    /// Output path (create mode).
    pub output: Option<String>,
    /// Existing merged file to append into (append mode).
    pub into: Option<String>,
    /// Patch specs: "file.h5ad:slot/name[,...]"
    pub patches: Vec<String>,
    /// Conflict resolution policy.
    pub on_conflict: String,
    /// key=value tags to embed in provenance.
    pub tags: Vec<String>,
    /// Optional path to a YAML config file (overrides above flags).
    pub config: Option<String>,
    /// Streaming chunk size.
    pub chunk_size: usize,
}

pub async fn run_merge(args: MergeArgs) -> anyhow::Result<()> {
    let mut mgr = if let Some(cfg_path) = &args.config {
        build_from_config(cfg_path, args.chunk_size)?
    } else {
        build_from_flags(&args)?
    };

    mgr.run().await.map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!("merge complete");
    Ok(())
}

fn build_from_flags(args: &MergeArgs) -> anyhow::Result<SlotPatchManager> {
    let conflict = parse_conflict(&args.on_conflict)?;
    let mode = resolve_mode(args.base.as_deref(), args.output.as_deref(), args.into.as_deref())?;

    let mut mgr = SlotPatchManager::new(mode);
    mgr.chunk_size = args.chunk_size;

    for raw in &args.patches {
        let (source, slots) =
            parse_patch_spec(raw).map_err(|e| anyhow::anyhow!("invalid patch '{raw}': {e}"))?;
        mgr.add_patch(PatchSpec::new(source, slots, conflict));
    }

    for kv in &args.tags {
        let (k, v) = kv.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("--tag must be key=value, got '{kv}'")
        })?;
        mgr.add_tag(k, v);
    }

    Ok(mgr)
}

fn build_from_config(path: &str, chunk_size: usize) -> anyhow::Result<SlotPatchManager> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read config '{path}': {e}"))?;
    let cfg: MergeConfig = serde_yaml::from_str(&src)
        .map_err(|e| anyhow::anyhow!("invalid config '{path}': {e}"))?;

    let conflict: ConflictPolicy = cfg.on_conflict.into();
    let mode = resolve_mode(cfg.base.as_deref(), cfg.output.as_deref(), cfg.into.as_deref())?;

    let mut mgr = SlotPatchManager::new(mode);
    mgr.chunk_size = chunk_size;

    for entry in cfg.patches {
        let source = PathBuf::from(&entry.source);
        let slots = entry
            .slots
            .iter()
            .map(|s| {
                scx_core::merge::SlotSelector::parse(s)
                    .map_err(|e| anyhow::anyhow!("invalid slot '{s}': {e}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        mgr.add_patch(PatchSpec::new(source, slots, conflict));
    }

    for (k, v) in cfg.tags {
        mgr.add_tag(k, v);
    }

    Ok(mgr)
}

fn resolve_mode(
    base: Option<&str>,
    output: Option<&str>,
    into: Option<&str>,
) -> anyhow::Result<MergeMode> {
    match (base, output, into) {
        (Some(b), Some(o), None) => Ok(MergeMode::Create {
            base: PathBuf::from(b),
            output: PathBuf::from(o),
        }),
        (None, None, Some(i)) => Ok(MergeMode::Append {
            into: PathBuf::from(i),
        }),
        (None, None, None) => {
            anyhow::bail!("specify either --base + --output (create mode) or --into (append mode)")
        }
        _ => anyhow::bail!(
            "--base/--output and --into are mutually exclusive; \
             use --base + --output for create mode, --into for append mode"
        ),
    }
}

fn parse_conflict(s: &str) -> anyhow::Result<ConflictPolicy> {
    match s {
        "error" => Ok(ConflictPolicy::Error),
        "skip" => Ok(ConflictPolicy::Skip),
        "overwrite" => Ok(ConflictPolicy::Overwrite),
        other => anyhow::bail!("unknown --on-conflict value '{other}'; use error, skip, or overwrite"),
    }
}
