# scx merge

## Motivation

A single-cell analysis pipeline produces one h5ad per stage: raw counts,
normalised expression, HVG flags, PCA embeddings, cluster labels.
`scx merge` assembles those outputs into one portable container without
copying matrix data unnecessarily.

The container is a standard h5ad that any tool can open. Every slot carries a
provenance record — source file, SHA-256, timestamp — so the merged file is
self-describing.

---

## Modes

### Create mode

Copies the base file to the output path, then patches slots from one or more
source files into the copy.

```sh
scx merge \
  --base   data/source.h5ad \
  --output results/merged.h5ad \
  --patch  results/normalized.h5ad:layers/normalized \
  --patch  results/hvg.h5ad:var/highly_variable,var/dispersions \
  --patch  results/pca.h5ad:obsm/X_pca,varm/PCs \
  --tag    pipeline=snakemake \
  --tag    genome=GRCh38
```

`--base` is the authoritative source of X, obs, and var. It is never
modified — the output is an independent copy.

### Append mode

Opens an existing merged file in place and adds new slots.

```sh
scx merge \
  --into   results/merged.h5ad \
  --patch  results/umap.h5ad:obsm/X_umap \
  --patch  results/clusters.h5ad:obs/leiden
```

The target must have been created by `scx merge` (it must contain
`uns["scx_provenance"]`). Provenance is updated in-place.

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base <path>` | — | Base file (create mode; mutually exclusive with `--into`) |
| `--output <path>` | — | Output path (create mode) |
| `--into <path>` | — | Existing merged file to append into (append mode) |
| `--patch <spec>` | repeatable | Patch spec: `file.h5ad:slot[,slot…]` |
| `--on-conflict` | `error` | What to do when a slot already exists: `error`, `skip`, `overwrite` |
| `--tag <key=value>` | repeatable | Free-form tags written into provenance |
| `--config <yaml>` | — | YAML config file (overrides all other flags) |
| `--chunk-size` | `5000` | Rows per streaming chunk for layer patches |

---

## Slot specifier syntax

```
file.h5ad:group/name[,group/name…]
```

| Prefix | What is patched | Shape |
|--------|-----------------|-------|
| `layers/<name>` | Named sparse layer | n_obs × n_vars |
| `obs/<col>` | Single obs metadata column | n_obs |
| `var/<col>` | Single var metadata column | n_vars |
| `obsm/<name>` | Dense embedding matrix | n_obs × k |
| `varm/<name>` | Dense gene embedding | n_vars × k |

Patches from any supported input format (h5ad, h5seurat, BPCells directory)
are accepted — the source format is auto-detected.

---

## Obs/var alignment

The patch's obs index must be a subset of the base obs index. Rows missing
from the patch are NA-filled: `NaN` for floats, `0` for integers, `false` for
booleans, empty string for strings, code `0` for categoricals.

Var alignment works identically on the var axis.

This means a patch covering only HVG-selected genes can be applied to a base
covering all genes — the non-HVG rows are zero-filled.

---

## Conflict policy

| `--on-conflict` | Behaviour |
|-----------------|-----------|
| `error` (default) | Abort immediately if the slot already exists |
| `skip` | Leave the existing slot unchanged, continue |
| `overwrite` | Delete the existing slot and write the new one |

---

## Config file

All flags can be driven from a YAML config, which is easier to manage in
pipelines than long command lines.

```yaml
# merge.yaml
base:   data/source.h5ad
output: results/merged.h5ad

patches:
  - source: results/normalized.h5ad
    slots:  [layers/normalized]
  - source: results/hvg.h5ad
    slots:  [var/highly_variable, var/dispersions]
  - source: results/pca.h5ad
    slots:  [obsm/X_pca, varm/PCs]

on_conflict: error

tags:
  pipeline: snakemake
  genome:   GRCh38
```

```sh
scx merge --config merge.yaml
```

CLI flags and `--config` are mutually exclusive — `--config` wins if both
are supplied.

---

## Provenance

Every merged file contains `uns["scx_provenance"]` — a JSON block recording
the base file anchor and every slot that was patched in:

```json
{
  "scx_version": "0.2.0",
  "base": {
    "path": "data/source.h5ad",
    "sha256": "a3f9c1d2…",
    "n_obs": 8312,
    "n_vars": 33694
  },
  "slots": {
    "layers/normalized": {
      "source_path": "results/normalized.h5ad",
      "sha256": "b1c2e3f4…",
      "added_at": "2026-04-25T10:04:12Z"
    },
    "obsm/X_pca": {
      "source_path": "results/pca.h5ad",
      "sha256": "c8d9e0f1…",
      "added_at": "2026-04-25T10:04:13Z"
    }
  },
  "tags": {
    "pipeline": "snakemake",
    "genome": "GRCh38"
  }
}
```

`scx inspect` renders the provenance block as a dedicated section above `uns`:

```
provenance
  scx_version    0.2.0
  base.path      data/source.h5ad
  base.sha256    a3f9c1d2ef05…
  base.n_obs     8312

provenance slots (2 patched)
  layers/normalized    results/normalized.h5ad  [b1c2e3f4…]  2026-04-25T10:04:12Z
  obsm/X_pca           results/pca.h5ad         [c8d9e0f1…]  2026-04-25T10:04:13Z
```

Skipped slots (conflict=skip) are not recorded. Overwritten slots show the
new source.

---

## Snakemake integration

```python
# Snakefile

STAGES = ["normalized", "hvg", "pca", "clusters"]

rule merge:
    input:
        base    = "data/source.h5ad",
        patches = expand("results/{stage}.h5ad", stage=STAGES),
    output:
        "results/merged.h5ad"
    params:
        patches = lambda wc, input: " ".join([
            "--patch results/normalized.h5ad:layers/normalized",
            "--patch results/hvg.h5ad:var/highly_variable,var/dispersions",
            "--patch results/pca.h5ad:obsm/X_pca,varm/PCs",
            "--patch results/clusters.h5ad:obs/leiden",
        ])
    shell:
        """
        scx merge \
          --base {input.base} \
          --output {output} \
          {params.patches} \
          --tag snakemake_rule=merge \
          --tag pipeline_version=0.2.0
        """

rule add_umap:
    input:  "results/merged.h5ad"
    output: touch("results/.umap_appended")
    shell:
        """
        scx merge \
          --into results/merged.h5ad \
          --patch results/umap.h5ad:obsm/X_umap
        """
```

---

## Implementation

```
scx-core/src/merge/mod.rs       — SlotPatchManager, MergeMode, PatchSpec, SlotSelector
scx-core/src/merge/align.rs     — build_obs_reindex, reindex_column, unify_categorical_levels
scx-core/src/merge/provenance.rs — SlotProvenance, BaseAnchor, SlotEntry
scx-cli/src/cmd_merge.rs        — CLI argument parsing, YAML config loading
scx-cli/src/main.rs             — Merge subcommand wiring
```
