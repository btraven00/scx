# scx merge + provenance

## Motivation

Benchmarking pipelines (e.g. Snakemake) produce one authoritative immutable source
h5ad, then separate stage outputs — normalization, HVG selection, filtering, PCA,
neighbor graphs, etc. `scx merge` assembles these into a single h5ad container
suitable for downstream `scx convert` to BPCells / H5Seurat.

The merged file carries a full audit trail in `uns["scx_provenance"]` so every
downstream analysis knows exactly where each slot came from and which version of
SCX assembled it.

---

## CLI

```sh
scx merge \
  --base source.h5ad \
  --patch normalized.h5ad:layers/normalized \
  --patch hvg.h5ad:var/highly_variable,var/dispersions \
  --patch pca.h5ad:obsm/X_pca,varm/PCs \
  --patch graph.h5ad:obsp/connectivities,obsp/distances \
  --patch meta.h5ad:uns/hvg_params \
  --tag snakemake_rule=merge_stages \
  --tag pipeline_version=0.4.1 \
  --tag genome=GRCh38 \
  --on-conflict error \
  --output merged.h5ad
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | required | Authoritative source — provides X, obs, var. Never modified. |
| `--patch file:slot[,slot…]` | repeatable | File to pull slots from. Omit slot spec to take all non-base slots. |
| `--tag key=value` | repeatable | Free-form pipeline tags written into provenance. |
| `--on-conflict` | `error` | What to do when a slot already exists: `error`, `skip`, `overwrite`. |
| `--chunk-size` | `5000` | Rows per streaming chunk for sparse matrix copy. |
| `--output` | required | Output h5ad path (new file; never overwrites base). |

---

## Slot specifier syntax

A slot spec has the form `prefix/name`:

| Prefix | IR slot | Notes |
|--------|---------|-------|
| `layers/` | `Layers` | Sparse matrix (n_obs × n_vars) |
| `obsm/` | `Embeddings` | Dense embedding (n_obs × k) |
| `varm/` | `Varm` | Dense gene embedding (n_vars × k) |
| `obsp/` | `Obsp` | Sparse cell-cell graph (n_obs × n_obs) |
| `varp/` | `Varp` | Sparse gene-gene matrix (n_vars × n_vars) |
| `var/col` | `VarTable` column | Single metadata column |
| `obs/col` | `ObsTable` column | Single metadata column |
| `uns/key` | `UnsTable` subtree | Recursive JSON subtree |

Multiple slots from the same file are comma-separated:
`hvg.h5ad:var/highly_variable,var/dispersions,var/dispersions_norm`

---

## Alignment

### obs alignment
Each patch file's `obs.index` must be a subset of the base `obs.index`. Rows are
reindexed to match the base order. This handles filtered patches (e.g. a file
produced after QC that drops low-quality cells).

### var alignment
Each patch file's `var.index` must be a subset of the base `var.index`. Missing
rows in boolean/float var columns are filled with `false` / `NaN`. This handles
HVG-selection patches that only cover the selected gene subset.

### Conflict detection
Before writing, SCX checks whether a requested target slot already exists in the
output (populated by an earlier patch). Behaviour is controlled by `--on-conflict`:
- `error` (default) — abort with a clear message naming the conflicting slot and both source files.
- `skip` — keep the first-written slot, emit a warning.
- `overwrite` — replace with the later patch, emit a warning.

---

## Provenance schema

Written to `uns["scx_provenance"]` in the output file. Readable in Python via
`adata.uns["scx_provenance"]` and in R via `obj@misc$scx_provenance`.

```json
{
  "scx_version": "0.3.0",
  "merged_at": "2026-04-17T14:32:00Z",
  "base": {
    "path": "data/source.h5ad",
    "sha256": "a3f9c1d2…",
    "n_obs": 8000,
    "n_vars": 33000
  },
  "patches": [
    {
      "path": "results/normalized.h5ad",
      "sha256": "b1c2e3f4…",
      "slots": ["layers/normalized"]
    },
    {
      "path": "results/hvg.h5ad",
      "sha256": "d4e5f6a7…",
      "slots": ["var/highly_variable", "var/dispersions"]
    },
    {
      "path": "results/pca.h5ad",
      "sha256": "c8d9e0f1…",
      "slots": ["obsm/X_pca", "varm/PCs"]
    },
    {
      "path": "results/graph.h5ad",
      "sha256": "e2f3a4b5…",
      "slots": ["obsp/connectivities", "obsp/distances"]
    }
  ],
  "tags": {
    "snakemake_rule": "merge_stages",
    "pipeline_version": "0.4.1",
    "genome": "GRCh38"
  }
}
```

### Fields

| Field | Source |
|-------|--------|
| `scx_version` | `env!("CARGO_PKG_VERSION")` at compile time |
| `merged_at` | UTC timestamp at merge start |
| `base.sha256` | SHA-256 of the base file bytes — the immutable anchor |
| `patch.sha256` | SHA-256 of each patch file |
| `base.n_obs / n_vars` | Read from file shape, not recomputed |
| `tags` | `--tag key=value` flags, verbatim |

SHA-256 is computed on the raw file bytes and is always included. For large files
the cost is negligible because the OS page cache is warm after the data read.

---

## Implementation sketch

```
scx-core/src/merge.rs      — MergeSpec, SlotSpec parser, alignment logic, provenance builder
scx-core/src/hash.rs       — sha256_file(path) -> String
scx-cli/src/main.rs        — Merge { base, patches, tags, on_conflict, output } arm
```

`merge.rs` reads the base with `H5AdReader`, then for each patch opens a second
`H5AdReader`, extracts the requested slots, aligns obs/var indices, and writes
everything through a single `H5AdWriter`. Provenance is assembled incrementally
and written as the final `uns` entry before `finalize()`.
