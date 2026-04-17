# Provenance

SCX embeds provenance in every output artifact so downstream analyses know
exactly where data came from and which tool version produced it.

---

## Convert provenance

Every `scx convert` run produces two provenance outputs.

### Artifact — `uns["scx_provenance"]`

A deterministic block baked into the output file. Contains only fields that
are a pure function of the inputs, so the artifact is byte-reproducible:
running the same conversion twice with the same inputs produces identical
output bytes.

```json
{
  "scx_version": "0.1.0",
  "source": {
    "path": "data/pbmc3k.h5seurat",
    "url": "https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat",
    "sha256": "703a1b4a…"
  },
  "n_obs": 2700,
  "n_vars": 13714
}
```

`url` is omitted when `--source-url` is not provided.

### Sidecar — `<output>.prov.json`

Written alongside the output file. Adds non-deterministic fields
(timestamp, output path, output SHA256) that would break reproducibility
if baked into the artifact:

```json
{
  "scx_version": "0.1.0",
  "converted_at": "2026-04-17T09:26:53Z",
  "source": {
    "path": "data/pbmc3k.h5seurat",
    "url": "https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat",
    "sha256": "703a1b4a…"
  },
  "output": {
    "path": "results/pbmc3k.h5ad",
    "sha256": "27242b60…",
    "n_obs": 2700,
    "n_vars": 13714
  }
}
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--source-url <url>` | Canonical origin URL of the source file (optional) |

### hapiq integration

`--source-url` is designed for use with
[hapiq](https://github.com/btraven00/hapiq), a declarative downloader.
The typical pipeline is:

```bash
hapiq get https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat \
  -o data/pbmc3k.h5seurat

scx convert data/pbmc3k.h5seurat results/pbmc3k.h5ad \
  --source-url https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat
```

The artifact then carries the full lineage: origin URL + source SHA256 +
scx version. Anyone with the artifact can re-derive the source file
independently.

---

## Merge provenance (planned)

`scx merge` assembles multiple pipeline stage outputs into a single h5ad.
The merged file will carry a full audit trail covering the base file and
every patch applied.

### CLI

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
| `--base` | required | Authoritative source — provides X, obs, var |
| `--patch file:slot[,slot…]` | repeatable | File to pull slots from |
| `--tag key=value` | repeatable | Free-form pipeline tags written into provenance |
| `--on-conflict` | `error` | `error`, `skip`, or `overwrite` |
| `--chunk-size` | `5000` | Rows per streaming chunk |
| `--output` | required | Output h5ad path |

### Slot specifier syntax

| Prefix | Slot | Notes |
|--------|------|-------|
| `layers/` | `Layers` | Sparse matrix (n_obs × n_vars) |
| `obsm/` | `Embeddings` | Dense embedding (n_obs × k) |
| `varm/` | `Varm` | Dense gene embedding (n_vars × k) |
| `obsp/` | `Obsp` | Sparse cell-cell graph (n_obs × n_obs) |
| `varp/` | `Varp` | Sparse gene-gene matrix (n_vars × n_vars) |
| `var/col` | `VarTable` column | Single metadata column |
| `obs/col` | `ObsTable` column | Single metadata column |
| `uns/key` | `UnsTable` subtree | Recursive JSON subtree |

### Alignment

**obs:** each patch's `obs.index` must be a subset of the base. Rows are
reindexed to match base order — handles filtered patches.

**var:** each patch's `var.index` must be a subset of the base. Missing
entries in boolean/float columns are filled with `false` / `NaN` —
handles HVG-selection patches covering only selected genes.

### Schema

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
    { "path": "results/normalized.h5ad", "sha256": "b1c2e3f4…", "slots": ["layers/normalized"] },
    { "path": "results/hvg.h5ad",        "sha256": "d4e5f6a7…", "slots": ["var/highly_variable", "var/dispersions"] },
    { "path": "results/pca.h5ad",        "sha256": "c8d9e0f1…", "slots": ["obsm/X_pca", "varm/PCs"] },
    { "path": "results/graph.h5ad",      "sha256": "e2f3a4b5…", "slots": ["obsp/connectivities", "obsp/distances"] }
  ],
  "tags": {
    "snakemake_rule": "merge_stages",
    "pipeline_version": "0.4.1",
    "genome": "GRCh38"
  }
}
```

| Field | Source |
|-------|--------|
| `scx_version` | `env!("CARGO_PKG_VERSION")` at compile time |
| `merged_at` | UTC wall-clock at merge start (sidecar only in future) |
| `base.sha256` | SHA-256 of base file bytes — immutable anchor |
| `patch.sha256` | SHA-256 of each patch file |
| `tags` | `--tag key=value` flags, verbatim |
