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
| `--source-sha256 <hex>` | Pre-computed SHA-256 (64 hex chars). Skips rehashing — intended for pipelines (e.g. hapiq) that already hashed on download. |

### hapiq integration

`--source-url` is designed for use with
[hapiq](https://github.com/btraven00/hapiq), a declarative downloader.
The typical pipeline is:

```bash
hapiq get https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat \
  -o data/pbmc3k.h5seurat

scx convert data/pbmc3k.h5seurat results/pbmc3k.h5ad \
  --source-url https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat \
  --source-sha256 "$(hapiq hash data/pbmc3k.h5seurat)"
```

Passing `--source-sha256` from the downloader avoids a second full read of
the source file during conversion.

The artifact then carries the full lineage: origin URL + source SHA256 +
scx version. Anyone with the artifact can re-derive the source file
independently.

### Viewing provenance

`scx inspect <file>` prints the embedded `scx_provenance` block as a
dedicated section above `uns`:

```
provenance
  scx_version    0.1.0
  source.url     https://datasets.cellxgene.cziscience.com/pbmc3k.h5seurat
  source.path    data/pbmc3k.h5seurat
  source.sha256  703a1b4a…
```

Files converted without `--source-url` show `(none)` for the URL line.
Files produced by other tools (no `scx_provenance` block) show
`provenance: (none)`.

---

## Merge provenance

`scx merge` assembles multiple pipeline stage outputs into a single h5ad.
The merged file carries a full audit trail: the base file anchor and every
slot that was patched in. See [merge.md](merge.md) for the full command
reference.

### Schema

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

| Field | Description |
|-------|-------------|
| `scx_version` | `env!("CARGO_PKG_VERSION")` at compile time |
| `base.sha256` | SHA-256 of base file bytes — immutable anchor |
| `slots.<key>.sha256` | SHA-256 of the patch source file at the time of patching |
| `slots.<key>.added_at` | UTC timestamp of the patch operation |
| `tags` | `--tag key=value` flags, verbatim |

### Storage

The provenance block is stored as a single JSON string scalar at
`uns["scx_provenance"]`. This avoids HDF5 path-separator mangling of slot
keys that contain `/` (e.g. `layers/normalized`). `H5AdReader.uns()` parses
the string back to a JSON object transparently, so callers see a normal
nested structure.
