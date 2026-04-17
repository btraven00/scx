# SCX — Single-Cell format conversion

Swiss-army knife for single-cell format conversion, designed for reproducible benchmarks.

## Install

```bash
cargo install --path scx-cli
```

Or build from source:

```bash
cargo build --release -p scx-cli
# binary at target/release/scx
```

Requires Rust ≥ 1.70 and a system HDF5 installation.

## Usage

### Convert

```bash
scx convert pbmc.h5seurat pbmc.h5ad
scx convert pbmc.h5ad pbmc.h5seurat
```

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size N` | 5000 | Cells per streaming chunk |
| `--dtype` | f32 | Output matrix dtype (`f32`, `f64`, `i32`, `u32`) |
| `--assay` | RNA | Seurat assay (H5Seurat input only) |

### Inspect

```bash
scx inspect pbmc.h5ad
scx inspect pbmc.h5seurat
```

Prints format, shape, and a summary of every slot (obs, var, obsm, layers,
obsp, varm, uns) without loading the matrix.

## Provenance

Every `scx convert` run writes a sidecar `<output>.prov.json` with the source
SHA256, output SHA256, shape, and timestamp. The artifact itself contains a
deterministic `uns["scx_provenance"]` block (no timestamp) so byte-level
reproducibility is preserved.

## Docs

Design notes, scope, formats, and developer workflows live in [`docs/`](docs/).

## License

GPL-3.
