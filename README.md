# SCX — Single-Cell format conversion

[![codecov](https://codecov.io/github/btraven00/scx/graph/badge.svg?token=NSMO60CRF5)](https://codecov.io/github/btraven00/scx)

Swiss-army knife for single-cell format conversion, designed for reproducible benchmarks.

## Install

```bash
cargo install scx-cli
```

Build from source:

```bash
cargo build --release -p scx-cli
# binary at target/release/scx
```

Requires Rust ≥ 1.70. HDF5 is built from source and statically linked by
default, so the resulting binary has no system `libhdf5` dependency. Cold
builds add ~1–2 min for compiling HDF5; CMake and a C compiler must be on
PATH.

Packagers who prefer to link the system `libhdf5` instead of the vendored
build can opt out with `--no-default-features` — see
[`docs/packaging.md`](docs/packaging.md).

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

## Coverage

[![Coverage grid](https://codecov.io/github/btraven00/scx/graphs/tree.svg?token=NSMO60CRF5)](https://codecov.io/github/btraven00/scx)

Each cell is a file; size = lines, color = coverage.

## License

GPL-3.
