# SCX — Single-Cell format conversion

SCX is a lean interoperability engine for single-cell format conversion,
designed to support reproducible benchmarking of correctness, throughput,
and memory use.

The project currently focuses on a narrow core:

- conversion between H5Seurat and H5AD, with attention to correctness and bounded memory use
- a Rust streaming pipeline intended to keep memory use bounded by `chunk_size`
- thin language bindings for interop workflows and benchmarking support

Peak memory is intended to scale primarily with `chunk_size` rather than total dataset size.

## Components

| Path | Description |
|------|-------------|
| `crates/scx-core` | Rust core: readers, writers, IR, streaming traits |
| `crates/scx-cli` | `scx` command-line tool |
| `r/picklerick` | R package (extendr bindings) |
| `python/picklerick` | Python package (hybrid bindings: CLI-backed with optional native PyO3 acceleration) |
| `docs/` | Roadmap and design notes |

## Scope

SCX is an interop library, not a full analysis framework. The main goal
is reproducible benchmarking of format conversion.

### In scope

- H5Seurat ↔ H5AD conversion
- bounded-memory conversion with explicit `chunk_size`
- correctness checks against reference fixtures
- CLI and thin language bindings for R/Python
- internal formats or checkpoints that reduce benchmarking overhead

### Out of scope

- replacing AnnData / Scanpy / Seurat / BPCells
- broad cloud-native storage support
- feature parity across language bindings
- supporting every single-cell format

## Quick start

### CLI

```bash
scx convert pbmc.h5seurat pbmc.h5ad
scx convert pbmc.h5ad pbmc.h5seurat
```

### R

```r
library(picklerick)

adata <- read_h5seurat("pbmc.h5seurat")   # → anndataR::InMemoryAnnData
write_h5ad(adata, "pbmc.h5ad")
write_h5seurat(adata, "pbmc.h5seurat")

seu <- read_seurat("pbmc.h5seurat")       # → Seurat object (requires Seurat ≥ 5)
sce <- read_sce("pbmc.h5seurat")          # → SingleCellExperiment
```

See [`r/picklerick/docs/usage.md`](r/picklerick/docs/usage.md) for the full R API.

## Build

```bash
# Rust CLI
cargo build --release -p scx-cli

# R package (requires Cargo ≥ 1.70 and system HDF5)
R CMD INSTALL r/picklerick

# Python package
pip install -e python/picklerick

# Tests
cargo test
Rscript -e "testthat::test_dir('r/picklerick/tests/testthat')"
python -m pytest python/picklerick/tests -q
```

## Pixi workflows

For repo-local testing, pixi provides the most reliable workflows.

### Python package (CLI-backed path)

```bash
pixi run -e test install-picklerick-py
pixi run -e test verify-picklerick-py
```

### Python package (optional native PyO3 backend)

```bash
pixi run -e py313 install-picklerick-py-native
pixi run -e py313 verify-picklerick-py-native
```

### Shared test fixtures

If the golden fixtures are missing, generate them with:

```bash
pixi run -e test fixtures
```

## Formats

| Format | Read | Write |
|--------|------|-------|
| H5Seurat (SeuratDisk) | yes | yes |
| H5AD (AnnData ≥ 0.8) | yes | yes |
| SCX internal `.h5` | yes | — |

Dense X/layers and nullable obs columns (anndata `IntNA`/`FloatNA`/`BoolNA`)
are supported.

## Internal benchmarking format

SCX also has an internal `.npy` snapshot format used as an implementation and
benchmarking aid. It is intended as an internal, exploratory mechanism and should be understood as:

- an internal checkpoint format for isolating read/write costs in benchmarks
- a debugging substrate for inspecting the intermediate representation
- a way to reduce overhead in benchmarking tasks without going through HDF5 each time

It is not currently positioned as a primary public interchange format.

## License

GPL-3. See individual crate `Cargo.toml` files for dependency licenses.
