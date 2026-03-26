# SCX — Single-Cell format conversion

SCX is a lean, format-to-format interoperability engine for single-cell data,
optimized for reproducible benchmarking of conversion correctness, throughput,
and memory use.

Today the project focuses on a narrow core:

- robust conversion between H5Seurat and H5AD
- a memory-bounded streaming pipeline written in Rust
- thin language bindings for benchmarking and interop workflows

Peak memory scales with `chunk_size`, not dataset size.

## Components

| Path | Description |
|------|-------------|
| `crates/scx-core` | Rust core: readers, writers, IR, streaming traits |
| `crates/scx-cli` | `scx` command-line tool |
| `r/picklerick` | R package (extendr bindings) |
| `docs/` | Roadmap and design notes |

## Scope

SCX is an interop engine first, not a full analysis framework. The main product
goal is lean, reproducible benchmarking of format conversion.

### In scope

- H5Seurat ↔ H5AD conversion
- bounded-memory conversion with explicit `chunk_size`
- correctness checks against reference fixtures
- CLI and thin language bindings for R/Python
- internal formats or checkpoints that reduce benchmarking overhead

### Out of scope for the core project

- replacing AnnData / Scanpy / Seurat / BPCells
- becoming a general lazy matrix engine or data platform
- broad cloud-native storage support
- community-driven feature parity across every language binding
- universal zero-copy guarantees
- supporting every possible single-cell format

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

See [`r/picklerick/docs/usage.md`](r/picklerick/docs/usage.md) for full R API.

## Build

```bash
# Rust CLI
cargo build --release -p scx-cli

# R package (requires Cargo ≥ 1.70 and system HDF5)
R CMD INSTALL r/picklerick

# Tests
cargo test
Rscript -e "testthat::test_dir('r/picklerick/tests/testthat')"
```

## Formats

| Format | Read | Write |
|--------|------|-------|
| H5Seurat (SeuratDisk) | yes | yes |
| H5AD (AnnData ≥ 0.8) | yes | yes |
| SCX internal `.h5` | yes | — |

Dense X/layers and nullable obs columns (anndata `IntNA`/`FloatNA`/`BoolNA`)
are fully supported.

## Internal benchmarking format

SCX also has an internal `.npy` snapshot format used as an implementation and
benchmarking aid. It is intentionally exploratory and should be understood as:

- an internal checkpoint format for isolating read/write costs in benchmarks
- a debugging substrate for inspecting the intermediate representation
- a way to reduce overhead in benchmarking tasks without going through HDF5 each time

It is not currently positioned as a primary public interchange format.

## License

GPL-3. See individual crate `Cargo.toml` files for dependency licenses.
