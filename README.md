# SCX — Single-Cell format conversion

Converts between single-cell genomics formats (H5Seurat, H5AD) via a
memory-bounded streaming pipeline written in Rust. Peak memory scales with `chunk_size`,
not dataset size — suitable for atlas-scale data (10M–100M cells).

## Components

| Path | Description |
|------|-------------|
| `crates/scx-core` | Rust core: readers, writers, IR, streaming traits |
| `crates/scx-cli` | `scx` command-line tool |
| `r/picklerick` | R package (extendr bindings) |
| `docs/` | Roadmap and design notes |

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

## License

GPL-3. See individual crate `Cargo.toml` files for dependency licenses.
