# picklerick

R bindings to the [SCX](https://github.com/your-org/scx) streaming single-cell interop engine.

Converts between H5Seurat, H5AD, and SCX formats via a memory-bounded Rust pipeline.
Returns `anndataR::InMemoryAnnData` objects. No Python dependency.

## Install

```r
# from source (requires Cargo >= 1.70)
pak::pkg_install("local::r/picklerick")
```

## Quick start

```r
library(picklerick)

# H5Seurat → AnnData
adata <- read_h5seurat("pbmc3k.h5seurat")

# H5AD → AnnData
adata <- read_h5ad("pbmc3k.h5ad")

# AnnData → H5Seurat
write_h5seurat(adata, "out.h5seurat")

# AnnData → H5AD
write_h5ad(adata, "out.h5ad")

# Direct coercions (require Seurat / SingleCellExperiment packages)
seu <- read_seurat("pbmc3k.h5seurat")
sce <- read_sce("pbmc3k.h5seurat")
```

See `docs/usage.md` for full details.

## License

GPL-3. See [LICENSE](LICENSE).
