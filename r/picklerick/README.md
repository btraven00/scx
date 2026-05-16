# picklerick

R bindings to the SCX single-cell format conversion engine.

Converts between H5Seurat, H5AD, and SCX formats via a memory-bounded pipeline
written in Rust. Returns `anndataR::InMemoryAnnData` objects. No Python dependency.

## Install

```r
# from source (requires Cargo >= 1.70)
pak::pkg_install("local::r/picklerick")
```

### System requirements

A system HDF5 library is required at both build and run time:

- Debian/Ubuntu: `apt install libhdf5-dev`
- macOS: `brew install hdf5`
- conda: `conda install -c conda-forge hdf5`

`Makevars` resolves it via `pkg-config --libs hdf5` with a Debian fallback
to `/usr/lib/x86_64-linux-gnu/hdf5/serial`.

### Note: no vendored HDF5 (unlike the Rust CLI / Python wheel)

The workspace `scx-cli` binary and the `picklerick` Python wheel default to
a vendored static HDF5 (see the top-level README), so end users do not need
to install `libhdf5` separately. **The R package is the exception** — it
always links dynamically.

Two reasons:

1. **Builds outside the cargo workspace.** For r-universe compatibility,
   `r/picklerick/src/rust/` is its own crate with a vendored copy of
   `scx-core` under `r/picklerick/src/rust/scx-core/` (synced via
   `r/picklerick/sync-scx-core.sh`). The workspace-level `vendored-hdf5`
   cargo feature does not reach it.
2. **`rhdf5` / `hdf5r` coexistence risk.** Loading two distinct `libhdf5`
   builds in the same R session corrupts HDF5 property-list IDs. See
   `docs/roadmap.md` §0.0.6. Until that interaction is re-validated under
   a vendored static `libhdf5`, the safe default is to share the system
   `libhdf5` with `rhdf5` / Bioconductor.

Known limitation regardless of vendoring: do **not** load `hdf5r` in the
same R session as `picklerick` native mode — `hdf5r` links against a
different `libhdf5.so` build and property-list IDs may corrupt. `rhdf5`
coexistence has been empirically confirmed safe for simple
open-read-close conversions.

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
