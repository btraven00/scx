# picklerick

R bindings to [SCX](https://github.com/btraven00/scx), a Rust engine that
converts single-cell data between formats in bounded memory. Reads AnnData
(`.h5ad` and Zarr), H5Seurat, 10x and BPCells files into
`SingleCellExperiment` or `Seurat` objects, converts file to file, and
inspects files without loading the matrix. No Python dependency.

## Install

```r
pak::pak("btraven00/scx/r/picklerick")
```

The package is compiled from source. You need a Rust toolchain (`cargo`),
CMake and a C/C++ compiler. HDF5 is built from source and linked
statically into the package, so no system `libhdf5` is needed, and
picklerick's HDF5 stays separate from the one `rhdf5` or `hdf5r` load in
the same session.

## Quick start

```r
library(picklerick)

inspect("pbmc3k.h5ad")                      # shape, columns, keys; no matrix read

sce <- read_h5ad("pbmc3k.h5ad")             # SingleCellExperiment (default)
obj <- read_h5ad("pbmc3k.h5ad", as = "Seurat")
a   <- read_h5ad("pbmc3k.h5ad", as = "AnnData")   # light handle; as_sce(a), as_seurat(a)
sce <- read_zarr("pbmc3k.zarr")             # AnnData Zarr store, v2 or v3
sce <- read_h5ad("filtered_feature_bc_matrix.h5")  # the format is detected, not the extension

uns(sce)                                    # uns keys; uns(sce, "key") reads one
write_h5ad(sce, "out.h5ad")
convert("pbmc3k.h5seurat", "pbmc3k.h5ad")   # file to file, streamed; .h5ad / .h5seurat / .zarr
```

One reader for every format, detected by content:

| Format | Read | Write |
|---|:-:|:-:|
| AnnData `.h5ad` | ✓ | ✓ |
| AnnData Zarr store (`.zarr` directory, v2 and v3) | ✓ | ✓ (v3) |
| H5Seurat (dgCMatrix or BPCells-backed) | ✓ | ✓ |
| 10x HDF5 (Cell Ranger `.h5`) | ✓ | |
| 10x MatrixMarket directory | ✓ | |
| BPCells matrix directory | ✓ | X only |

Two vignettes: `vignette("overview")` says what the package is and indexes
its functions; `vignette("picklerick")` walks through each target object and
format on a dataset shipped with the package. Every exported function has a
help page (`?read_h5ad`, `?convert`, ...); all of it is also on the
[documentation site](https://btraven00.github.io/scx/r/).

## Lazy matrices (DelayedArray)

`PicklerickMatrix()` is a `DelayedMatrix` (features x cells) whose blocks are
read on demand through scx, for any format scx reads. Selecting cells reads
only those cells from disk on `.h5ad` and AnnData Zarr stores; the row
pointer is loaded, then just the selected rows' slices of `data`/`indices`.

```r
m <- PicklerickMatrix("pbmc.zarr")          # or a layer: layer = "counts"
m[, 1:100]                                  # reads 100 cells
DelayedArray::colSums(m)                    # block-wise, chunk_size cells per block
sce <- read_zarr("pbmc.zarr", lazy = TRUE)  # counts(sce) is this matrix
```

For plain `.h5ad`, `read_h5ad(lazy = TRUE)` keeps using
`HDF5Array::H5SparseMatrix` when that package is installed: it is the
conventional seed for that layout and reads it the same way. The scx seed
is what makes Zarr v3 (and 10x, H5Seurat, MatrixMarket, BPCells) lazy in R.

## License

MIT. See [LICENSE](LICENSE).
