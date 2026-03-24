# picklerick usage guide

picklerick wraps the SCX Rust engine to stream single-cell data between formats
without loading the full dataset into memory.

## Reading

### H5Seurat → AnnData

```r
adata <- read_h5seurat(
  path       = "pbmc3k.h5seurat",
  assay      = "RNA",          # Seurat assay to read (default "RNA")
  layer      = "counts",       # Seurat layer to read (default "counts")
  chunk_size = 5000L,          # cells per streaming chunk
  dtype      = "f32"           # output numeric type: f32, f64, i32, u32
)
```

### H5AD → AnnData

```r
adata <- read_h5ad("pbmc3k.h5ad")
```

### Auto-detect format

```r
adata <- read_dataset("pbmc3k.h5seurat")  # dispatches on extension
```

### Direct coercions

These require the **Seurat** (>= 5) or **SingleCellExperiment** package installed.

```r
seu <- read_seurat("pbmc3k.h5seurat")
sce <- read_sce("pbmc3k.h5seurat")
```

Both functions pass `...` through to `read_h5seurat()`, so all parameters
(`assay`, `layer`, `chunk_size`, `dtype`) are available.

---

## Writing

### AnnData → H5AD

```r
write_h5ad(adata, "out.h5ad")
```

### AnnData → H5Seurat

```r
write_h5seurat(
  adata      = adata,
  path       = "out.h5seurat",
  assay      = "RNA",     # target assay name (default "RNA")
  chunk_size = 5000L
)
```

---

## Low-level convert

`convert()` is the format-agnostic single-file-to-single-file entry point:

```r
convert(
  input      = "in.h5seurat",
  output     = "out.h5ad",
  chunk_size = 5000L,
  dtype      = "f32",
  assay      = "RNA",
  layer      = "counts"
)
```

---

## Supported formats

| Format | Read | Write |
|--------|------|-------|
| H5Seurat (SeuratDisk) | yes | yes |
| H5AD (AnnData) | yes | yes |
| SCX internal `.h5` | yes | — |

### Dense X / layers

Both sparse CSR and dense 2-D matrix layouts are supported for `/X` and
`/layers/*`. Dense arrays are converted to sparse on read (exact zeros are
dropped).

### Nullable obs/var columns

AnnData nullable integer, float, and boolean columns (`IntNA`, `FloatNA`,
`BoolNA`) are read: floats/ints use `NaN` as the NA sentinel; booleans
map NA to `FALSE`.

---

## Memory usage

Peak memory is proportional to `chunk_size` (cells), not dataset size.
For atlas-scale data (10M–100M cells) the default `chunk_size = 5000L`
keeps per-conversion memory well under 1 GB.

---

## Native vs. CLI mode

picklerick tries to call the Rust engine in-process (native mode).
If the shared library is unavailable it falls back to invoking the `scx`
command-line binary on `PATH`. Native mode is preferred for performance;
both modes produce identical output.

Check which mode is active:

```r
picklerick:::.native_available()
```
