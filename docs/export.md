# scx export

## Motivation

`scx export` dumps a tabular slot from any supported input format to a flat
file. The primary use case is getting cell or gene metadata into Python
(pandas) or R (data.frame) for downstream analysis without parsing HDF5
directly.

Output format is inferred from the file extension: `.csv` for
comma-separated, `.parquet` for Apache Parquet.

---

## CLI

```sh
# Cell metadata → CSV
scx export merged.h5ad --slot obs --output cells.csv

# Gene metadata → Parquet
scx export merged.h5ad --slot var --output genes.parquet

# PCA embedding → CSV (index + dim_0, dim_1, …)
scx export merged.h5ad --slot obsm/X_pca --output pca.csv
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--slot <slot>` | `obs` | Slot to export: `obs`, `var`, or `obsm/<name>` |
| `--output <path>` / `-o` | required | Output path; format from extension (`.csv` or `.parquet`) |
| `--assay <name>` | `RNA` | Seurat assay (H5Seurat inputs only) |
| `--layer <name>` | `counts` | Seurat layer (H5Seurat inputs only) |

---

## Supported slots

| Slot | Columns | Row count |
|------|---------|-----------|
| `obs` | `index` + all obs columns | n_obs |
| `var` | `index` + all var columns | n_vars |
| `obsm/<name>` | `index`, `dim_0`, `dim_1`, … | n_obs |

The first column is always `index` (cell barcodes for `obs`/`obsm`, gene
names for `var`). Categorical columns are decoded to their level strings.

---

## Output formats

### CSV

Written with polars `CsvWriter`. UTF-8, comma-separated, header row
included.

```sh
scx export merged.h5ad --slot obs --output cells.csv
```

```python
import pandas as pd
df = pd.read_csv("cells.csv", index_col="index")
```

```r
df <- read.csv("cells.csv", row.names = "index")
```

### Parquet

Written with polars `ParquetWriter`. Snappy-compressed by default. Reads
faster than CSV for large metadata tables.

```sh
scx export merged.h5ad --slot obs --output cells.parquet
```

```python
import pandas as pd
df = pd.read_parquet("cells.parquet")

import polars as pl
df = pl.read_parquet("cells.parquet")
```

```r
library(arrow)
df <- read_parquet("cells.parquet")
```

---

## Column type mapping

| scx `ColumnData` | CSV / Parquet dtype |
|-----------------|---------------------|
| `Float` (f64) | Float64 |
| `Int` (i32) | Int32 |
| `Bool` | Boolean |
| `String` | Utf8 |
| `Categorical` | Utf8 (levels decoded) |

Categorical columns are expanded to their string level values. If you need
the integer codes or the ordered factor representation, read the h5ad directly
via anndata or scx-py.

---

## Input formats

Any format supported by `scx inspect` is accepted. The input format is
auto-detected by content.

| Extension / content | Reader |
|--------------------|--------|
| `.h5ad` | H5AdReader |
| `.h5seurat` | H5SeuratReader |
| BPCells directory | BpcellsDatasetReader |

For H5Seurat inputs, `--assay` and `--layer` select which assay metadata
to load.

---

## Examples

### Export cluster labels after merge

```sh
scx merge \
  --base data/source.h5ad \
  --patch results/clusters.h5ad:obs/leiden \
  --output results/merged.h5ad

scx export results/merged.h5ad \
  --slot obs \
  --output results/cell_meta.parquet
```

### Export PCA coordinates for plotting

```sh
scx export results/merged.h5ad \
  --slot obsm/X_pca \
  --output results/pca.csv
```

```python
import pandas as pd
pca = pd.read_csv("results/pca.csv", index_col="index")
# pca columns: dim_0, dim_1, dim_2, ...
```

### Export gene metadata for filtering

```sh
scx export results/merged.h5ad --slot var --output results/gene_meta.csv
```

---

## Implementation

```
scx-cli/src/cmd_export.rs   — ExportArgs, run_export(), column_data_to_column()
scx-cli/src/main.rs         — Export subcommand wiring
```

Depends on `polars 0.46` (features: `csv`, `parquet`, `dtype-categorical`).
