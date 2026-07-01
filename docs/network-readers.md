# Network-backed readers

## Motivation

`scx` can read a single-cell dataset **directly from object storage** — S3, GCS,
an HTTP(S) URL, or a local path — and convert it without staging the file to
disk first. The first network format is **Parquet**; the machinery underneath
(an [`object_store`](https://docs.rs/object_store) transport plus a tokio
runtime constructed at one entry point) is shared, so future network readers
(Zarr, ranged HDF5) plug into the same path.

The headline use case is the [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
perturbation atlas: 100M cells distributed as Parquet on HuggingFace. `scx`
streams a shard over HTTPS and reconstructs a correct, gene-named `.h5ad`.

---

## Building

Network support is behind the **`net`** cargo feature (off by default, so the
standard local-HDF5 build stays runtime-free):

```sh
cargo build -p scx-cli --features net
```

Without `net`, a Parquet or URL input errors with a message telling you to
rebuild with the feature.

---

## Quick start

```sh
# Local Parquet file
scx convert cells.parquet out.h5ad --n-vars 36000

# Remote over HTTPS (no local copy)
scx convert https://host/path/cells.parquet out.h5ad --n-vars 36000

# S3 / GCS
scx convert s3://bucket/key.parquet out.h5ad --n-vars 36000
scx convert gs://bucket/key.parquet out.h5ad --n-vars 36000
```

An input is treated as a network location when it parses as a URL with one of
these schemes; anything else is a local path:

| Scheme | Backend | Status |
|--------|---------|--------|
| `https://`, `http://` | `object_store` HTTP | ✅ enabled |
| `s3://` | Amazon S3 | ✅ enabled |
| `gs://`, `gcs://` | Google Cloud Storage | ✅ enabled |
| `file://`, bare path | local filesystem | ✅ enabled |
| `az://`, `azure://` | Azure Blob | recognized, **not built in v1** |

> **Credentials.** v1 targets public data — no credentials are wired up, so
> private S3/GCS buckets are not supported yet. `AWS_*` / `GOOGLE_*` env vars are
> **not** consulted.

---

## Matrix layouts

The on-disk encoding of the count matrix is **sniffed** from the Parquet schema:

| Layout | Shape | `n_vars` |
|--------|-------|----------|
| **Per-cell lists** | one row per cell; two list columns `genes: List<Int64>` + `expressions: List<Float32>` (the Tahoe layout) | must be supplied (`--n-vars` or `--genes`) |
| **Dense** | one row per cell; one float column per gene | **derived** from the column count |

Long-format COO (`cell, gene, value` triples) is **not supported yet** — the
sniffer rejects it with a clear error rather than misread it.

### `--n-vars`

The per-cell-list expression file does not carry the full gene axis, so the
number of genes must be provided — either explicitly with `--n-vars`, or
implicitly via `--genes` (below). Dense files carry it in their columns, so
neither flag is needed there.

---

## Gene dictionaries (`--genes`) — required for Tahoe

In a per-cell-list file like Tahoe, the integers in `genes` are **token IDs**
from a vocabulary, **not column indices**, each cell's list is led by a **marker
token to ignore**, and the gene axis lives in a *separate* `gene_metadata`
Parquet (`token_id`, `ensembl_id`, `gene_symbol`). Without joining the two, the
matrix is scrambled, every cell gets a phantom marker entry, and `var` is empty.

`--genes <path-or-url>` supplies that dictionary. When given, `scx`:

- remaps each `genes` token id to its real matrix column,
- drops the marker (and any token not in the dictionary),
- populates `var` (index = `ensembl_id`, plus a `gene_symbol` column),
- **derives `n_vars`** from the dictionary (so `--n-vars` is unnecessary).

The dictionary is a second object-store location and flows through the same
transport, so it can be a local path or a URL.

---

## Worked example: Tahoe-100M over HuggingFace

Tahoe ships the matrix as sharded Parquet under `data/` and the gene dictionary
under `metadata/`. Convert the first shard:

```sh
BASE=https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main

scx convert \
  "$BASE/data/train-00000-of-03388.parquet" \
  tahoe_shard0.h5ad \
  --genes "$BASE/metadata/gene_metadata.parquet"
```

This streams the ~68 MB shard over HTTPS and writes a `28225 × 62710` `.h5ad`:
`n_vars` derived from the dictionary, the marker token dropped, and the
perturbation metadata (`drug`, `cell_line_id`, `moa-fine`, …) carried into
`obs`. No `--n-vars` needed.

> **Use a slash-free ref.** `object_store` percent-decodes URL paths, so the
> HuggingFace auto-convert ref `refs%2Fconvert%2Fparquet` gets mangled into
> `refs/convert/parquet` and 404s. Use the **`main`** branch (as above); `scx`
> rejects a `%2F`-containing URL early with this guidance.

---

## Output formats

The reader is fully decoupled from the writer — the conversion pipeline selects
the writer by the **output** extension. So a Parquet input converts to any
supported output:

```sh
scx convert cells.parquet out.h5ad       # AnnData H5AD
scx convert cells.parquet out.h5seurat   # SeuratDisk H5Seurat
```

Any future output writer inherits Parquet input automatically, and any future
network *input* reader (e.g. Zarr) inherits every output writer.

---

## Compression

Parquet column compression is handled transparently on read — **zstd, snappy,
gzip, lz4, and brotli** are all supported, so a zstd-compressed cloud shard
needs no special handling.

---

## Limitations (v1)

- **`convert` only.** `inspect`, `validate`, and `snapshot` do not accept
  Parquet/URL input yet (they report it explicitly).
- **Matrix + obs only.** `obsm`, `uns`, `varm`, `layers`, and `obsp` are not read
  from Parquet (the per-cell/dense sources don't carry them); they come out
  empty.
- **No COO** long-format input.
- **No credentials** — public buckets / URLs only.
- **Dense values are read as f32** (f64 columns are cast).
- The file is read once for the matrix and its obs columns are projected
  separately, so bandwidth is ~1× the file (not 2×). Very large single shards
  are still fetched in full by a single `convert`.
