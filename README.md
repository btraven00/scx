# SCX — Single-Cell format conversion

[![codecov](https://codecov.io/github/btraven00/scx/graph/badge.svg?token=NSMO60CRF5)](https://codecov.io/github/btraven00/scx)
[![crates.io](https://img.shields.io/crates/v/scx-cli?label=crates.io%20scx-cli)](https://crates.io/crates/scx-cli)
[![PyPI](https://img.shields.io/pypi/v/scx-picklerick?label=PyPI%20scx-picklerick)](https://pypi.org/project/scx-picklerick/)
[![docs](https://img.shields.io/badge/docs-btraven00.github.io%2Fscx-blue)](https://btraven00.github.io/scx/)

<img src="docs/logo.png" alt="SCX logo" width="200">

The swiss-army knife for single-cell format conversion.

## Install

```bash
cargo install scx-cli
```

Build from source:

```bash
cargo build --release -p scx-cli
# binary at target/release/scx
```

Requires Rust ≥ 1.91. HDF5 is built from source and statically linked by
default, so the resulting binary has no system `libhdf5` dependency. Cold
builds add ~1–2 min for compiling HDF5; CMake and a C compiler must be on
PATH.

Packagers who prefer to link the system `libhdf5` instead of the vendored
build can opt out with `--no-default-features` — see
[`docs/packaging.md`](docs/packaging.md).

## Formats

| Format | Read | Write |
|---|:-:|:-:|
| AnnData `.h5ad` (also the pre-0.7 layout, read only) | ✓ | ✓ |
| AnnData Zarr (v2, v3; directory) | ✓ | ✓ (v3) |
| MuData `.h5mu` (one modality per conversion, picked with `--assay`) | ✓ | |
| Seurat `.h5seurat` (dgCMatrix or BPCells-backed) | ✓ | ✓ |
| BPCells matrix directory | ✓ | ✓ X only |
| 10x HDF5 (Cell Ranger `.h5`) | ✓ | |
| MatrixMarket `.mtx[.gz]` / 10x MEX directory | ✓ | |
| 10x Visium (Space Ranger `outs/`) and Visium HD (one bin, e.g. `binned_outputs/square_008um`): positions, scale factors and images, laid out as `scanpy.read_visium` | ✓ | |
| Parquet, local or S3/GCS/HTTP (build with `--features net`) | ✓ | |
| Seurat `.rds` / `.rda` (layers, cell and feature metadata, reductions, graphs; v3/v4 `Assay`, v5 `Assay5`) | ✓ | [no](docs/rds-write.md) |
| bixverse binary sparse store, v3 and v4 (build with `--features bixverse`) | ✓ | |

Any readable format converts to any writable one: everything goes through a
single internal representation. Input is detected by content; output by
extension (`.h5ad`, `.h5seurat`, `.zarr`, `.bpcells`).

With `--features net`, the input can also be a URL (`https://`, `s3://`,
`gs://`, …). Parquet is read in place by byte ranges; any other single-file
format is downloaded to a temporary file first, and its SHA-256 recorded.
`-` reads the input from stdin the same way, e.g.
`curl -sL https://host/x.h5ad | scx convert - x.h5seurat`.

## Usage

### Convert

```bash
scx convert pbmc.h5seurat pbmc.h5ad
scx convert pbmc.zarr pbmc.h5seurat     # AnnData Zarr store (v2 or v3)
scx convert pbmc.h5ad pbmc.zarr         # AnnData Zarr v3 store
scx convert pbmc.h5ad pbmc.h5seurat
```

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size N` | 5000 | Cells per streaming chunk |
| `--dtype` | f32 | Output matrix dtype (`f32`, `f64`, `i32`, `u32`) |
| `--assay` | RNA | Seurat assay (H5Seurat, `.rds`); MuData modality (`.h5mu`: the only one, else `rna`) |

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

You can find [user-facing tutorials and the command-line, Python and R references](https://btraven00.github.io/scx/) at the live site. PRs welcome!

Design notes, scope, formats, and developer workflows live in [`docs/`](docs/).

## Coverage

[![Coverage grid](https://codecov.io/github/btraven00/scx/graphs/tree.svg?token=NSMO60CRF5)](https://codecov.io/github/btraven00/scx)

Each cell is a file; size = lines, color = coverage.

## Acknowledgements

Copying an h5ad's sparse matrices into a dgCMatrix h5seurat as stored, compressed
chunks moved without decoding, follows
[scConvert](https://github.com/mianaz/scConvert) (Ziyu Zeng), which converts
h5ad to h5Seurat the same way. scx's implementation is its own; it also checks
the dgCMatrix row rules during the copy and streams the matrix instead when
they fail.

## License

MIT. See [LICENSE](LICENSE).
