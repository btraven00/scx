# picklerick

Python bindings for [SCX](https://github.com/btraven00/scx), a Rust engine
that converts single-cell data between formats in bounded memory.

```bash
pip install --pre scx-picklerick   # 0.1.0a1; drop --pre once 0.1.0 is out
```

```python
import picklerick as pk
```

The PyPI distribution is named `scx-picklerick` because `picklerick` is too
close to an existing project name. The import name is `picklerick`.

Requires Python 3.14 or newer. Wheels are built for Linux (x86_64, aarch64)
and macOS (arm64) and include a statically linked HDF5. You don't need a
system HDF5.

> **Alpha.** The API below is the one meant to stay, but it may still
> change before 0.1.0. Please report problems at
> <https://github.com/btraven00/scx/issues>.

## What it does

| Function | Purpose |
|---|---|
| `pk.convert(src, dst, *, dtype="f32", assay="RNA", layer="counts", chunk_size=5000)` | Convert a file to `.h5ad` or `.h5seurat`. The output format is chosen from the extension of `dst`. |
| `pk.read(path, *, assay="RNA", layer="counts", dtype="f32", chunk_size=5000)` | Read any supported input into an in-memory `anndata.AnnData`. |
| `pk.write_h5seurat(adata, path, *, assay="RNA", chunk_size=5000)` | Write an `AnnData` to H5Seurat, in the SeuratDisk layout. |
| `pk.inspect(path)` | Return shape, columns, layers and nnz statistics without reading the matrix. |
| `pk.open_stream(path, *, chunk_size=5000, assay="RNA", layer="counts")` | Iterate over X as CSR row blocks (`pk.MatrixChunk`). |

Supported inputs: H5AD, H5Seurat (dgCMatrix or BPCells-backed), BPCells
directories, 10x HDF5, MatrixMarket, and AnnData Zarr. Engine errors are
raised as `pk.PickleRickError`.

`convert` streams the matrix in chunks of `chunk_size` cells, so peak
memory does not grow with the number of cells. When converting to H5Seurat
from a source that has a `counts` layer, X is treated as normalised data:
X is written to the Seurat `data` slot and the `counts` layer to the
`counts` slot.

## Examples

```python
import picklerick as pk

pk.convert("pbmc3k.h5seurat", "pbmc3k.h5ad")
pk.convert("pbmc3k.h5ad", "pbmc3k.h5seurat")

adata = pk.read("pbmc3k.h5seurat")          # AnnData
pk.write_h5seurat(adata, "copy.h5seurat")

pk.inspect("atlas.h5ad")["n_obs"]
```

### Streaming

`open_stream` yields CSR row blocks without loading the whole matrix or
building an AnnData object. Decoding runs on a background thread a few
chunks ahead of the consumer. The numpy arrays take ownership of the
decoded buffers, so there is no copy per chunk.

```python
import numpy as np
import picklerick as pk

gene_sums = None
for chunk in pk.open_stream("atlas.h5ad", chunk_size=5000):
    if gene_sums is None:
        gene_sums = np.zeros(chunk.n_vars)
    gene_sums += np.bincount(chunk.indices, weights=chunk.data, minlength=chunk.n_vars)
```

For deflate-compressed H5AD, chunks are decompressed on all cores. HDF5's
own filter pipeline would decompress them on a single core.

Peak RSS and wall time for a per-gene-sum workload (`bench/python/`,
release build, chunk 5000, 16 cores):

| dataset | size | eager `read_h5ad` | anndata backed | `open_stream` |
|---------|-----:|------------------:|---------------:|--------------:|
| pbmc3k  | 29 MB  | 0.15 GB / 0.42 s | 0.15 GB / 0.43 s | 0.15 GB / 0.44 s |
| norman  | 79 MB  | 0.24 GB / 1.16 s | 0.23 GB / 1.03 s | 0.11 GB / 0.42 s |
| hlca    | 5.7 GB | 18.2 GB / 50 s   | 0.93 GB / 30 s   | 0.55 GB / 17 s   |

Peak memory grows with `chunk_size`: HLCA uses 0.55 GB at 5000 and 3.6 GB at
50000.

## Development

From the repository root:

```bash
pixi run -e py test-picklerick            # debug build + pytest
pixi run -e py install-picklerick-release # before any benchmarking
```

A debug build is about 3× slower at HDF5 decoding, so use the release build
for timings. Tests use the small fixtures tracked in git. Tests that need the
large local fixtures (`pixi run -e test fixtures`) are skipped, with the
reason shown, when those fixtures are missing.

## License

MIT.
