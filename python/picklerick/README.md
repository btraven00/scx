# picklerick (Python)

Thin Python bindings for the SCX single-cell interoperability engine.

`picklerick` is a small Python package that returns real `anndata.AnnData`
objects where appropriate and uses the SCX engine for format conversion. It is
designed for lean, reproducible benchmarking and interop workflows, not as a
separate analysis engine.

The package supports two execution modes:

- **CLI-backed mode** — the default portable path; calls the `scx` executable
- **native mode** — optional in-process acceleration via a PyO3 extension

Both modes share the same public Python API.

## Scope

This package intentionally keeps a narrow scope:

- read H5Seurat into `anndata.AnnData`
- read H5AD through a thin convenience wrapper
- write H5Seurat from `anndata.AnnData`
- write H5AD through a thin convenience wrapper
- call the SCX conversion engine from Python

It does **not** aim to:

- replace `anndata`, `scanpy`, or Seurat
- provide a full lazy/backed data API
- duplicate the SCX engine in Python

## Installation

For now, `picklerick` is expected to be used from a source checkout of SCX.

### CLI-backed mode

Build the `scx` binary:

```bash
cargo build --release -p scx-cli
```

Install the Python package:

```bash
pip install -e python/picklerick
```

Make sure the `scx` binary is available either:

- on your `PATH`, or
- via an environment variable such as `SCX_BIN`

Example:

```bash
export SCX_BIN="$(pwd)/target/release/scx"
```

### Native mode

`picklerick` can also use an optional in-process native backend built with
PyO3. This is intended as an acceleration path over the same engine semantics,
not as a separate feature set.

If the native extension is present, the Python package may use it for core
conversion operations. If it is absent, `picklerick` falls back to the CLI path.

The package is therefore usable in either mode:

- **without native extension**: requires `scx` on `PATH` or `SCX_BIN`
- **with native extension**: may avoid subprocess overhead for supported calls

For the native backend, use the dedicated pixi Python 3.13 environment:

```bash
pixi run -e py313 install-picklerick-py-native
```

## Test workflows

### CLI-backed tests

From the SCX repository root:

```bash
pixi run -e test install-picklerick-py
pixi run -e test verify-picklerick-py
```

This path validates the Python package in fallback mode against the shared test
fixtures.

### Native backend tests

From the SCX repository root:

```bash
pixi run -e py313 install-picklerick-py-native
pixi run -e py313 verify-picklerick-py-native
```

This path validates the optional PyO3 backend in the dedicated Python 3.13
environment.

## API

The initial Python API mirrors the core shape of the R package where it makes
sense for Python:

- `picklerick.read(path, ...)`
- `picklerick.read_dataset(path, ...)`
- `picklerick.read_h5seurat(path, assay="RNA", layer="counts", chunk_size=5000, dtype="f32")`
- `picklerick.read_h5ad(path)`
- `picklerick.write_h5ad(adata, path, compression="gzip")`
- `picklerick.write_h5seurat(adata, path, assay="RNA", chunk_size=5000)`
- `picklerick.convert(input, output, chunk_size=5000, dtype="f32", assay="RNA", layer="counts")`
- `picklerick.open_stream(path, chunk_size=5000, assay="RNA", layer="counts")` — native backend only

## Examples

### Read H5Seurat into AnnData

```python
import picklerick as pk

adata = pk.read_h5seurat("pbmc3k.h5seurat")
print(adata.shape)
```

### Format-agnostic read

```python
import picklerick as pk

adata = pk.read("pbmc3k.h5seurat")
adata2 = pk.read("pbmc3k_reference.h5ad")
```

### Convert without constructing AnnData

```python
import picklerick as pk

pk.convert(
    "pbmc3k.h5seurat",
    "pbmc3k.h5ad",
    chunk_size=5000,
    dtype="f32",
    assay="RNA",
    layer="counts",
)
```

### Write H5Seurat from AnnData

```python
import picklerick as pk

adata = pk.read_h5ad("pbmc3k_reference.h5ad")
pk.write_h5seurat(adata, "pbmc3k_out.h5seurat")
```

### Stream a matrix chunk-by-chunk (bounded memory)

`open_stream` yields CSR row blocks without materializing the whole matrix or
constructing an AnnData object, so peak memory is governed by `chunk_size`
rather than dataset size. The same API works across H5AD, H5Seurat, ScxH5, and
BPCells-dir inputs (native backend only). The `indptr`/`indices`/`data` arrays
are zero-copy numpy arrays that own the decoded buffer (moved, not copied); they
are writable and stay valid after the chunk is gone, so you can keep or mutate
them freely.

```python
import numpy as np
import picklerick as pk

gene_sums = None
for chunk in pk.open_stream("counts.h5seurat", chunk_size=5000):
    if gene_sums is None:
        gene_sums = np.zeros(chunk.n_vars, dtype=np.float64)
    # chunk.indptr / chunk.indices / chunk.data are the CSR triplet.
    # bincount accumulates in float64 internally, so the native f32 weights
    # give a bit-identical result without a full-nnz astype copy.
    gene_sums += np.bincount(chunk.indices, weights=chunk.data, minlength=chunk.n_vars)
```

For deflate-chunked H5AD `X`, the matrix is decoded by reading the raw gzip
chunks and inflating them across cores, rather than through libhdf5's
single-threaded filter; other filter pipelines (shuffle, blosc) fall back to the
normal read. (anndata's `read_h5ad(backed="r")` + `chunked_X` is also a
bounded-memory reader for h5ad, decoding on one core.)

> Build the extension `--release` (`pixi run -e py313
> install-picklerick-py-native-release`) for representative timings; a debug
> `maturin develop` build is ~3× slower on the HDF5 decode path.

Peak RSS / wall, per-gene-sum workload (`bench/python/`, release build, chunk 5000, 16 cores):

| dataset | size | eager `read_h5ad` | anndata backed | `open_stream` |
|---------|-----:|------------------:|---------------:|--------------:|
| pbmc3k  | 29 MB  | 0.15 GB / 0.42 s | 0.15 GB / 0.43 s | 0.15 GB / 0.44 s |
| norman  | 79 MB  | 0.24 GB / 1.16 s | 0.23 GB / 1.03 s | 0.11 GB / 0.42 s |
| hlca    | 5.7 GB | 18.2 GB / 50 s   | 0.93 GB / 30 s   | 0.55 GB / 17 s   |

Notes:

- The multicore decode shows at scale (hlca): ~5 s to decode `X` vs ~21 s
  single-threaded; the remaining time is the per-gene-sum reduction, not the
  read. Small datasets are a single chunk at chunk 5000, so there is nothing to
  parallelize.
- Peak RSS grows with `chunk_size` (prefetch + per-row-block inflate buffers):
  hlca is 0.55 GB at chunk 5000 but 3.6 GB at chunk 50000. Keep `chunk_size`
  modest (a few thousand).

Reach for `open_stream` when the matrix is in a non-anndata format (H5Seurat,
BPCells-dir, ScxH5), when you want bounded-memory access without an anndata
dependency, or for a bounded-memory read of deflate-chunked h5ad.

## Implementation notes

The implementation is intentionally simple:

- H5AD reads/writes use normal `anndata` I/O
- H5Seurat reads convert into a temporary `.h5ad`, then load with `anndata`
- H5Seurat writes first materialize a temporary `.h5ad`
- core conversion is performed by the SCX engine, either through:
  - the `scx` CLI, or
  - an optional native PyO3 backend when available
- engine failures are surfaced as Python exceptions

This keeps the public API small while reusing the existing Rust engine.

## Relationship to SCX

SCX is a lean, format-to-format interoperability engine for single-cell data,
optimized for reproducible benchmarking of conversion correctness, throughput,
and memory use.

`picklerick` is the Python convenience layer over that engine.

## Status

Early scaffold. The goal of the first release is correctness and a stable,
boring API, not broad feature coverage.