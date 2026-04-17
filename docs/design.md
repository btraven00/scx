# Design notes

## Components

| Path | Description |
|------|-------------|
| `crates/scx-core` | Rust core: readers, writers, IR, streaming traits |
| `crates/scx-cli` | `scx` command-line tool |
| `r/picklerick` | R package (extendr bindings) |
| `python/picklerick` | Python package (hybrid: CLI-backed with optional native PyO3 acceleration) |
| `docs/` | Design notes |

## Scope

SCX is an interop library, not a full analysis framework.

### In scope

- H5Seurat ↔ H5AD conversion
- Bounded-memory conversion with explicit `chunk_size`
- Correctness checks against reference fixtures
- CLI and thin language bindings for R/Python
- Internal formats or checkpoints that reduce benchmarking overhead

### Out of scope

- Replacing AnnData / Scanpy / Seurat / BPCells
- Broad cloud-native storage support
- Feature parity across language bindings
- Supporting every single-cell format

## Formats

| Format | Read | Write |
|--------|------|-------|
| H5Seurat (SeuratDisk) | yes | yes |
| H5AD (AnnData ≥ 0.8) | yes | yes |
| SCX internal `.h5` | yes | — |

Dense X/layers and nullable obs columns (`IntNA`/`FloatNA`/`BoolNA`) are supported.

## Internal snapshot format

SCX has an internal `.npy` snapshot format used as a benchmarking aid:

- Isolates read/write costs by caching the intermediate representation to disk
- Useful for debugging the IR without going through HDF5 each time
- Not a public interchange format

See [`npy-format.md`](npy-format.md) for the layout.

## Build

```bash
# Rust CLI
cargo build --release -p scx-cli

# R package (requires Cargo ≥ 1.70 and system HDF5)
R CMD INSTALL r/picklerick

# Python package
pip install -e python/picklerick

# Tests
cargo test
Rscript -e "testthat::test_dir('r/picklerick/tests/testthat')"
python -m pytest python/picklerick/tests -q
```

## Pixi workflows

For repo-local testing, pixi provides the most reliable environment.

### Python package (CLI-backed path)

```bash
pixi run -e test install-picklerick-py
pixi run -e test verify-picklerick-py
```

### Python package (optional native PyO3 backend)

```bash
pixi run -e py313 install-picklerick-py-native
pixi run -e py313 verify-picklerick-py-native
```

### Test fixtures

If the golden fixtures are missing:

```bash
pixi run -e test fixtures
```
