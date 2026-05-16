# SCX — Single-Cell format conversion

Swiss-army knife for single-cell format conversion, designed for reproducible benchmarks.

## Install

```bash
cargo install scx-cli
```

Build from source:

```bash
cargo build --release -p scx-cli
# binary at target/release/scx
```

Requires Rust ≥ 1.70. HDF5 is built from source and statically linked by
default, so the resulting binary has no system `libhdf5` dependency. Cold
builds add ~1–2 min for compiling HDF5; CMake and a C compiler must be on
PATH.

### For packagers — opting out of vendored HDF5

Distro and conda packagers usually ship their own `libhdf5` and want to
link against it instead of duplicating the library. Pass
`--no-default-features` to disable the `vendored-hdf5` feature (defined on
`scx-core`, mirrored on `scx-cli` and `picklerick`):

| Build path | Opt-out command |
|---|---|
| `cargo install scx-cli` | `cargo install scx-cli --no-default-features` |
| `cargo build` from source | `cargo build --release -p scx-cli --no-default-features` |
| Python wheel (maturin/pip) | `MATURIN_PEP517_ARGS=--no-default-features pip install python/picklerick` |

System `libhdf5` is then required at build *and* run time (`libhdf5-dev`
on Debian/Ubuntu, `hdf5` via Homebrew on macOS, `hdf5` package in conda).
The cargo build resolves it via `HDF5_DIR` or `pkg-config`.

The recipes under `conda.recipe/` (`scx`, `picklerick-python`) already
pass `--no-default-features` and depend on `hdf5` from the host
environment, so they serve as working examples.

The R `picklerick` package currently always links dynamically — it is
built outside the cargo workspace for r-universe compatibility, so the
workspace-level feature doesn't apply to it.

## Usage

### Convert

```bash
scx convert pbmc.h5seurat pbmc.h5ad
scx convert pbmc.h5ad pbmc.h5seurat
```

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size N` | 5000 | Cells per streaming chunk |
| `--dtype` | f32 | Output matrix dtype (`f32`, `f64`, `i32`, `u32`) |
| `--assay` | RNA | Seurat assay (H5Seurat input only) |

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

Design notes, scope, formats, and developer workflows live in [`docs/`](docs/).

## License

GPL-3.
