# scx-cli

Command-line interface for [SCX](../../README.md) — single-cell format conversion.

## Install

```bash
cargo install scx-cli
```

Requires a system HDF5 library (`libhdf5-dev` on Debian/Ubuntu, `hdf5` via Homebrew on macOS).

## Commands

```
scx convert <input> <output>   convert between h5ad and h5seurat
scx inspect <file>             summarise slots without loading the matrix
```

## License

GPL-3.
