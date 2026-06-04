# Packaging — opting out of vendored HDF5

By default SCX builds HDF5 from source and links it statically, so the
resulting binary/wheel has no system `libhdf5` dependency. Distro and conda
packagers usually ship their own `libhdf5` and want to link against it
instead of duplicating the library. Pass `--no-default-features` to disable
the `vendored-hdf5` feature (defined on `scx-core`, mirrored on `scx-cli`
and `picklerick`):

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
