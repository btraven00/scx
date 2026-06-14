# Packaging — HDF5 linking and the scx-core git dependency

## Two HDF5 link modes

`scx-core` exposes a `vendored-hdf5` feature (default on; mirrored on `scx-cli`
and used by `picklerick`). It controls **how** HDF5 + zlib are linked, and the
choice is load-bearing for distribution:

- **Vendored / static (default).** HDF5 + zlib are built from source and linked
  statically, so the binary/wheel has no system `libhdf5` dependency. Requires
  `cmake` + a C/C++ compiler at build time (the from-source HDF5 build,
  `hdf5-metno-src`, drives cmake).
- **System / dynamic (`--no-default-features`).** Links the system `libhdf5`,
  resolved via `HDF5_DIR` or `pkg-config`. Then `libhdf5` is required at build
  *and* run time.

> ABI note: the raw chunk-read symbol is HDF5-version-specific. `scx-core`'s
> `build.rs` derives an `hdf5_2_0` cfg from the **detected** library version
> (`DEP_HDF5_VERSION_2_0_0`), so both link modes pick the right symbol — do not
> gate that on the `vendored-hdf5` feature. See `crates/scx-core/src/h5_chunk.rs`.

## CLI (`scx`) and Python (`picklerick`)

These can link either way. Distro/conda packagers usually opt out of vendoring
to share the system `libhdf5`:

| Build path | Opt-out command |
|---|---|
| `cargo install scx-cli` | `cargo install scx-cli --no-default-features` |
| `cargo build` from source | `cargo build --release -p scx-cli --no-default-features` |
| Python wheel (maturin/pip) | `MATURIN_PEP517_ARGS=--no-default-features pip install python/picklerick` |

The `conda.recipe/scx` and `conda.recipe/picklerick-python` recipes pass
`--no-default-features` and depend on host `hdf5`, so they are working examples
of the system-HDF5 path.

## R (`picklerick`) — MUST use vendored-static HDF5

**The R package cannot use the system/dynamic HDF5 path.** This is the
non-obvious constraint that bit us (load-time `undefined symbol:
H5Sselect_elements`):

`picklerick-r` compiles to a Rust **`staticlib`** (`libpicklerick_r.a`), which R
then links into `picklerick.so` via `src/Makevars` (`PKG_LIBS = $(STATLIB) -ldl
-lm`). A Rust `staticlib` only *bundles* native dependencies that are linked
**statically**; cargo's link directives for a *dynamic* library are **not**
propagated to R's separate `.so` link step, and `Makevars` passes no `-lhdf5`.
So if HDF5 is linked dynamically (e.g. `HDF5_DIR` set, or a host `hdf5` dep),
the HDF5 symbols are simply absent from `picklerick.so` and it fails to load.

Therefore the `conda.recipe/r-picklerick` recipe **must**:

- **not** set `HDF5_DIR` and **not** depend on system `hdf5`/`zlib` (host or
  run) — let `vendored-hdf5` build and bundle them statically;
- include `cmake` and a C++ compiler in build requirements (for the from-source
  HDF5 build).

This also isolates picklerick's HDF5 from R's `rhdf5` / `hdf5r` (avoiding the
ABI conflicts that motivated static vendoring in the first place).

## scx-core is a git dependency, not vendored

`r/picklerick/src/rust/Cargo.toml` depends on `scx-core` via a **pinned git
dependency** on `btraven00/scx` (not a path dep, not an in-tree copy). The R
package builds outside the cargo workspace and R-universe ships only the package
directory, so a path dep to `crates/scx-core` would not resolve there; cargo
fetches the pinned rev at build time instead (build environments have network).

Consequences:
- There is **no** vendored `r/picklerick/src/rust/scx-core` copy and **no**
  `sync-scx-core.sh` — both were removed. Don't reintroduce them.
- A committed `r/picklerick/src/rust/Cargo.lock` is **required**: a fresh
  resolve can pull `hdf5-metno 0.12.5` → `ndarray 0.17` alongside scx-core's
  `0.16`, causing a `Selection`/`SliceInfo` trait skew. Keep `hdf5-metno` pinned
  to a single-ndarray combination (currently `0.12.4`).
- **Bump the pinned rev (or use a release tag) when scx-core changes** that the
  R bindings need — local edits to `crates/scx-core` are not seen until pushed
  and the pin is updated.
