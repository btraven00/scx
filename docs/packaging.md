# Packaging — HDF5 linking and the scx-core dependency

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

## R (`picklerick`) — HDF5 from Rhdf5lib, linked by R

`picklerick-r` compiles to a Rust **`staticlib`** (`libpicklerick_r.a`), which R
links into `picklerick.so` via `src/Makevars`. A Rust `staticlib` only *bundles*
native dependencies linked **statically** by cargo; its link directives for any
other library are **not** propagated to R's separate `.so` link step. Linking
HDF5 that way leaves its symbols undefined at load (`undefined symbol:
H5Sselect_elements`).

So HDF5 goes on R's link line instead, as in `rhdf5`: the package has
`LinkingTo: Rhdf5lib (>= 2.0.0)` (HDF5 1.14), and `Makevars` adds
`Rhdf5lib::pkgconfig("PKG_C_LIBS")` (its static `libhdf5.a` plus libcurl,
openssl, libaec and zlib) to `PKG_LIBS`. The R crate builds scx-core without
`vendored-hdf5`, so there is no cmake or HDF5 C build. Rhdf5lib 2.x is
Bioconductor 3.23 but installs on R 4.5, so `rproject.toml` takes it from 3.23.

Two build details in `Makevars`:

- **Version shim.** hdf5-metno-sys reads the version from `HDF5_DIR`'s headers
  but also dlopens a shared libhdf5 to check it, and Rhdf5lib ships only the
  static one. `src/rust/hdf5_version_shim.c` is a two-function stand-in for that
  check, compiled at build time and never linked into the package. It can go
  once hdf5-metno-sys skips the check for a static-only HDF5.
- **Rebuild on an Rhdf5lib upgrade.** The HDF5 version (`Rhdf5lib::getHdf5Version()`)
  is passed as `HDF5_VERSION` and stamped into the target directory, so a new
  HDF5 rebuilds the staticlib. The version picks the chunk-read ABI (`build.rs`).

The `conda.recipe/r-picklerick` recipe depends on host
`bioconductor-rhdf5lib >=2.0.0` and must **not** add a system `hdf5`.

### Offline builds (CRAN / Bioconductor)

Both build packages offline. `r/picklerick/tools/vendor.sh` vendors the crates
into `src/rust/vendor.tar.xz.NN` before `R CMD build` (4 MB parts, since
Bioconductor caps each file at 5 MB), and `Makevars` builds `--offline` from
them when present. The script stubs crates no R platform compiles, drops tests,
benches and winapi's MinGW import libraries (a Windows build must set
`WINAPI_NO_BUNDLED_LIBRARIES=1`), and keeps only enabled windows-sys modules.
The parts are generated, not committed.

## scx-core comes from crates.io

`r/picklerick/src/rust/Cargo.toml` depends on the published `scx-core`
(`scx-core = "X.Y.Z"`), not a path dep, a git rev or an in-tree copy. The R
package builds outside the cargo workspace, and R-universe and Bioconductor ship
only the package directory, so a path dep to `crates/scx-core` would not resolve
there.

- **The version moves with the release.** The line carries an
  `x-release-please-version` marker, so the release PR bumps it with the rest of
  the linked scx group.
- **CI and conda build against the in-tree crate.** `r.yml` and the
  `r-picklerick` conda recipe add `[patch.crates-io] scx-core = { path = ... }`
  to cargo's config. So `main` is tested against `main`, and a release's conda
  package doesn't wait for `cargo publish`. r.yml fails if the patch isn't
  applied, which happens when the in-tree version stops matching the requirement.
- **The committed `Cargo.lock` is required.** The R crate must not depend on
  `ndarray` itself. hdf5-metno accepts several ndarray versions, so a second
  one in the graph resolves hdf5-metno against it, and scx-core then fails to
  compile with `Selection`/`SliceInfo` trait errors. Keep hdf5-metno and
  ndarray at the versions in the workspace `Cargo.lock`.
- There is **no** vendored `r/picklerick/src/rust/scx-core` copy and **no**
  `sync-scx-core.sh`. Don't reintroduce them.
- To build locally against an unreleased scx-core, add the same patch for one
  command: `cargo --config 'patch.crates-io.scx-core.path="/path/to/scx/crates/scx-core"' ...`.
