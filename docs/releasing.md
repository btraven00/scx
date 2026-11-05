# Releasing

## Checklist

1. Record performance on the release PR's head and compare it with the
   previous release ([below](#performance-record)). Commit the result file to
   the release PR. Explain any flagged regression in the PR, or fix it first.
2. Merge the release PR (below). This tags the release and starts the conda
   and wheel builds.
3. Publish the crates: `scx-core`, then `scx-cli`.
4. Refresh the R package's lockfile, and commit it.
5. Upload the wheels to PyPI.

## scx-core, scx-cli, scx-picklerick and picklerick (R)

The crates, the Python package and the R package's Rust crate are released
together by release-please (`release-please-config.json`), with linked
versions: one number means one engine across the CLI, Python and R. After CI
passes on `main`, release-please keeps a release PR up to date; its changelog
entries come from the conventional-commit subjects. Merging it bumps every
version, creates the GitHub releases and tags (`scx-core-vX.Y.Z`,
`scx-cli-vX.Y.Z`, `picklerick-vX.Y.Z`) and builds the CLI binaries
(`.github/workflows/release-please.yml`).

It then starts two builds for the new tags: the conda packages (`scx`,
`picklerick-python`, `r-picklerick`), which `conda-package.yml` publishes to
prefix.dev's `almost-conductor` channel, and the PyPI wheels (below). Every
push to `main` also publishes conda packages to `edge`, versioned
`X.Y.Z.postN` after the latest release. To redo a release's conda packages:

```sh
gh workflow run conda-package.yml -f tag=scx-cli-vX.Y.Z
```

## Performance record

Every release records the read and write hot paths, so a gain or a
regression shows up as a number against the previous release. It runs before
a release, not on every commit: it takes about 30 minutes and needs a quiet
machine. Run it on omni. The reference numbers come from there, and numbers
from different CPUs don't compare.

```sh
CARGO_TARGET_DIR=$PWD/target python3 bench/release/release_bench.py run \
  --data /data/ben/scx-bench/ref --work /data/ben/scx-bench/tmp
python3 bench/release/release_bench.py compare \
  bench/results/release-<previous>.json bench/results/release-<this>.json
```

`run` refuses to start when the 1-minute load is above `--max-load` (default 4).
It writes two layers to `bench/results/release-<version>.json`:

- **End to end**: `scx convert` wall time and peak RSS (GNU time), median of 3,
  at 8 rayon threads. Each input is converted to BPCells h5seurat, dgCMatrix
  h5seurat and h5ad. The inputs are the gzip reference and an uncompressed copy.
- **Criterion `perf` bench** (`crates/scx-core/benches/perf.rs`): the BP-128
  kernels, each X writer, and the X reader (uncompressed and gzip). It uses
  seeded synthetic data, so it runs on any checkout. Writers and the reader run
  at 1 and 8 threads; if both take the same time, the path has gone serial.

The reference is the HLCA core h5ad (584,944 cells, gzip). It is fetched with
[hapiq](https://github.com/btraven00/hapiq) from its versioned CELLxGENE URL
and checked against a pinned sha256 (`REF_URL` and `REF_SHA256` in the script).
It is never committed. `h5repack -f NONE` (from the pixi env's `hdf5`) makes the
uncompressed copy, so both inputs hold the same values. Both are cached in
`--data` (~24 GB). To use a hapiq binary that isn't on `PATH`, set `HAPIQ`.

`compare` prints every metric and flags changes beyond 5% (`--threshold`). It
exits 1 if anything got slower or used more memory. It warns when the two files
come from different CPUs.

To backfill a release recorded before this suite existed, point `run` at that
release's binary and skip criterion:

```sh
python3 bench/release/release_bench.py run --scx /path/to/scx-0.5.0 --no-criterion --label 0.5.0
```

## Publishing the crates

From a checkout of the release tag, with a crates.io token
(`cargo login`). Publish `scx-core` first, because `scx-cli` depends on it by
version:

```sh
git checkout --detach scx-cli-vX.Y.Z
cargo publish -p scx-core
cargo publish -p scx-cli
```

crates.io never accepts the same version twice. A broken release gets
`cargo yank` and a new version.

## The R package and scx-core

`r/picklerick/src/rust` is its own crate, outside the cargo workspace. It
depends on scx-core from crates.io:

```toml
scx-core    = "X.Y.Z" # x-release-please-version
```

The marker comment lets the release PR bump this requirement with the other
versions; don't edit it by hand. Which scx-core a build uses depends on where
it runs:

| Build | scx-core |
|---|---|
| `r.yml` (PRs and `main`) | the checkout's `crates/scx-core`, patched in |
| conda `r-picklerick` (edge and releases) | the tagged tree's `crates/scx-core`, patched in |
| R-universe, Bioconductor, `R CMD INSTALL` outside CI | crates.io, at the version in `Cargo.lock` |

CI and conda add `[patch.crates-io] scx-core = { path = ... }` to cargo's
config, so R is tested against `main`, and r.yml runs on every change to
`crates/scx-core`. Neither needs the release on crates.io. r.yml fails if the
patch isn't applied. That happens when the in-tree scx-core version no longer
matches the requirement, for example after a manual edit to either version.

Builds outside the repo use the committed `Cargo.lock`. It still names the
previous scx-core after a release PR bumps the requirement, so refresh it once
the crate is on crates.io (checklist step 3):

```sh
cargo update -p scx-core --manifest-path r/picklerick/src/rust/Cargo.toml
git commit -m "chore(r): lock scx-core X.Y.Z" r/picklerick/src/rust/Cargo.lock
```

To build the R package locally against unreleased scx-core changes, put the
same patch where cargo finds it when Makevars runs (`r/picklerick/.cargo/` is
gitignored and kept out of the R build):

```sh
mkdir -p r/picklerick/.cargo
printf '[patch.crates-io]\nscx-core = { path = "%s/crates/scx-core" }\n' "$PWD" \
  > r/picklerick/.cargo/config.toml
R CMD INSTALL r/picklerick
```

## Publishing scx-picklerick to PyPI

The PyPI name is `scx-picklerick`; the import name is `picklerick`. Its
version is `python/picklerick/Cargo.toml`, which release-please bumps
(`pyproject.toml` declares `dynamic = ["version"]`). Upload is manual, from
the wheels CI builds for the release tag.

### 1. The wheels for the tag

release-please.yml starts the wheel build for the new `picklerick-vX.Y.Z`
tag. To start it by hand (tags that release-please creates with the default
token don't trigger workflows themselves):

```sh
gh workflow run python.yml --ref picklerick-vX.Y.Z
```

`.github/workflows/python.yml` builds:

- Linux x86_64 and aarch64 wheels (manylinux_2_28)
- a macOS arm64 wheel
- an sdist

Each wheel is tested against anndata and h5py from PyPI. The wheels use the
abi3 stable ABI, `cp314-abi3`, so one wheel per platform covers Python 3.14 and
every later version.

### 2. Download the artifacts

When the run is green, download the merged `dist` artifact:

```sh
gh run list --workflow python.yml --branch picklerick-vX.Y.Z
gh run download <run-id> -n dist -D dist/
ls dist/            # expect 3 wheels + 1 sdist, all with the same version
uvx twine check dist/*
```

### 3. Publish

Use a PyPI API token scoped to the `scx-picklerick` project:

```sh
UV_PUBLISH_TOKEN=pypi-... uv publish dist/*
```

To rehearse the release first, publish to TestPyPI. It needs a separate account
and token.

```sh
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-... dist/*
```

### 4. Verify

```sh
uv venv -p 3.14 /tmp/pk && . /tmp/pk/bin/activate
uv pip install scx-picklerick==X.Y.Z
python -c "import picklerick as pk; print(pk.__version__)"
```

Notes:

- PyPI never accepts the same version twice, even after you delete it. If a
  release is broken, yank it on the PyPI web UI and release a new version.
- pip only installs alpha, beta and rc versions when the user passes `--pre`
  or pins the exact version. A plain `pip install scx-picklerick` ignores them.
- Version 0.0.1 was an empty placeholder, uploaded to claim the name.
