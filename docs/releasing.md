# Releasing

## Checklist

1. Merge the release PR (below). This tags the release and starts the conda
   and wheel builds.
2. Publish the crates: `scx-core`, then `scx-cli`.
3. Refresh the R package's lockfile, and commit it.
4. Upload the wheels to PyPI.

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
