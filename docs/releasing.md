# Releasing

## scx-core, scx-cli and scx-picklerick

The two crates and the Python package are released together by release-please
(`release-please-config.json`), with linked versions: one number means one
engine across the CLI, Python and (later) R. After CI passes on `main`,
release-please keeps a release PR up to date; its changelog entries come from
the conventional-commit subjects. Merging it bumps every version, creates the
GitHub releases and tags (`scx-core-vX.Y.Z`, `scx-cli-vX.Y.Z`,
`picklerick-vX.Y.Z`) and builds the CLI binaries
(`.github/workflows/release-please.yml`). Tags starting with `v` also publish
the conda packages to prefix.dev (`.github/workflows/conda-package.yml`).

## Publishing scx-picklerick to PyPI

The PyPI name is `scx-picklerick`; the import name is `picklerick`. Its
version is `python/picklerick/Cargo.toml`, which release-please bumps
(`pyproject.toml` declares `dynamic = ["version"]`). Upload is manual, from
the wheels CI builds for the release tag.

### 1. Build the wheels for the tag

Tags that release-please creates with the default token don't trigger other
workflows (GitHub prevents that), so start the wheel build by hand:

```sh
gh workflow run python.yml --ref picklerick-v0.4.0
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
gh run list --workflow python.yml --branch picklerick-v0.4.0
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
uv pip install scx-picklerick==0.4.0
python -c "import picklerick as pk; print(pk.__version__)"
```

Notes:

- PyPI never accepts the same version twice, even after you delete it. If a
  release is broken, yank it on the PyPI web UI and release a new version.
- pip only installs alpha, beta and rc versions when the user passes `--pre`
  or pins the exact version. A plain `pip install scx-picklerick` ignores them.
- Version 0.0.1 was an empty placeholder, uploaded to claim the name.
