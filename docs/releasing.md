# Releasing

## scx-core and scx-cli

These two crates are released by release-please (`release-please-config.json`).
Their versions are linked. After CI passes on `main`, release-please keeps a
release PR up to date. Merging that PR creates the tag and the GitHub release,
and builds the CLI binaries (`.github/workflows/release-please.yml`). Tags
starting with `v` also publish the conda packages to prefix.dev
(`.github/workflows/conda-package.yml`).

## Python package (`scx-picklerick` on PyPI)

The Python package is not managed by release-please. You release it by hand
from a tag, using the wheels CI builds for that tag.

The PyPI name is `scx-picklerick`. The import name is still `picklerick`.

### 1. Set the version

Cargo is the only place the version is stored. `pyproject.toml` declares
`dynamic = ["version"]`, and maturin converts the Cargo version to PEP 440:

| `python/picklerick/Cargo.toml` | PyPI version |
|---|---|
| `0.1.0-alpha.1` | `0.1.0a1` |
| `0.1.0-beta.2` | `0.1.0b2` |
| `0.1.0-rc.1` | `0.1.0rc1` |
| `0.1.0` | `0.1.0` |

```sh
$EDITOR python/picklerick/Cargo.toml        # version = "..."
cargo check -p picklerick-py-native          # refreshes Cargo.lock
git commit -am "picklerick: release 0.1.0a1"
```

### 2. Tag and push

```sh
git tag picklerick-v0.1.0a1
git push origin main picklerick-v0.1.0a1
```

The `picklerick-v*` prefix triggers `.github/workflows/python.yml`. It does not
match the conda workflow's `v*` filter, so pushing this tag doesn't publish
conda packages.

The workflow builds these artifacts:

- Linux x86_64 and aarch64 wheels (manylinux_2_28)
- a macOS arm64 wheel
- an sdist

Each wheel is tested against anndata and h5py from PyPI. The wheels use the
abi3 stable ABI, `cp314-abi3`, so one wheel per platform covers Python 3.14 and
every later version.

### 3. Download the artifacts

When the run is green, download the merged `dist` artifact:

```sh
gh run list --workflow python.yml --branch picklerick-v0.1.0a1
gh run download <run-id> -n dist -D dist/
ls dist/            # expect 3 wheels + 1 sdist, all with the same version
uvx twine check dist/*
```

### 4. Publish

Use a PyPI API token scoped to the `scx-picklerick` project:

```sh
UV_PUBLISH_TOKEN=pypi-... uv publish dist/*
```

To rehearse the release first, publish to TestPyPI. It needs a separate account
and token.

```sh
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-... dist/*
```

### 5. Verify

```sh
uv venv -p 3.14 /tmp/pk && . /tmp/pk/bin/activate
uv pip install --pre scx-picklerick==0.1.0a1
python -c "import picklerick as pk; print(pk.__version__)"
```

Notes:

- PyPI never accepts the same version twice, even after you delete it. If a
  release is broken, yank it on the PyPI web UI and release a new version.
- pip only installs alpha, beta and rc versions when the user passes `--pre`
  or pins the exact version. A plain `pip install scx-picklerick` ignores them.
- Version 0.0.1 was an empty placeholder, uploaded to claim the name.
