#!/usr/bin/env bash
# Build the docs site into site/_site. Used locally and by
# .github/workflows/docs.yml.
#
#   scripts/render_docs.sh            render the whole site
#   scripts/render_docs.sh preview    serve it, re-rendering each page on save
#
# Installs the current scx CLI, R package and Python package, then renders
# the Quarto site with each language in its own environment: knitr runs R
# and shell chunks with $RSCRIPT, Jupyter runs Python cells with the docs
# venv. Every chunk runs on every render, so a broken example fails here.
#
# Environment:
#   RSCRIPT    Rscript to render R pages with (default: Rscript on PATH).
#   R_LIBS     R library holding the docs dependencies (default: the rv
#              project library, when rv is installed).
#   DOCS_VENV  Python venv made from site/requirements.txt
#              (default: .venv-docs).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

rscript=${RSCRIPT:-$(command -v Rscript)}
r_bin=$(dirname "$rscript")
if [ -z "${R_LIBS:-}" ] && command -v rv >/dev/null; then
  export R_LIBS="$root/$(rv library)"
fi
venv=${DOCS_VENV:-$root/.venv-docs}
if [ ! -x "$venv/bin/python" ]; then
  echo "No Python venv at $venv: python3 -m venv $venv && $venv/bin/pip install -r site/requirements.txt" >&2
  exit 1
fi

echo "==> scx CLI"
cargo build --release -p scx-cli
export PATH="$root/target/release:$PATH"

echo "==> R package"
# Build against the checked-out scx-core rather than the crates.io release
# r/picklerick names, as r.yml does. The .cargo directory is git-ignored.
cargo_cfg=r/picklerick/.cargo/config.toml
if [ ! -f "$cargo_cfg" ]; then
  mkdir -p "$(dirname "$cargo_cfg")"
  printf '[patch.crates-io]\nscx-core = { path = "%s/crates/scx-core" }\n' "$root" > "$cargo_cfg"
fi
"$r_bin/R" CMD INSTALL --no-test-load r/picklerick

echo "==> Python package"
VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" \
  maturin develop --release --manifest-path python/picklerick/Cargo.toml

echo "==> Python reference"
(cd site && "$venv/bin/quartodoc" build)

echo "==> Example files"
export SCX_DOCS_WORK="$root/site/_work"
rm -rf "$SCX_DOCS_WORK"
cp -R tests/fixtures/tiny "$SCX_DOCS_WORK"

echo "==> Site"
export QUARTO_R="$rscript" QUARTO_PYTHON="$venv/bin/python"
# Plain scx log lines on the page, not ANSI colour escapes.
export NO_COLOR=1
quarto render site

echo "==> R site"
# After the Quarto render, which clears site/_site. install = FALSE: the
# package was installed above.
"$rscript" -e 'pkgdown::build_site("r/picklerick", install = FALSE, preview = FALSE,
                 override = list(destination = "../../site/_site/r"))'

if [ "${1:-}" = preview ]; then
  # --no-render keeps the site built above (with its r/ part); pages are
  # then re-rendered one at a time as they are saved.
  quarto preview site --no-render
else
  echo "Site written to site/_site"
fi
