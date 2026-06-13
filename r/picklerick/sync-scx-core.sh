#!/usr/bin/env bash
# Sync the vendored scx-core copy used by picklerick's native bindings.
# Run after any change to crates/scx-core/.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
src="$repo_root/crates/scx-core"
dst="$here/src/rust/scx-core"

if [ ! -d "$src" ]; then
    echo "error: $src not found" >&2
    exit 1
fi

rm -rf "$dst/src" "$dst/Cargo.toml" "$dst/build.rs"
mkdir -p "$dst"
cp -r "$src/src" "$dst/src"
cp "$src/Cargo.toml" "$dst/Cargo.toml"
# build.rs derives the HDF5-ABI cfg (hdf5_2_0) used by h5_chunk.rs; required.
cp "$src/build.rs" "$dst/build.rs"

# Strip dev-dependencies and bench targets — vendored copy is build-only.
python3 - "$dst/Cargo.toml" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()
text = re.sub(r'\n\[dev-dependencies\][\s\S]*?(?=\n\[|\Z)', '', text)
text = re.sub(r'\n\[\[bench\]\][\s\S]*?(?=\n\[|\Z)', '', text)
text = text.rstrip() + "\n\n# Vendored from crates/scx-core via sync-scx-core.sh — do not edit by hand.\n"
open(path, 'w').write(text)
PY

echo "synced $src -> $dst"
