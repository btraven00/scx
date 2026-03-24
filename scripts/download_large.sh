#!/usr/bin/env bash
# download_large.sh
#
# Downloads the Human Lung Cell Atlas (HLCA core, 584k cells) from CellxGene
# for use as a large-scale benchmark fixture.
#
# Usage:
#   bash scripts/download_large.sh
#
# Output: tests/golden/hlca_core.h5ad

set -euo pipefail

URL="https://datasets.cellxgene.cziscience.com/7bcad396-49c3-40d9-80c1-16d74e7b88bd.h5ad"
OUT="tests/golden/hlca_core.h5ad"
EXPECTED_CELLS=584944

mkdir -p tests/golden

if [ -f "$OUT" ]; then
  echo "Already exists: $OUT — delete it to re-download."
  exit 0
fi

echo "Downloading HLCA core (584k cells, ~2 GB) from CellxGene..."
echo "  -> $OUT"
echo ""

curl --progress-bar -L "$URL" -o "${OUT}.tmp"
mv "${OUT}.tmp" "$OUT"

SIZE_MB=$(( $(stat -c%s "$OUT") / 1048576 ))
echo ""
echo "Done: $OUT (${SIZE_MB} MB)"
echo ""
echo "Quick check:"
python3 - << EOF
import h5py, sys
with h5py.File("$OUT") as f:
    shape = f["X"].attrs.get("shape", None)
    if shape is None and "X" in f:
        # Try reading shape from group attrs
        shape = list(f["X"].attrs.get("shape", []))
    enc = f.attrs.get("encoding-type", "unknown")
    print(f"  encoding : {enc}")
    print(f"  X shape  : {list(shape) if shape is not None else 'see X group'}")
    print(f"  obs keys : {list(f['obs'].keys())[:8]}")
    print(f"  obsm keys: {list(f.get('obsm', {}).keys())}")
EOF
