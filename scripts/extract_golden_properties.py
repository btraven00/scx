"""Extract golden properties from h5ad fixtures using anndata (the oracle).

Run once in an environment with anndata; commit the resulting JSON. Tests then
assert picklerick reproduces these digests without needing anndata.

  pixi run -e py313 python scripts/extract_golden_properties.py

Output: tests/golden/properties/<fixture-stem>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad

# Reuse the single source of truth for the digest contract.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/picklerick/tests"))
from golden_props import build_properties  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden"
OUT = GOLDEN / "properties"

# h5ad fixtures the anndata oracle can read directly.
H5AD_FIXTURES = [
    "pbmc3k_reference.h5ad",
    "norman_subset.h5ad",
    "hlca_core.h5ad",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in H5AD_FIXTURES:
        path = GOLDEN / name
        if not path.exists():
            print(f"!! skip {name}: not found", file=sys.stderr)
            continue
        print(f"extracting {name} ...", file=sys.stderr, flush=True)
        adata = ad.read_h5ad(path)
        props = build_properties(adata)
        props["_source"] = {"tool": "anndata", "fixture": name}
        out = OUT / f"{path.stem}.json"
        out.write_text(json.dumps(props, indent=2, sort_keys=True) + "\n")
        print(f"  -> {out.relative_to(REPO)}  (X digest {props['X']['digest'][:12]}…)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
