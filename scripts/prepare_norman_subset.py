"""
Subsample the Norman perturbation H5AD to a small committed test fixture.

Usage:
    python scripts/prepare_norman_subset.py \
        --input  /path/to/norman_perturbation.h5ad \
        --output tests/golden/norman_subset.h5ad \
        [--n-obs 500] [--n-vars 200] [--seed 42]
"""
import argparse
import numpy as np
import anndata as ad

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", default="tests/golden/norman_subset.h5ad")
    p.add_argument("--n-obs",  type=int, default=500)
    p.add_argument("--n-vars", type=int, default=200)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"reading {args.input} ...")
    adata = ad.read_h5ad(args.input)
    print(f"  full shape: {adata.shape}")

    n_obs  = min(args.n_obs,  adata.n_obs)
    n_vars = min(args.n_vars, adata.n_vars)

    cell_idx = np.sort(rng.choice(adata.n_obs,  size=n_obs,  replace=False))
    gene_idx = np.sort(rng.choice(adata.n_vars, size=n_vars, replace=False))

    sub = adata[cell_idx, gene_idx].copy()
    print(f"  subset shape: {sub.shape}")
    print(f"  obs columns: {list(sub.obs.columns)}")
    print(f"  layers: {list(sub.layers.keys())}")

    sub.write_h5ad(args.output)
    print(f"written to {args.output}")

if __name__ == "__main__":
    main()
