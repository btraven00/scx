# Gene-major X (`scx convert --csc`)

## Motivation

Per-gene consumers, such as one GLM per gene, want each gene's cells
contiguous. Most files store cells as rows (h5ad CSR, 10x, Seurat, BPCells),
so something has to transpose. In memory, `adata.X.tocsc()` holds the CSR and
the CSC at once, about twice the matrix. At atlas scale that's beyond a
workstation.

`--csc` writes X as anndata's `csc_matrix` with memory bounded by a budget
you choose. The output is a plain h5ad: `anndata.read_h5ad` returns a
`csc_matrix`, and in backed mode `adata.X[:, genes]` reads only those genes.

## Usage

```sh
scx convert atlas.h5ad atlas_csc.h5ad --csc        # 4 GB budget
scx convert atlas.h5ad atlas_csc.h5ad --csc 0.5    # 0.5 GB budget
```

Any input scx reads works. Output must be `.h5ad`. obs, var and the other
slots are written as usual, and layers and obsp stay CSR.

From Rust, the same thing is `scx_core::transpose::write_x_csc(reader,
writer, mem_bytes)`.

## How it works

Two passes over X, then one more per block:

1. **Count.** Stream the cell chunks and count each gene's nonzeros. That
   count is the CSC `indptr`.
2. **Block.** Split the genes into consecutive blocks whose nonzeros fit the
   budget. A gene with more nonzeros than the budget gets a block of its own.
3. **Fill.** For each block, stream X again and keep only that block's
   genes. Each value goes straight to its gene's next free slot. Cells arrive
   in order, so they come out sorted within each gene. Append the block to
   the output.

So X is read `1 + blocks` times. When X fits the budget, that's one block and
two reads. The budget covers the block being filled (cell id, value, and the
i32 copy of the ids made on write). One reader chunk (`--chunk-size` cells)
and the `n_vars + 1` indptr come on top. Writes only ever append, so `--compress`
works too.

If a later pass sees different nonzeros from the counting pass, the
conversion fails rather than writing a wrong matrix.

## Cost

Synthetic 200,000 × 2,000 f32, 5% dense (20M nonzeros, 160 MB as CSR, 169 MB
h5ad), release build, one machine:

| run                               | blocks | reads | wall   | peak RSS |
|-----------------------------------|-------:|------:|-------:|---------:|
| `scx convert` (CSR, baseline)     |      — |     1 | 0.55 s |    56 MB |
| `scx convert --csc 4`             |      1 |     2 | 0.69 s |   295 MB |
| `scx convert --csc 0.1`           |      3 |     4 | 0.93 s |   158 MB |
| `scx convert --csc 0.04`          |      7 |     8 | 1.13 s |   100 MB |
| `scx convert --csc 0.01`          |     25 |    26 | 2.71 s |    61 MB |
| anndata `read` → `tocsc` → `write` |      — |     — | 1.13 s |   453 MB |

Peak RSS above the CSR baseline tracks the budget. Halving the budget
roughly doubles the blocks. On real atlases, decoding X dominates each read,
so the wall time grows roughly with `1 + blocks`.

## Choosing a budget

Give it what the machine can spare. More budget means fewer reads. If
`2 × matrix` fits in RAM, a single block is as fast as the in-memory route.
The budget only matters when it doesn't fit.

## Limits

- **scx can't read the result's X back yet.** `H5AdReader` refuses CSC X,
  because its row streams would have to transpose it silently. A gene-axis
  stream (`open_stream(axis="var")` over a CSC file) is the natural
  follow-up. Until then, read the output with anndata, or with h5py or
  HDF5Array by column.
- `.h5ad` only. Zarr, h5seurat and BPCells output refuse `--csc`.
- Every pass decodes all of X. A single-pass alternative would spill
  per-block buckets to temporary files and then read each one back. Add it if
  the number of reads ever dominates.
