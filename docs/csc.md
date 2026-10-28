# Gene-major X: transpose once, stream by gene

## The pattern

Per-gene work (one GLM per gene, per-gene dispersion, gene-wise tests)
needs all of a gene's cells at once, many times over. Most files store cells
as rows (h5ad CSR, 10x, Seurat, BPCells). In memory, `adata.X.tocsc()` holds
the CSR and the CSC at once, about twice the matrix. At atlas scale that's
beyond a workstation.

scx splits the job into two steps, each with bounded memory:

```sh
# 1. Once per dataset: write a gene-major copy, holding at most 8 GB.
scx convert atlas.h5ad atlas_csc.h5ad --csc 8
```

```python
# 2. Every fit: stream blocks of whole genes.
import picklerick as pk

for chunk in pk.open_stream("atlas_csc.h5ad", axis="var", chunk_size=512):
    Y = chunk.to_scipy()   # csc_array, cells x genes [var_offset, var_offset + ncols)
    fit(Y)                 # the next block decodes on a background thread meanwhile
```

To split the genes across jobs (SLURM array tasks, GPUs), give each job its
own range:

```python
i, n_jobs = int(os.environ["SLURM_ARRAY_TASK_ID"]), 16
n_vars = pk.inspect("atlas_csc.h5ad")["n_vars"]
lo, hi = i * n_vars // n_jobs, (i + 1) * n_vars // n_jobs
for chunk in pk.open_stream("atlas_csc.h5ad", axis="var", genes=[(lo, hi)]):
    ...
```

Each job reads only its genes' slices of the file.

Step 1 costs `1 + blocks` reads of X, once per dataset. Step 2 costs one read
of the selected genes per fit, and holds a few blocks of `chunk_size` genes.
If one fit runs over the data only once, step 1 doesn't pay off. Stream cells
with `open_stream` and accumulate per-gene statistics instead.

The gene-major copy is a plain h5ad (`csc_matrix` X), so other tools read it
too. `anndata.read_h5ad` returns a `csc_matrix`, backed `adata.X[:, genes]`
reads only those genes, and HDF5Array reads it by column in R.

## Step 1: `scx convert --csc`

```sh
scx convert atlas.h5ad atlas_csc.h5ad --csc        # 4 GB budget
scx convert atlas.h5ad atlas_csc.h5ad --csc 0.5    # 0.5 GB budget
```

Any input scx reads works, and the output must be `.h5ad`. obs, var and the
other slots are written as usual. Layers and obsp stay CSR. From Rust, it's
`scx_core::transpose::write_x_csc(reader, writer, mem_bytes)`.

How it works:

1. **Count.** Stream the cell chunks and count each gene's nonzeros. That
   count is the CSC `indptr`.
2. **Block.** Split the genes into consecutive blocks whose nonzeros fit the
   budget. A gene with more nonzeros than the budget gets a block of its own.
3. **Fill.** For each block, stream X again and keep only that block's
   genes. Each value goes straight to its gene's next free slot. Cells arrive
   in order, so they come out sorted within each gene. Append the block to
   the output.

The budget covers the block being filled: cell id, value, and the i32 copy
of the ids made on write. One reader chunk (`--chunk-size` cells) and the
`n_vars + 1` indptr come on top. Writes only ever append, so `--compress`
works too. If a later read sees different nonzeros from the counting read,
the conversion fails rather than writing a wrong matrix.

Cost on a synthetic 200,000 × 2,000 f32 matrix, 5% dense (20M nonzeros,
160 MB as CSR, 169 MB h5ad), release build, one machine:

| run                                | blocks | reads | wall   | peak RSS |
|------------------------------------|-------:|------:|-------:|---------:|
| `scx convert` (CSR, baseline)      |      — |     1 | 0.55 s |    56 MB |
| `scx convert --csc 4`              |      1 |     2 | 0.69 s |   295 MB |
| `scx convert --csc 0.1`            |      3 |     4 | 0.93 s |   158 MB |
| `scx convert --csc 0.04`           |      7 |     8 | 1.13 s |   100 MB |
| `scx convert --csc 0.01`           |     25 |    26 | 2.71 s |    61 MB |
| anndata `read` → `tocsc` → `write` |      — |     — | 1.13 s |   453 MB |

Peak RSS above the CSR baseline tracks the budget. Halving the budget
roughly doubles the blocks. On real atlases, decoding X dominates each read,
so the wall time grows roughly with `1 + blocks`. Give it what the machine
can spare. If `2 × matrix` fits in RAM, one block is as fast as the
in-memory route.

## Step 2: `open_stream(axis="var")`

- `chunk_size` counts genes. Each chunk spans every cell: `row_offset` is 0,
  `nrows` is `n_obs`, and `var_offset`/`ncols` give its genes.
- `chunk.to_scipy()` is a `csc_array` sharing `data` and `indices` with the
  chunk, so there's no copy. `chunk.indptr` holds per-gene pointers, and
  `chunk.indices` holds cell ids, ascending within each gene.
- `genes=` takes what `rows=` takes: `(start, stop)` ranges, indices or a
  boolean mask. Chunks never span a gap in the selection.
- `rows=` and `where=` select cells, so they're refused with `axis="var"`.
  Subset cells inside the fit, or write a filtered copy first.

The storage has to match the axis, and the mismatch is an error, not a slow
path:

- **`axis="var"` on X stored by cell:** an error pointing to `scx convert
  --csc`. Reading all of X for every gene block would be a full pass per
  block.
- **`axis="obs"` (the default) on CSC X:** an error. Reading the gene
  pointer as a row pointer would transpose the matrix silently.

In Rust, step 2 is `DatasetReader::x_stream_cols(ranges)`. Only `H5AdReader`
on CSC X implements it, and every other reader returns the error above.

## Limits

- Gene-axis streaming reads h5ad CSC X only. Zarr CSC X and the R binding
  aren't wired; in R, HDF5Array already reads CSC h5ad by column.
- `.h5ad` output only. Zarr, h5seurat and BPCells output refuse `--csc`.
- `scx convert` from a CSC h5ad fails at X, because it streams by cell.
  Convert to CSR with anndata first.
- Every read in step 1 decodes all of X. A single-read alternative would
  spill per-block buckets to temporary files and read each one back. Add it
  if the number of reads ever dominates.
