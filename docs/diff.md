# scx diff

Status: implemented (`scx-core/src/diff.rs`, `scx diff`). Where this design and
the code differ, the code is described below under "Limits".

## Why

scx's claim is that conversions are correct. Today that rests on round-trip
tests over small fixtures. There is no tool that takes a real source file and a
real converted file and says, slot by slot, what survived.

`scx diff A B` compares two datasets in any formats scx reads. It reports per
slot whether they are identical, equal within tolerance, different (with a
summary), or present on one side only. It streams matrices, so memory stays
bounded on atlas-sized files.

This is not `scx validate`, which checks one file against a schema and reads
no matrix data. diff reads both files' data.

To avoid false positives, empty and absent slots compare equal.

## CLI

```sh
scx diff source.h5ad converted.h5seurat
scx diff a.h5ad b.h5ad --rtol 1e-6 --atol 0 --json
scx diff a.h5ad b.h5seurat --map X=data --map raw=counts --only X,obs
```

| flag | meaning |
|---|---|
| `--rtol`, `--atol` | float tolerance: `abs(a - b) <= atol + rtol * abs(b)` (numpy's `isclose`). Defaults `rtol = 1e-6`, `atol = 0`, so f32 ↔ f64 round trips pass. `--exact` sets both to 0. |
| `--only`, `--exclude` | slots to compare, same syntax as `convert --only/--exclude`. |
| `--map A=B` | pair matrix slots whose names differ (repeatable). Names: `X`, `raw`, `layers/<name>`, `obsp/<name>`. |
| `--by position \| name` | how cells and genes are matched (below). Default `name`. |
| `--json` | machine-readable report on stdout. |
| `--chunk-size N` | rows per streamed chunk (default 5000). |
| `--max-examples N` | mismatching entries listed per slot (default 5). |

Exit codes match `scx validate`: 0 equal (within tolerance), 1 different or
not comparable (a SKIPPED slot), 2 couldn't run. In a terminal, SAME is green,
other statuses red and NOTE lines yellow (off with `NO_COLOR` or when piped).

## What is compared

| slot | how |
|---|---|
| shape | `n_obs`, `n_vars`. A mismatch is reported and stops the matrix comparison; metadata is still compared. |
| obs / var index | as sets and as order. Same set, other order → reported, and with `--by name` the rest of the comparison realigns. |
| obs / var columns | per column: present on both sides, dtype class, values. See "Types" below. |
| X, layers, raw, obsp | streamed, sparse-aware: nnz, pattern (indices), values within tolerance. Reports nnz each side, entries differing, max abs and rel diff, first `N` mismatches as (row, col, a, b). |
| obsm / varm | dense; same tolerance; reported per key with max diff. Embeddings may be transposed on disk; the comparison is on the IR's (n, k). |
| uns | JSON-level comparison of keys and scalar values; nested arrays compared with tolerance. scx's own `scx_provenance` key is ignored. |

Empty and absent compare equal for layers, obsm, varm, obsp and uns.

### Types

Values compare as the IR holds them, by class rather than exact storage type:

- integers (`Int`, `Int64`): equal if values are equal. int64 → int32 (what
  h5seurat/R does) is reported only if a value doesn't fit.
- floats: tolerance as above; NaN equals NaN.
- `Float` vs `Int` with integral values: equal, reported as a dtype note.
- categoricals: compared as their decoded strings per cell, so recoding is
  fine. Level order differs → a note; `ordered` flag differs → a difference.
- `String` vs `Categorical` with the same per-cell strings: equal, dtype note.
- `Bool` vs integer 0/1: dtype note, equal.

A "note" is printed but doesn't change the exit code; a "difference" does.
Notes cover what format conversion legitimately changes (storage dtype, level
order, int64 → int32).

### Matching cells and genes

`--by name` (default): obs and var are matched by index. Genes in another
order are remapped per row, at no extra cost. Cells in another order are
realigned for obs columns and obsm; matrices are reported SKIPPED (see
"Limits"). Names only on one side are reported and skipped. Duplicated index
names fall back to `position`, with a note.

`--by position`: row i of A against row i of B, for files whose converter
renamed the index (e.g. barcodes rewritten as `Cell_1..N`; with `--by name`
that would read as "every cell missing").

### Slot names across formats

Readers expose the names of their format; the h5seurat reader already maps a
reduction `pca` to obsm `X_pca`. Matrices pair by `--map` first, then by the
counts/data rule `scx convert` applies for h5seurat
(`h5seurat::auto_counts_source`): when one side keeps its counts in raw.X or a
`counts` layer and the other has a `data` layer instead, X pairs with
`layers/data` and the counts with the other side's X. Then by equal names.
Unpaired slots are reported "only in A/B" rather than guessed.

## Output

Illustrative (made-up values):

```
scx diff source.h5ad converted.h5seurat
SAME   shape        100000 x 28024
SAME   obs index    100000 names, same order
DIFF   obs columns  65 same, 1 different, 1 only in A
         cell_type_ontology_term_id: 3 of 100000 differ, e.g. cell 17: "CL:0000057" vs "CL:000057"
         only in A: _scvi_batch
NOTE   obs columns  dataset_id: int64 vs int32 (all values fit)
SAME   X            194456499 nnz, max rel diff 0 (vs data)
SAME   raw          194456499 nnz (vs counts)
SAME   obsm         X_umap, X_scanvi_emb (max abs diff 2e-7)
ONLY A obsp         connectivities, distances

1 slot differs.
```

`--json` gives the same per slot: status, counts, max diffs, examples, notes.

## Memory and time

Metadata (obs/var/obsm/varm/uns) is read whole, as `convert` does. Each matrix
is streamed: one chunk per side at a time (`--chunk-size`, default 5000 rows),
so peak memory is about two chunks plus metadata, the same as `convert`. Time
is one read of each file's matrices; with `--by name` and permuted rows, more
seeks but the same bytes.

## Implementation

- `scx-core/src/diff.rs`: `diff(a: &mut dyn DatasetReader, b: &mut dyn
  DatasetReader, opts) -> DiffReport`, with `DiffReport`/`SlotDiff` types like
  `ValidationReport`. A pure function of two readers, so it's testable without
  files and reusable from the bindings (picklerick could expose it later; not
  in this scope).
- `scx-cli`: a `Diff` subcommand that opens both inputs with `scx_core::open`
  and prints the report.
- Matrix comparison works on chunk pairs: same row range from both readers,
  compare CSR rows. Rows are sorted first where a reader doesn't already
  (`sparse::sort_csr_indices`, a no-op on sorted rows); the h5ad reader sorts,
  the others need checking. Dense X goes through the same path, since readers
  emit it as CSR chunks.

## Tests

- Fixture round trips that must diff SAME: tiny.h5ad → h5seurat (both
  writers) → h5ad, tiny.h5ad → zarr, 10x → h5ad, `.rds` → h5ad.
- Planted differences, each reported with the right slot and example: one
  value changed beyond tolerance, one nnz added, a categorical relabelled, a
  column dropped, rows permuted (SAME with `--by name`, DIFF with
  `--by position`), an empty layer group on one side (SAME).
- Exit codes 0/1/2, `--json` shape, `--map`.

## Decisions

The draft's open questions, as implemented:

1. Tolerance: `rtol = 1e-6`, `atol = 0` by default, so f32 ↔ f64 round trips
   pass; `--exact` for bit-for-bit.
2. A storage type change with equal values is a note, not a difference.
3. `--by name` is the default.
4. uns is compared in full: keys only on one side are differences; scx's own
   `scx_provenance` is ignored.

## Limits

- `x_stream_rows` returns rows in file order, so B's rows can't be read in A's
  order. With cells in another order, X, raw, layers and obsp are reported
  SKIPPED (exit 1) rather than compared; `--by position` compares them
  row by row.
- obs/var column differences list the first differing row per column.
- A full round trip of the tiny fixture through h5seurat shows varm `PCs`
  coming back as `X_pcs`: a renaming in the h5seurat varm path, which diff
  reports as "only in A / only in B".
