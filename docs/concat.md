# scx concat

## Motivation

Multi-sample experiments arrive as one h5ad per sample. `scx concat` stacks
them along the obs (cell) axis into a single h5ad, aligning genes by name.

It is the counterpart to [`scx merge`](merge.md): merge overlays *slots* onto a
base file with the same cells; concat adds *cells*.

Semantics follow `anndata.concat()` — same flag names, same defaults, same
NA-fill policy — so a pipeline can swap one for the other. The output is
verified against `anndata.concat()` for X, var/obs index order, obs columns and
the `label` column on both join modes — see `tests/python/test_concat.py`
(`pixi run -e test verify-python`), which builds its own fixtures and needs no
golden files.

---

## Usage

```sh
scx concat sample_*.h5ad -o atlas.h5ad --label sample
```

```sh
scx concat a.h5ad b.h5ad c.h5ad \
  -o atlas.h5ad \
  --join outer \
  --label sample \
  --keys donor1,donor2,donor3 \
  --index-unique - \
  --merge same \
  --compress 6
```

| Flag | Default | Meaning |
|---|---|---|
| `--join` | `inner` | `inner` keeps genes present in every input; `outer` takes the union and treats absent genes as zeros |
| `--label` | — | Name of an obs column recording each cell's source dataset (categorical) |
| `--keys` | file stems | One source name per input; used for `--label` and `--index-unique`. Error if the stems collide and no keys are given |
| `--index-unique` | — | Separator appended to obs_names: `cell1` + `-` + `donor2` → `cell1-donor2` |
| `--merge` | `none` | How to carry var columns over: `none`, `same`, `unique`, `first`, `only` |
| `--dtype` | `f32` | Output X dtype |
| `--chunk-size` | `5000` | Cells per streaming chunk |
| `--compress` | off | gzip level for the output |

Inputs may be any format `scx` reads (h5ad, h5seurat, 10x, mtx, BPCells,
NPY snapshots); output is always h5ad.

---

## What gets carried over

| Slot | Behaviour |
|---|---|
| `X` | Streamed chunk-by-chunk, column indices remapped onto the joined gene axis |
| `layers` | Same, per layer. Layer names follow `--join`; under `outer`, an input without a layer contributes zero rows |
| `obs` | Concatenated. Column names follow `--join` |
| `obs` index | Concatenated, optionally suffixed by `--index-unique`. Duplicates are warned about, not rejected |
| `var` | The joined gene index, plus whatever `--merge` keeps |
| `obsm` | Row-stacked; keys follow `--join`, missing keys NaN-filled |
| `uns` | Only `uns["scx_concat"]` — the input list, keys, per-input `n_obs`, join mode, timestamp |
| `obsp`, `varp`, `varm` | Not carried. `obsp` matches anndata's `pairwise=False` default; `varm`/`varp` are not on the streaming reader/writer path |

### NA fill

An obs column missing from one input is filled for that input's cells:
float → `NaN`, int → `0`, bool → `false`, string → `""`, categorical → an `"NA"`
level (created only when actually needed). This matches the merge path.

### Mixed dtypes

The same obs column with different dtypes across inputs is promoted:
int + float → float, string + categorical → categorical (levels unified),
anything else → string. This mirrors pandas' fallback to `object`.

---

## Memory

X and layers stream — peak memory is one chunk per input, not one dataset.
obs, var and obsm are materialised, as everywhere else in scx: the
`DatasetWriter` API takes whole frames. A concat of *N* inputs holds *N* open
readers, each with its X indptr (8 bytes per cell) resident.
