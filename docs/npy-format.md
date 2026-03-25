# NPY Exchange Format (0.0.7)

Raw-binary IR snapshots for benchmarking isolation, debugging, and as the
foundation of a zero-copy R conversion path.

---

## Motivation

HDF5 format overhead — schema validation, chunk bookkeeping, compression — can
dominate wall time for small-to-medium datasets and makes micro-benchmarking
noisy.  More critically, the current R conversion path copies data three times:

```
H5Seurat file
  → Rust heap (Vec<T> per CSR component)
  → temp H5AD file
  → anndataR parse → R dgCMatrix on GC heap
```

A flat binary snapshot eliminates the intermediate file and reduces this to a
single copy (OS page cache → R heap).  It also gives a standalone checkpoint
you can inspect with `numpy.load()` from any language.

**Use cases:**

- **Benchmark isolation** — `scx snapshot pbmc.h5seurat ir/` then
  `scx convert ir/ out.h5ad` measures H5AD write speed without H5Seurat read
  overhead in the loop.
- **Debugging** — dump the IR to disk, inspect arrays in Python or R without
  HDF5 tooling.
- **Test fixtures** — generate synthetic IRs from Rust, verify readers in
  Python/R without needing HDF5 files.
- **R fast path** — mmap the snapshot arrays directly into R SEXP objects
  (planned, see §mmap plan below).

---

## Relationship to spec.md §15

`spec.md` §15 sketched a `.scxd` format using a nested directory structure
(`X/data.npy`, `obsp/connectivities/data.npy`) and named the artifact
`.scxd`.  The implementation uses a **flat layout** instead:

| spec.md vision | implemented |
|---|---|
| `X/data.npy` | `X_data.npy` |
| `obsp/connectivities/data.npy` | `obsp_connectivities_data.npy` |
| `obsm/X_pca.npy` | `obsm_X_pca.npy` |
| `.scxd` extension required | any directory; `meta.json` presence is the signal |

Rationale: the flat layout avoids creating one subdirectory per slot key, is
simpler to iterate over, and keeps all files at one `readdir` depth.  The
mmap semantics from the spec are unchanged — each file is independently
mmap-able at offset 128.  The spec's `meta.json` per-array shape/dtype
metadata is preserved.

The `.scxd` convention can still be used as a naming recommendation for
snapshot directories (e.g. `pbmc3k.scxd/`) but is not enforced.

---

## File layout

```
ir_snapshot/                      ← any directory name; conventionally *.scxd
  meta.json                        # manifest (see schema below)

  X_data.npy                       # (nnz,)      f32|f64|i32|u32
  X_indices.npy                    # (nnz,)      u32
  X_indptr.npy                     # (n_obs+1,)  u64

  obs_index.txt                    # n_obs lines, one barcode per line
  var_index.txt                    # n_vars lines, one gene name per line

  obs_{col}.npy                    # numeric obs column  (int → <i4, float → <f8)
  obs_{col}.npy                    # bool obs column     (|b1)
  obs_{col}_strings.txt            # string obs column, one value per line
  obs_{col}_codes.npy              # categorical obs column: codes (u32, <u4)
  obs_{col}_levels.txt             # categorical obs column: levels, one per line

  var_{col}.npy / _strings.txt / _codes.npy / _levels.txt   # same as obs_*

  obsm_{key}.npy                   # (n_obs, k)   f64  dense, C-contiguous
  varm_{key}.npy                   # (n_vars, k)  f64  dense, C-contiguous

  layers_{name}_data.npy           # layer CSR — same dtype as X
  layers_{name}_indices.npy        # (nnz,)  u32
  layers_{name}_indptr.npy         # (n_obs+1,) u64

  obsp_{name}_data.npy             # pairwise obs CSR
  obsp_{name}_indices.npy
  obsp_{name}_indptr.npy

  varp_{name}_data.npy             # pairwise var CSR
  varp_{name}_indices.npy
  varp_{name}_indptr.npy

  uns.json                         # serde_json dump of UnsTable.raw (omitted if empty)
```

### Naming rules

- `{col}` and `{key}` / `{name}` are used verbatim from the IR.  No
  sanitisation is applied; column names that contain filesystem-illegal
  characters (e.g. `/`) will fail at write time with an I/O error.  In
  practice, single-cell column names (`nCount_RNA`, `orig.ident`,
  `seurat_clusters`) are always filesystem-safe on Linux.
- `meta.json` always written last.  A directory without `meta.json` is not a
  valid snapshot.

---

## `meta.json` schema

```json
{
  "n_obs": 2638,
  "n_vars": 1838,
  "x_dtype": "f32",
  "x_present": true,
  "obs_index_present": true,
  "var_index_present": true,
  "obs_columns": [
    { "name": "nCount_RNA",    "kind": "float" },
    { "name": "orig.ident",    "kind": "categorical" },
    { "name": "active",        "kind": "bool" },
    { "name": "notes",         "kind": "string" },
    { "name": "n_genes",       "kind": "int" }
  ],
  "var_columns": [
    { "name": "highly_variable", "kind": "bool" }
  ],
  "obsm_keys":   ["X_pca", "X_umap"],
  "varm_keys":   ["PCs"],
  "layers_keys": ["spliced", "unspliced"],
  "obsp_keys":   ["connectivities", "distances"],
  "varp_keys":   [],
  "uns_present": false
}
```

`x_dtype` is one of `"f32"`, `"f64"`, `"i32"`, `"u32"`.  Column `kind` is one
of `"int"`, `"float"`, `"bool"`, `"string"`, `"categorical"`.

The reader uses `meta.json` as the sole source of truth for which files to
expect.  Any `.npy` file not referenced in `meta.json` is ignored.

---

## NPY format (v1.0)

Each `.npy` file is NumPy format v1.0.  No external crate is required to
produce or consume it.

### Header layout

```
offset 0  : \x93NUMPY               (6 bytes — magic)
offset 6  : \x01 \x00               (2 bytes — major=1, minor=0)
offset 8  : header_len              (2 bytes — u16 LE)
offset 10 : header string           (header_len bytes)
offset 10+header_len : raw data     (C-contiguous, little-endian)
```

The header string is a Python dict literal, space-padded, terminated with `\n`.
The constraint `(10 + header_len) % 64 == 0` must hold.  For all arrays SCX
produces, this results in a 128-byte total header (10 prefix + 118 header
string), so **data always starts at offset 128**.

This fixed offset is the key property for mmap: after parsing `meta.json` once
to get shape and dtype, the data region of any `.npy` file starts at byte 128.

### Dtype tags used

| IR type    | `.npy` descr | notes |
|------------|--------------|-------|
| `f32`      | `<f4`        | X data when dtype=F32 |
| `f64`      | `<f8`        | X data when dtype=F64; obsm/varm always f64 |
| `i32`      | `<i4`        | int obs/var columns |
| `u32`      | `<u4`        | X indices; categorical codes |
| `u64`      | `<u8`        | X indptr (supports >4B nnz) |
| `bool`     | `\|b1`       | bool obs/var columns (1 byte, 0x00 or 0x01) |

All arrays are little-endian, C-contiguous (`fortran_order: False`).

---

## Slot filter

`NpyIrWriter::write` accepts a `SlotFilter` that controls which slots are
materialised.  The same syntax is used by the CLI `--only` / `--exclude`
flags.

### Specifiers

| specifier       | matches |
|-----------------|---------|
| `X`             | X matrix (data, indices, indptr) |
| `obs_index`     | obs barcode index |
| `var_index`     | var gene name index |
| `uns`           | unstructured metadata |
| `obs`           | all obs columns |
| `obs:nCount_RNA`| specific obs column |
| `var`           | all var columns |
| `var:col_name`  | specific var column |
| `obsm`          | all obsm embeddings |
| `obsm:X_pca`    | specific obsm key |
| `varm`          | all varm |
| `varm:PCs`      | specific varm key |
| `layers`        | all layers |
| `layers:spliced`| specific layer |
| `obsp`          | all obsp |
| `obsp:connectivities` | specific obsp key |
| `varp`          | all varp |
| `varp:key`      | specific varp key |

`obs` matches `obs:anything` (prefix rule); `obs:nCount_RNA` matches only that
column.  Exclusions take priority over `--only`.

### CLI examples

```bash
# Everything (default)
scx snapshot pbmc.h5seurat ir/

# X matrix + obs barcodes only (minimal for write benchmarks)
scx snapshot pbmc.h5seurat ir/ --only X,obs_index

# X + specific obs columns + all embeddings
scx snapshot pbmc.h5seurat ir/ --only X,obs:nCount_RNA,obs:orig.ident,obsm

# Everything except layers and pairwise matrices
scx snapshot pbmc.h5seurat ir/ --exclude layers,obsp,varp

# Feed the snapshot into a convert benchmark (no H5Seurat read overhead)
scx convert ir/ out.h5ad
```

### Rust API

```rust
// Write
NpyIrWriter::write(dir, &dataset, &SlotFilter::all())?;
NpyIrWriter::write(dir, &dataset, &SlotFilter::from_only("X,obs_index"))?;
NpyIrWriter::write(dir, &dataset, &SlotFilter::from_exclude("layers,obsp"))?;

// Read (one-shot)
let reader = NpyIrReader::open(dir, chunk_size)?;
let dataset: SingleCellDataset = reader.into_dataset();

// Read (streaming — implements DatasetReader)
let mut reader = NpyIrReader::open(dir, 5000)?;
// use as any other DatasetReader: reader.obs().await?, reader.x_stream(), ...
```

`NpyIrReader` fully implements `DatasetReader` so it composes directly with
`scx convert` and the `convert_with_reader` path in the CLI.  The X matrix is
loaded into memory at `open()` time; `x_stream()` yields row-chunks from the
in-memory CSR via Arc-shared slices (no per-chunk re-read from disk).

---

## Implementation notes

### No external crate

The NPY header is written by `write_npy_header` in `npy.rs` (~20 lines).  The
reader parses the Python dict literal with simple string searching.  Neither
path needs `ndarray-npy`, `numpy`, or any other crate beyond `std`.

### Byte casting

Raw data is written/read via unsafe pointer casts (`std::slice::from_raw_parts`
with `*const u8`).  This is sound for all types used (`f32`, `f64`, `i32`,
`u32`, `u64`, `bool`) because:
- All have no padding.
- All bit patterns are valid (including NaN for floats in Rust).
- The `Copy` bound is checked by the compiler.

### Partial snapshots and the reader

`NpyIrReader` degrades gracefully: if `x_present: false`, it returns an empty
CSR (all-zero indptr, no data).  If `obs_index_present: false`, it synthesises
integer indices (`"0"`, `"1"`, …).  This means a snapshot written with
`--only X,obs_index` can still be fed into `scx convert` and produce a valid
H5AD.

---

## Planned: mmap path in picklerick (Phase C)

The current picklerick conversion path writes a temp H5AD then calls
`anndataR::read_h5ad()`.  The NPY snapshot enables a direct path with one copy
instead of three.

### Copy count comparison

| path | copies of X data |
|------|-----------------|
| current (H5Seurat → Rust Vec → temp H5AD → anndataR → R heap) | 3 |
| NPY snapshot + mmap → R heap | 1 |
| NPY snapshot + mmap + ALTREP (future) | 0 |

### Proposed implementation

**Step 1 — Add `memmap2` to picklerick Rust deps**

```toml
# r/picklerick/src/rust/Cargo.toml
memmap2 = "0.9"
```

**Step 2 — New `NpyMmapReader` in scx-core (or inline in picklerick)**

```rust
pub struct NpyMmapReader {
    dir: PathBuf,
    meta: Meta,
    // Mmap handles kept alive for the duration of the reader
    x_data_mmap:    Mmap,
    x_indices_mmap: Mmap,
    x_indptr_mmap:  Mmap,
}

impl NpyMmapReader {
    pub fn open(dir: &Path) -> Result<Self> { ... }

    // Returns typed slice into the mmap'd region (zero copy from OS cache)
    pub fn x_data_f32(&self) -> &[f32] {
        // offset 128 = fixed NPY header size for all SCX-produced files
        let bytes = &self.x_data_mmap[128..];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.meta.nnz()) }
    }
    pub fn x_indices(&self) -> &[u32] { ... }
    pub fn x_indptr(&self)  -> &[u64] { ... }
}
```

**Step 3 — New extendr FFI function in picklerick**

```rust
#[extendr]
fn scx_read_npy_snapshot(dir: &str, target: &str) -> List {
    let r = NpyMmapReader::open(Path::new(dir))?;
    let (n_obs, n_vars) = (r.meta.n_obs, r.meta.n_vars);

    // CSR → CSC transpose (required for dgCMatrix; avoidable for dgRMatrix)
    let (p, i, x) = csr_to_csc(
        r.x_data_f32(), r.x_indices(), r.x_indptr(), n_vars
    );

    // Small metadata — load eagerly
    let obs_index = r.read_obs_index()?;
    let var_index = r.read_var_index()?;
    let obs_cols  = r.read_obs_columns()?;
    let obsm      = r.read_obsm()?;

    list!(
        p = p, i = i, x = x,        // CSC components
        n_obs = n_obs as i32,
        n_vars = n_vars as i32,
        obs_index = obs_index,
        var_index = var_index,
        obs = obs_cols,
        obsm = obsm,
    )
}
```

**Step 4 — R-side assembly**

```r
# R/read.R
read_npy_snapshot <- function(dir, output = c("anndata", "sce", "seurat")) {
  output <- match.arg(output)
  raw <- scx_read_npy_snapshot(dir, output)    # Rust FFI call

  # Build dgCMatrix from CSC components
  x <- new("dgCMatrix",
    p    = raw$p,
    i    = raw$i,
    x    = raw$x,
    Dim  = c(raw$n_obs, raw$n_vars),
    Dimnames = list(raw$obs_index, raw$var_index)
  )
  obs <- as.data.frame(raw$obs, row.names = raw$obs_index)

  switch(output,
    anndata = anndataR::InMemoryAnnData$new(
      X = x, obs = obs, var = data.frame(row.names = raw$var_index),
      obsm = raw$obsm
    ),
    sce = {
      sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = t(x)))
      SummarizedExperiment::colData(sce) <- DataFrame(obs)
      SingleCellExperiment::reducedDims(sce) <- raw$obsm
      sce
    },
    seurat = {
      adata <- anndataR::InMemoryAnnData$new(X = x, obs = obs, obsm = raw$obsm,
                                              var = data.frame(row.names = raw$var_index))
      adata$as_Seurat()
    }
  )
}
```

### CSR → CSC and the SCE shortcut

`dgCMatrix` (R's standard sparse class) is CSC.  Converting CSR→CSC is
O(nnz + n_vars) and requires one allocation.  It is unavoidable for
`dgCMatrix` and Seurat.

For `SingleCellExperiment`, R's `Matrix` package also supports `dgRMatrix`
(CSR).  SCE accepts it via `SummarizedExperiment::assay()`.  With `dgRMatrix`,
the CSR arrays from the NPY snapshot can be passed directly without
transposing — the only allocation is copying the mmap'd bytes to R's heap.

```rust
// In the extendr function, dgRMatrix path:
// No transpose needed — return CSR components directly.
list!(
    j = r.x_indices().to_vec(),    // column indices  (CSR "indices")
    p = r.x_indptr_as_i32()?,      // row pointers    (CSR "indptr")
    x = r.x_data_f32().to_vec(),
    n_obs = n_obs as i32,
    n_vars = n_vars as i32,
    ...
)
```

```r
# R side for SCE path:
x_csr <- new("dgRMatrix",
  j    = raw$j,
  p    = raw$p,
  x    = raw$x,
  Dim  = c(raw$n_obs, raw$n_vars),
)
# t(x_csr) gives dgCMatrix if needed, but SCE works with dgRMatrix directly
```

### What is NOT zero-copy (ALTREP)

To avoid the final `to_vec()` copy from mmap → R heap entirely, R's ALTREP
system would let you expose the `Mmap` object as an R numeric/integer vector
without copying.  ALTREP requires a C-level class registration
(`R_make_altreal_class` etc.) and is not supported by extendr.

This is a well-defined future project (a separate `altrep-npy` crate or R
package), but it is out of scope for the current spike.  The single-copy
mmap path is already a substantial improvement over the current three-copy
path.

### BPCells sidecar (Seurat v5)

BPCells' on-disk format (bitpacked integers) cannot be produced by zero-copy
from NPY CSR.  The binding would write a sidecar directory on first access:

```
pbmc3k.scxd/               ← canonical NPY snapshot
pbmc3k.scxd.bpcells/       ← sidecar, generated by R on first bpcells=TRUE call
  X/                        ← BPCells bitpacked format
```

See spec.md §15 for the full sidecar strategy.  This is not part of the
current implementation scope.

---

## Status

| component | status |
|-----------|--------|
| `NpyIrWriter::write` | ✅ implemented (`scx-core/src/npy.rs`) |
| `NpyIrReader::open` + `into_dataset` | ✅ implemented |
| `NpyIrReader` implements `DatasetReader` | ✅ implemented |
| `SlotFilter` with `--only` / `--exclude` | ✅ implemented |
| `scx snapshot` CLI command | ✅ implemented |
| `scx convert <npy-dir> output.h5ad` | ✅ implemented |
| `scx inspect <npy-dir>` | ✅ implemented |
| NPY header written from scratch (no external crate) | ✅ |
| All column kinds (int, float, bool, string, categorical) | ✅ |
| 5 unit tests (roundtrip, streaming, filter modes) | ✅ |
| `NpyMmapReader` with `memmap2` | ⬜ planned (Phase C) |
| `scx_read_npy_snapshot` extendr FFI | ⬜ planned (Phase C) |
| R `read_npy_snapshot()` assembling dgCMatrix/SCE/Seurat | ⬜ planned (Phase C) |
| ALTREP zero-copy path | ⬜ future (separate project) |

---

## Open questions

**Flat vs nested layout.** The flat layout (`X_data.npy`) works well up to
~100 keys per slot.  If a dataset has hundreds of layers or obsp keys, the
directory gets unwieldy.  The nested layout from spec.md (`layers/spliced/data.npy`)
would be cleaner but adds subdirectory creation.  Consider migrating in 0.1.x
if the key count grows.

**Column name sanitisation.** Column names are used verbatim as file stems.
If a column name contains `/`, `\0`, or other filesystem-illegal characters,
the write fails.  A sanitisation pass (replacing illegal chars with `_`, storing
the original name in `meta.json`) would be robust.  Not needed for current
datasets.

**indptr dtype.** `X_indptr.npy` uses `u64` (`<u8`) to support >4B non-zero
entries.  Most datasets have nnz < 2^32.  A `u32` indptr would halve the size
and be directly usable as R integer.  Could add dtype field per-array in
`meta.json` and write `u32` when nnz < 2^32.

**Layers dtype.** Layers inherit `x_dtype` from `meta.json`.  Separate dtypes
per layer would be more faithful but complicate `meta.json`.

**Python consumer.** A thin Python reader (20 lines using `numpy.load` /
`np.memmap` + `json.load`) would complete the cross-language story.  Could
live in `scripts/` or a future `picklerick-py`.
