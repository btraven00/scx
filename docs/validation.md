# scx validate

## Motivation

Snakemake pipelines produce one h5ad per stage. `scx validate` checks that each
output matches expectations before the next rule runs — catching shape regressions,
missing slots, or wrong dtypes early rather than at conversion time.

Validation is metadata-only by default: shape, slot names, dtypes, and index
properties are all readable without touching matrix data.

---

## CLI

```sh
# Schema file (preferred — reusable across samples)
scx validate results/normalized.h5ad --schema schemas/after_normalization.yaml

# Inline flags (quick one-offs)
scx validate results/normalized.h5ad \
  --has-layer normalized \
  --min-obs 1000 \
  --dtype f32

# Exit non-zero on failure (Snakemake integration)
scx validate results/normalized.h5ad --schema schemas/after_normalization.yaml \
  || { echo "validation failed"; exit 1; }
```

Exits `0` on pass, `1` on validation failure, `2` on file-not-found / parse error.
Prints a structured report to stdout (one line per check, PASS/FAIL prefix).

---

## Schema file format

YAML. All fields are optional — omit what you don't need.

```yaml
# schemas/after_normalization.yaml

# --- Shape constraints ---
obs: ">= 1000"       # also: "== 8000", "> 0", plain integer means exact match
vars: ">= 500"

# --- Required slots ---
layers:
  - normalized
obsm:
  - X_pca
  - X_umap
obsp:
  - connectivities
  - distances
var_columns:
  - highly_variable
  - dispersions
  - dispersions_norm
obs_columns:
  - leiden
  - batch

# --- Type constraints ---
x_dtype: f32         # f32 | f64 | i32 | u32

# --- Index integrity (metadata-only checks) ---
obs_index_unique: true
var_index_unique: true

# --- Data checks (require matrix scan, opt-in) ---
check_finite: false          # scan X for NaN / Inf
check_finite_layers:         # scan specific layers instead of X
  - normalized
check_sorted_indices: false  # verify CSR indices are sorted per row
```

### Qualifier syntax for numeric fields

| Expression | Meaning |
|------------|---------|
| `500` | exactly 500 |
| `"== 500"` | exactly 500 |
| `">= 1000"` | at least 1000 |
| `"> 0"` | strictly positive |
| `"<= 50000"` | at most 50000 |
| `"< 100"` | strictly less than 100 |

Applies to `obs` and `vars`. Plain integers are treated as `==`.

---

## Output format

```
PASS  shape           8312 obs × 33694 vars  (>= 1000 obs, >= 500 vars)
PASS  x_dtype         f32
PASS  layers          normalized
PASS  obsm            X_pca, X_umap
PASS  obsp            connectivities, distances
PASS  var_columns     highly_variable, dispersions, dispersions_norm
FAIL  obs_columns     missing: leiden
PASS  obs_index       unique
PASS  var_index       unique

1 check failed.
```

Machine-readable JSON output with `--json`:

```json
{
  "file": "results/normalized.h5ad",
  "schema": "schemas/after_normalization.yaml",
  "passed": 8,
  "failed": 1,
  "checks": [
    { "name": "obs_columns", "status": "FAIL", "detail": "missing: leiden" }
  ]
}
```

---

## Cost model

All checks below run without reading any matrix data:

| Check | Data read |
|-------|-----------|
| `obs` / `vars` | shape attribute only |
| `layers` / `obsm` / `obsp` / `var_columns` / `obs_columns` | group/dataset names only |
| `x_dtype` | X/data dtype attribute only |
| `obs_index_unique` / `var_index_unique` | index string array |

Opt-in data scans:

| Check | Data read |
|-------|-----------|
| `check_finite` | full X data array |
| `check_finite_layers` | named layer data arrays |
| `check_sorted_indices` | X/indices array |

---

## Snakemake integration

Define one schema per pipeline stage and validate as a rule input check:

```python
rule normalize:
    input:  "data/source.h5ad"
    output: "results/normalized.h5ad"
    shell:  "my_normalize.py {input} {output}"

rule validate_normalized:
    input:  "results/normalized.h5ad"
    output: touch("results/.normalized.validated")
    shell:  "scx validate {input} --schema schemas/after_normalization.yaml"

rule pca:
    input:
        data      = "results/normalized.h5ad",
        validated = "results/.normalized.validated"
    output: "results/pca.h5ad"
    ...
```

---

## Implementation

```
scx-core/src/validate.rs   — ValidationSchema, Qualifier, CheckResult, ValidationReport, run_validation()
scx-cli/src/main.rs        — Validate { input, schema, json } CLI arm + print_report_human / print_report_json
```

`run_validation()` accepts any `&mut dyn DatasetReader` and reads lazily — only
the slots actually referenced by the schema are fetched. Shape and dtype checks
cost nothing beyond `reader.shape()` / `reader.dtype()`, which read only HDF5
attributes. Each slot type (`layer_metas`, `obsm`, `obsp`, `obs`, `var`) is
read at most once per validation call regardless of how many checks reference it.

`serde_yaml` lives in `scx-cli` (runtime) and `scx-core` dev-dependencies (tests
only), keeping the core library free of a YAML parse dependency.
