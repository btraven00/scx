# Zero-copy interop format comparison

Requirements: zero-copy across Rust / Python / R, sparse-native, cheap row-subsetting on single-cell matrices (minimum IO).

Note: Parquet ≠ Arrow memory format on disk. Parquet is encoded/compressed and decoded into Arrow record batches on read — not zero-copy. Arrow IPC (`.arrow` / Feather v2) is the zero-copy, mmap-friendly counterpart.

| Format | Zero-copy? | Sparse-native? | R/Py/Rust bindings | Row-subset IO |
|---|---|---|---|---|
| **Arrow IPC** | ✅ mmap of buffers | ✗ column-oriented; CSR awkward to encode | all three mature | good for obs/var; mediocre for CSR |
| **Parquet** | ✗ (decode required) | ✗ | all three mature | predicate/column pushdown, but full decode |
| **Zarr** | ⚠ only if chunks uncompressed; chunk granularity forces over-read | ✗ (sparse via non-standard extensions) | zarr-python ✅, Rarr (R) newish, zarr-rs immature | chunk-sized over-read, not row-precise |
| **TileDB / TileDB-SOMA** | ✗ — always goes through libtiledb storage manager | ✅ first-class | tiledb-py ✅, tiledb-r ✅, rust crate less mature | excellent (multi-dim index) but each read is a copy |
| **.scxd (`.npy` bundle)** | ✅ on all three (R sparse is the exception) | ✅ CSR triplet | trivial — no deps | page-fault granularity, follows actual access pattern |

## Verdict

- No existing standard simultaneously satisfies zero-copy AND sparse-native AND tri-language. That's the gap `.scxd` targets.
- **TileDB-SOMA** is the strongest "don't invent it" argument — CZI has standardized single-cell on it, sparse is first-class. Cost: not zero-copy (libtiledb owns the buffers), heavy C++ dep.
- **Arrow IPC** is the right answer for columnar slots (obs/var/obsm) regardless of the matrix format choice.
- **Zarr** is wrong for local row-subsetting — chunk granularity forces over-read; compressed chunks kill zero-copy.
- **Parquet** is out — decode cost contradicts "minimum IO".
- Composite recommendation: TileDB-SOMA if copy cost is acceptable and ecosystem standardization matters; `.scxd` (raw `.npy` bundle) if zero-copy is non-negotiable; Arrow IPC for the tabular side either way.
