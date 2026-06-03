# Format benchmark design

Goal: characterize the overhead of cross-language single-cell format access under typical access patterns. `.scxd` is the experimental probe (a cheap, uncompressed, mmap-friendly internal format), not a product. The deliverable is a regime map, not a winner.

## The 2×2 to measure

|  | Single-pass (QC scan, log-normalize) | Iterative (PCA matvec, kNN refinement) |
|---|---|---|
| **Cold cache** | compression wins (less disk read) | mixed — first pass cold, then warm |
| **Warm cache** | flat — page cache already paid | zero-copy wins biggest (no per-access decode) |

Only measuring cold+single-pass makes `.scxd` look bad. Only measuring warm+iterative makes it look like a slam dunk. The truth is in the mixture.

## Workloads mapped to each cell

- **Cold + single-pass**: first-ever `colSums`, library size, basic QC. Single read, never revisited.
- **Cold + iterative**: PCA on a fresh load. First epoch cold, subsequent epochs warm. Realistic Scanpy/Seurat startup.
- **Warm + single-pass**: re-running QC after a code change. Common dev-loop case.
- **Warm + iterative**: irlba/PCA inner loop (~50–100 matvecs for k=50), UMAP optimization on kNN graph, repeated kNN queries. The compounding case.

PCA via irlba is the canonical compounding workload — it's where zero-copy + page cache should beat decode-per-access.

## Cost decomposition

Wall-clock alone hides why. Capture:

1. **Disk bytes read** — `/proc/self/io` `read_bytes` or `iostat`. Attributable to format encoding.
2. **Peak RSS** — `/proc/self/status` `VmHWM`. The laziness claim.
3. **Page faults** (major vs minor) — `getrusage`. Separates disk-pulled from cached.
4. **CPU time on decode** — perf stat or sampling profiler attribution to libhdf5 / BPCells decode functions. The cost zero-copy avoids.
5. **Per-epoch wall-clock**, split into load vs compute, with epoch index on x-axis. This is the chart that tells the story: zero-copy is flat; decode-based formats pay every epoch.

## Comparison points

- `h5ad` eager (`sc.read_h5ad` — scanpy default)
- `h5ad` backed (`anndata.read_h5ad(backed='r')`)
- `BPCells` (Seurat v5 path)
- `.scxd` eager (load `.npy` into `Vec` — no-mmap reference)
- `.scxd` mmap (the zero-copy claim)
- *(optional)* TileDB-SOMA — industry-adopted sparse baseline

The `.scxd` eager vs `.scxd` mmap pair is the cleanest comparison: same bytes on disk, only access mechanism differs. Isolates the zero-copy effect from compression / dtype / schema confounds.

## Methodological traps

- **Page cache contamination**: `echo 3 > /proc/sys/vm/drop_caches` between runs, or report cold/warm as separate conditions. Do not average them.
- **Warmup vs steady-state**: report epoch-by-epoch, not just mean. Zero-copy is shape-of-curve.
- **One dataset isn't enough**: PBMC3k (toy), PBMC68k (mid), one atlas-scale (1M+). Curves cross. Atlas-scale is where laziness becomes load-bearing.
- **Dense vs sparse**: `obsm` access (PCA embedding) and sparse `X` access have completely different IO profiles. Report separately.
- **Disk medium**: NVMe vs SATA SSD vs network FS will reorder results. State explicitly.

## Deliverable

A regime map: "for cross-language single-cell workloads, zero-copy formats dominate when [access pattern X, cache state Y, scale Z]; compressed formats dominate elsewhere; here's the crossover." Useful contribution — nobody in the single-cell space has measured this rigorously.
