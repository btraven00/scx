# CELLxGENE corpus: size and plan

Format-conversion tools are usually validated on a handful of datasets,
checking shapes. This corpus is meant to go wide and deep: every published
CELLxGENE dataset under a size cap, each output read back by its format's
own reader and compared value by value.

Usage is in [`../README.md`](../README.md#4-corpus-checks-benchcorpus).

H5AD datasets in the 2026-11-07 catalogue snapshot (`sha 1333f4f3…`), by
publication year; none predate 2020.

| year | datasets | ≤ 500 MB | ≤ 2 GB | ≤ 5 GB | total size | cells |
|---|---|---|---|---|---|---|
| 2020 | 22 | 16 | 20 | 21 | 17 GB | 1.7 M |
| 2021 | 200 | 145 | 171 | 185 | 304 GB | 21 M |
| 2022 | 465 | 337 | 421 | 450 | 362 GB | 31 M |
| 2023 | 459 | 267 | 369 | 431 | 774 GB | 48 M |
| 2024 | 554 | 354 | 481 | 522 | 1.0 TB | 86 M |
| 2025 | 321 | 186 | 264 | 290 | 487 GB | 60 M |
| 2026 | 248 | 151 | 203 | 227 | 458 GB | 48 M |
| **all** | **2,269** | **1,456** (211 GB) | **1,929** (686 GB) | **2,126** (1.3 TB) | **3.4 TB** | **295 M** |

The largest file is 53 GB. Checking costs about 0.5 s per MB of input
sequentially (download, four arms, native readers, `scx diff`; measured on
18 datasets, so a rough guide).

| tier | cap | datasets | input | est. sequential | use |
|---|---|---|---|---|---|
| seed | 500 MB, one year, `--n 50` | 50 | ~25 GB | ~1 h | iterate on fixes; a new seed per run |
| full small | 500 MB, 2020–2026 | 1,456 | 211 GB | ~30 h | every schema era |
| paper | 2 GB, 2020–2026 | 1,929 | 686 GB | ~100 h | headline corpus (85 % of datasets) |
| paper, large | 2–5 GB | 197 | 620 GB | ~90 h+ | scale bugs; download, check, delete |

Two random 50-dataset seeds of the same year share about 50²/pool datasets
(≈13 for 2025's 186 under 500 MB); that is chance, not a sampling bug.

Before the paper tiers, `run_corpus.py` needs:

- `--jobs N`: datasets in parallel, grouped by size (R readers can use 3–5×
  the file size in memory).
- delete each dataset and its outputs after checking: the 5 GB tier's input
  alone exceeds the free disk on the bench host, and h5ad output is written
  uncompressed (a 444 MB input gave 1.27 GB).
- a per-step timeout scaled to file size, so slow native reads of multi-GB
  files are not counted as scx failures.
