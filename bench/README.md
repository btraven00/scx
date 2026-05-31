# Benchmarking & performance-regression procedure

Three benchmark layers live here. **Correctness gates performance**: never trust
a speed number until the bit-exact correctness suite is green (see step 0).

## 0. Prerequisites (read first)

- **Build the Python native extension `--release`.** A debug `maturin develop`
  build is ~3× slower on the HDF5 decode path and silently invalidates every
  Python benchmark:

  ```sh
  pixi run -e py313 install-picklerick-py-native-release
  ```

- **Correctness gate.** Before benchmarking a change to the read/decode path,
  run the differential-correctness suite. If any digest differs from the oracle,
  the speed number is meaningless:

  ```sh
  pixi run -e py313 python -m pytest python/picklerick/tests/test_golden_properties.py \
                                      python/picklerick/tests/test_native.py -q
  ```

  The golden digests live in `tests/golden/properties/*.json`; regenerate them
  only when a fixture changes (`pixi run extract-golden-properties`), and review
  the diff — a changed digest means the fixture changed, not the reader.

## 1. Rust — Criterion (`crates/scx-core/benches/`)

```sh
pixi run bench                 # cargo bench, all groups
cargo bench -p scx-core --bench conversion -- stream_chunk_size   # one group
```

- `conversion.rs` — ScxH5 read / roundtrip / npy / metadata throughput.
- `h5_read.rs` — h5ad X decode throughput; guards the parallel-inflate path
  (`h5_chunk.rs`). A regression here means the deflate-only fast path broke or
  silently fell back to the single-threaded HDF5 read.

Criterion stores baselines under `target/criterion/`. To compare a change:
`cargo bench` on the base commit (saves a baseline), then on the change — it
prints the delta. Benches `SKIP` cleanly when their golden fixture is absent.

## 2. Python — stream vs anndata harness (`bench/python/`)

```sh
# full matrix → bench/results/<label>.json (subprocess-isolated peak RSS)
pixi run -e py313 python bench/python/driver.py --label my-change --chunk-sizes 5000
pixi run -e py313 python bench/python/driver.py --label my-change --include-large   # + hlca 5.7 GB

# localise a regression: read vs reduce split (open_stream / anndata backed / h5py-raw)
pixi run -e py313 python bench/python/profile_read_split.py tests/golden/hlca_core.h5ad
```

- `runner.py` runs ONE scenario per subprocess so VmHWM peak RSS is isolated per
  mode; `driver.py` sweeps datasets/chunk sizes, **verifies each streamed
  per-gene sum against the eager `read_h5ad` oracle within rtol**, and writes the
  result envelope. Modes: `load` (eager), `stream` (`open_stream`), `backed`
  (anndata `read_h5ad(backed="r")` + `chunked_X`).
- `profile_read_split.py` splits read-cost from reduce-cost across the three
  readers — use it when a wall number moves and you need to know whether it was
  the decode or the reduction.

## 3. R — read harness (`bench/r/`)

```sh
pixi run -e test Rscript bench/r/read_h5ad_bench.R      # timings + RSS
pixi run -e test Rscript bench/r/profile_read_h5ad.R    # profvis
# heaptrack: bench/r/heaptrack_runner.R
```

## Results convention

`bench/results/*.json` is **gitignored** (only `.gitignore` + the harness are
committed). Each run carries `label / git_sha / branch / timestamp / versions /
results{}`. Use a descriptive `--label` and keep the JSON locally to diff
across commits; don't commit result files.

## When to re-run

- **Any change to the read/decode path** (`h5ad.rs`, `h5_chunk.rs`, the streaming
  binding): correctness suite (step 0) + the Python driver on hlca + `cargo bench
  --bench h5_read`.
- **Fixture changes**: regenerate `tests/golden/properties/` and re-baseline.
- **Before tagging a release**: run all three layers green and record the
  headline numbers in `docs/roadmap.md`.
