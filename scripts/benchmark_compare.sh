#!/usr/bin/env bash
# benchmark_compare.sh
#
# Compares scx vs anndataR for format conversion.
# Measures wall time (via hyperfine) and peak RSS (via GNU time).
#
# Usage:
#   bash scripts/benchmark_compare.sh           # pbmc3k (2.7k cells, default)
#   bash scripts/benchmark_compare.sh --large   # HLCA core (584k cells)
#
# Requirements:
#   - scx release binary (built via cargo build --release)
#   - hyperfine (https://github.com/sharkdp/hyperfine)
#   - rv (R package manager) with rproject.toml at repo root
#   - GNU time (/usr/bin/time -v)
#   - For --large: run scripts/download_large.sh first

set -euo pipefail

LARGE=0
for arg in "$@"; do
  [[ "$arg" == "--large" ]] && LARGE=1
done

if [[ $LARGE -eq 1 ]]; then
  GOLDEN="tests/golden/hlca_core.h5ad"
  DATASET_LABEL="HLCA core (584k cells)"
  # fewer hyperfine runs — each takes ~minutes
  SCX_RUNS=3; SCX_WARMUP=1
  R_RUNS=3;   R_WARMUP=0
else
  GOLDEN="tests/golden/pbmc3k.h5"
  GOLDEN_SEURAT="tests/golden/pbmc3k.h5seurat"
  DATASET_LABEL="pbmc3k (2.7k cells)"
  SCX_RUNS=10; SCX_WARMUP=3
  R_RUNS=5;    R_WARMUP=1
fi

SCX_BIN="target/release/scx"
OUT_DIR="tests/benchmark_results"
GTIME="/usr/bin/time"

mkdir -p "$OUT_DIR"
echo "tool,mean_s,stddev_s,min_s,max_s,peak_rss_mb" > "$OUT_DIR/results.csv"

if [ ! -f "$GOLDEN" ]; then
  if [[ $LARGE -eq 1 ]]; then
    echo "ERROR: $GOLDEN not found — run: bash scripts/download_large.sh"
  else
    echo "ERROR: $GOLDEN not found — run: pixi run prepare-test-data"
  fi
  exit 1
fi
if [ ! -f "$SCX_BIN" ]; then
  echo "Building release binary..."
  cargo build --release --bin scx
fi

if ! command -v hyperfine &>/dev/null; then
  echo "ERROR: hyperfine not found — install with: cargo install hyperfine"
  exit 1
fi

echo "============================================================"
echo "SCX benchmark — $(date)"
echo "Dataset: $DATASET_LABEL ($GOLDEN)"
echo "============================================================"
echo ""

# ─── Helper: peak RSS via GNU time ────────────────────────────────────────────

peak_rss_mb() {
  local tmplog
  tmplog=$(mktemp)
  "$GTIME" -v "$@" > /dev/null 2>"$tmplog" || true
  local rss_kb
  rss_kb=$(grep "Maximum resident" "$tmplog" | grep -oP '\d+' | tail -1)
  echo $(( ${rss_kb:-0} / 1024 ))
  rm -f "$tmplog"
}

# ─── scx ──────────────────────────────────────────────────────────────────────

echo "=== scx (Rust, release) ==="

# ── scx: SCX H5 → H5AD ────────────────────────────────────────────────────────
label="scx_h5"
out="$OUT_DIR/${label}.h5ad"
hf_json="$OUT_DIR/${label}_hyperfine.json"
echo "--- $label (SCX H5 → H5AD) ---"
hyperfine \
  --warmup "$SCX_WARMUP" \
  --runs   "$SCX_RUNS" \
  --prepare "rm -f $out" \
  --export-json "$hf_json" \
  --command-name "$label" \
  "$SCX_BIN convert $GOLDEN $out"
mean=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['mean']:.4f}\")")
std=$(python3  -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['stddev']:.4f}\")")
tmin=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['min']:.4f}\")")
tmax=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['max']:.4f}\")")
rss=$(peak_rss_mb "$SCX_BIN" convert "$GOLDEN" "$out")
echo "  Peak RSS  : ${rss} MB"
echo ""
echo "$label,$mean,$std,$tmin,$tmax,$rss" >> "$OUT_DIR/results.csv"

# ── scx: H5Seurat → H5AD ──────────────────────────────────────────────────────
if [[ -n "${GOLDEN_SEURAT:-}" && -f "$GOLDEN_SEURAT" ]]; then
  label="scx_h5seurat"
  out="$OUT_DIR/${label}.h5ad"
  hf_json="$OUT_DIR/${label}_hyperfine.json"
  echo "--- $label (H5Seurat → H5AD) ---"
  hyperfine \
    --warmup "$SCX_WARMUP" \
    --runs   "$SCX_RUNS" \
    --prepare "rm -f $out" \
    --export-json "$hf_json" \
    --command-name "$label" \
    "$SCX_BIN convert $GOLDEN_SEURAT $out"
  mean=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['mean']:.4f}\")")
  std=$(python3  -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['stddev']:.4f}\")")
  tmin=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['min']:.4f}\")")
  tmax=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['max']:.4f}\")")
  rss=$(peak_rss_mb "$SCX_BIN" convert "$GOLDEN_SEURAT" "$out")
  echo "  Peak RSS  : ${rss} MB"
  echo ""
  echo "$label,$mean,$std,$tmin,$tmax,$rss" >> "$OUT_DIR/results.csv"
else
  echo "--- scx_h5seurat: SKIPPED (no H5Seurat fixture) ---"
  echo ""
fi

# ─── scx NPY (full round-trip: h5 → npy dir → h5ad) ──────────────────────────
#
# Comparable to scx_chunk*: both measure a complete h5 → h5ad pipeline.
# The NPY path serialises to a staging directory first, so it exercises the
# snapshot writer and the NPY reader back-to-back in a single timed command.

echo "=== scx NPY round-trip ==="

NPY_DIR="$OUT_DIR/scx_npy_rt"
NPY_OUT="$OUT_DIR/scx_npy_rt.h5ad"
label="scx_npy"
hf_json="$OUT_DIR/${label}_hyperfine.json"

echo "--- $label ---"
hyperfine \
  --warmup "$SCX_WARMUP" \
  --runs   "$SCX_RUNS" \
  --prepare "rm -rf $NPY_DIR $NPY_OUT" \
  --export-json "$hf_json" \
  --command-name "$label" \
  "$SCX_BIN snapshot $GOLDEN $NPY_DIR --only X,obs_index,var_index && $SCX_BIN convert $NPY_DIR $NPY_OUT"

mean=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['mean']:.4f}\")")
std=$(python3  -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['stddev']:.4f}\")")
tmin=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['min']:.4f}\")")
tmax=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['max']:.4f}\")")

rss=$(peak_rss_mb bash -c "\"$SCX_BIN\" snapshot \"$GOLDEN\" \"${NPY_DIR}_rss\" --only X,obs_index,var_index && \"$SCX_BIN\" convert \"${NPY_DIR}_rss\" \"${NPY_OUT%.h5ad}_rss.h5ad\"")
echo "  Peak RSS  : ${rss} MB"
echo ""
echo "$label,$mean,$std,$tmin,$tmax,$rss" >> "$OUT_DIR/results.csv"

# ─── anndataR ─────────────────────────────────────────────────────────────────

echo "=== anndataR (R/Bioconductor) ==="

RV_LIB=$(rv library)
ANNDATA_SCRIPT=$(mktemp /tmp/anndata_XXXXXX.R)

cat > "$ANNDATA_SCRIPT" << REOF
suppressPackageStartupMessages({
  library(anndataR)
  library(rhdf5)
  library(Matrix)
})

f     <- "$GOLDEN"
large <- $LARGE

if (large) {
  # Large fixture is already H5AD (AnnData CSR) — read_h5ad then write to a new tempfile
  adata_in <- read_h5ad(f, as = "InMemoryAnnData")
  out <- tempfile(fileext = ".h5ad")
  write_h5ad(adata_in, out, mode = "w")
  message(sprintf("anndataR (large) written -> %s (%.0f MB)", out, file.size(out) / 1e6))
} else {
  # Small fixture is SCX internal HDF5 (CSC, gene-major)
  data_vals <- h5read(f, "X/data")
  indices   <- h5read(f, "X/indices")
  indptr    <- h5read(f, "X/indptr")
  shape_vec <- h5read(f, "X/shape")
  n_genes   <- as.integer(shape_vec[1])
  n_cells   <- as.integer(shape_vec[2])

  mat_csr <- t(sparseMatrix(
    i = indices + 1L, p = indptr, x = as.numeric(data_vals),
    dims = c(n_genes, n_cells)
  ))

  cell_names <- h5read(f, "obs/index")
  gene_names <- h5read(f, "var/index")
  obs_df <- data.frame(
    nCount_RNA   = h5read(f, "obs/nCount_RNA"),
    nFeature_RNA = h5read(f, "obs/nFeature_RNA"),
    orig.ident   = h5read(f, "obs/orig.ident"),
    row.names    = cell_names
  )
  read_obsm <- function(key) {
    m <- h5read(f, paste0("obsm/", key))
    if (ncol(m) == n_cells && nrow(m) != n_cells) m <- t(m)
    m
  }
  obsm_list <- list(X_pca = read_obsm("X_pca"), X_umap = read_obsm("X_umap"))

  adata <- AnnData(X = mat_csr, obs = obs_df, var = data.frame(row.names = gene_names), obsm = obsm_list)
  out <- tempfile(fileext = ".h5ad")
  write_h5ad(adata, out, mode = "w")
  message(sprintf("anndataR written %d x %d -> %s (%.1f MB)", n_cells, n_genes, out, file.size(out) / 1e6))
}
REOF

hf_json="$OUT_DIR/anndataR_hyperfine.json"

echo "--- anndataR ---"
hyperfine \
  --warmup "$R_WARMUP" \
  --runs   "$R_RUNS" \
  --export-json "$hf_json" \
  --command-name "anndataR" \
  "R_LIBS_USER=$RV_LIB Rscript $ANNDATA_SCRIPT"

mean=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['mean']:.4f}\")")
std=$(python3  -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['stddev']:.4f}\")")
tmin=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['min']:.4f}\")")
tmax=$(python3 -c "import json; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['max']:.4f}\")")

rss=$(peak_rss_mb env R_LIBS_USER="$RV_LIB" Rscript "$ANNDATA_SCRIPT")
echo "  Peak RSS  : ${rss} MB"
echo ""

echo "anndataR,$mean,$std,$tmin,$tmax,$rss" >> "$OUT_DIR/results.csv"

rm -f "$ANNDATA_SCRIPT"

# ─── Summary ──────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Results:"
echo ""
column -t -s',' "$OUT_DIR/results.csv"
echo ""
echo "Full results : $OUT_DIR/results.csv"
echo "Hyperfine JSON: $OUT_DIR/*_hyperfine.json"
echo "============================================================"
