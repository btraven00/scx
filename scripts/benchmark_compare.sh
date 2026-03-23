#!/usr/bin/env bash
# benchmark_compare.sh
#
# Compares scx vs anndataR for Seurat-to-AnnData conversion.
# Both start from the same pre-processed data (SCX HDF5 golden file).
# Measures wall time (via hyperfine) and peak RSS (via GNU time).
#
# Usage:
#   bash scripts/benchmark_compare.sh
#
# Requirements:
#   - scx release binary (built via cargo build --release)
#   - hyperfine (https://github.com/sharkdp/hyperfine)
#   - rv (R package manager) with rproject.toml at repo root
#   - GNU time (/usr/bin/time -v)

set -euo pipefail

GOLDEN="tests/golden/pbmc3k.h5"
SCX_BIN="target/release/scx"
OUT_DIR="tests/benchmark_results"
GTIME="/usr/bin/time"

mkdir -p "$OUT_DIR"
echo "tool,mean_s,stddev_s,min_s,max_s,peak_rss_mb" > "$OUT_DIR/results.csv"

if [ ! -f "$GOLDEN" ]; then
  echo "ERROR: $GOLDEN not found — run: pixi run prepare-test-data"
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
echo "Dataset: $GOLDEN"
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

for chunk in 500 2700; do
  out="$OUT_DIR/scx_chunk${chunk}.h5ad"
  label="scx_chunk${chunk}"

  echo "--- $label ---"

  # hyperfine: 3 warmup runs, 10 timed runs, export JSON for parsing
  hf_json="$OUT_DIR/${label}_hyperfine.json"
  hyperfine \
    --warmup 3 \
    --runs 10 \
    --prepare "rm -f $out" \
    --export-json "$hf_json" \
    --command-name "$label" \
    "$SCX_BIN convert $GOLDEN $out --chunk-size $chunk"

  # Extract stats from hyperfine JSON (mean, stddev, min, max in seconds)
  mean=$(python3 -c "import json,sys; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['mean']:.4f}\")")
  std=$(python3  -c "import json,sys; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['stddev']:.4f}\")")
  tmin=$(python3 -c "import json,sys; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['min']:.4f}\")")
  tmax=$(python3 -c "import json,sys; d=json.load(open('$hf_json')); print(f\"{d['results'][0]['max']:.4f}\")")

  # Peak RSS from a single /usr/bin/time run
  rss=$(peak_rss_mb "$SCX_BIN" convert "$GOLDEN" "$out" --chunk-size "$chunk")
  echo "  Peak RSS  : ${rss} MB"
  echo ""

  echo "$label,$mean,$std,$tmin,$tmax,$rss" >> "$OUT_DIR/results.csv"
done

# ─── anndataR ─────────────────────────────────────────────────────────────────

echo "=== anndataR (R/Bioconductor) ==="

RV_LIB=$(rv library)
ANNDATA_SCRIPT=$(mktemp /tmp/anndata_XXXXXX.R)

cat > "$ANNDATA_SCRIPT" << 'REOF'
suppressPackageStartupMessages({
  library(anndataR)
  library(rhdf5)
  library(Matrix)
})

f <- "tests/golden/pbmc3k.h5"

# Count matrix
data_vals <- h5read(f, "X/data")
indices   <- h5read(f, "X/indices")   # 0-based gene indices
indptr    <- h5read(f, "X/indptr")
shape_vec <- h5read(f, "X/shape")
n_genes   <- as.integer(shape_vec[1])
n_cells   <- as.integer(shape_vec[2])

# SCX HDF5 stores CSC (gene-major). Reconstruct dgCMatrix then transpose to CSR (cell-major).
mat_csc <- sparseMatrix(
  i    = indices + 1L,
  p    = indptr,
  x    = as.numeric(data_vals),
  dims = c(n_genes, n_cells)
)
mat_csr <- t(mat_csc)  # (n_cells, n_genes)

# obs metadata
cell_names <- h5read(f, "obs/index")
obs_df <- data.frame(
  nCount_RNA   = h5read(f, "obs/nCount_RNA"),
  nFeature_RNA = h5read(f, "obs/nFeature_RNA"),
  orig.ident   = h5read(f, "obs/orig.ident"),
  row.names    = cell_names
)

# var metadata (index only in golden file)
gene_names <- h5read(f, "var/index")
var_df <- data.frame(row.names = gene_names)

# obsm embeddings — stored (n_comps, n_cells); transpose to (n_cells, n_comps)
read_obsm <- function(key) {
  m <- h5read(f, paste0("obsm/", key))
  if (ncol(m) == n_cells && nrow(m) != n_cells) m <- t(m)
  m
}
obsm_list <- list(
  X_pca  = read_obsm("X_pca"),
  X_umap = read_obsm("X_umap")
)

adata <- AnnData(X = mat_csr, obs = obs_df, var = var_df, obsm = obsm_list)

out <- tempfile(fileext = ".h5ad")
write_h5ad(adata, out, mode = "w")
message(sprintf("anndataR written %d cells x %d genes -> %s (%.1f MB)", n_cells, n_genes, out, file.size(out) / 1e6))
REOF

R_LIBS_USER="$RV_LIB" Rscript -e "requireNamespace('anndataR', quietly=TRUE)" 2>/dev/null
hf_json="$OUT_DIR/anndataR_hyperfine.json"

echo "--- anndataR ---"
hyperfine \
  --warmup 1 \
  --runs 5 \
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
