#!/usr/bin/env bash
# benchmark_compare.sh
#
# Compares scx vs zellconverter for Seurat-to-AnnData conversion.
# Both start from the same pre-processed data (SCX HDF5 golden file).
# Measures wall time and peak RSS.
#
# Usage:
#   pixi run bash scripts/benchmark_compare.sh
#
# Requirements:
#   - scx release binary (built via cargo build --release)
#   - R with zellconverter, Matrix, hdf5r installed
#   - GNU time (/usr/bin/time -v)

set -euo pipefail

GOLDEN="tests/golden/pbmc3k.h5"
SCX_BIN="target/release/scx"
OUT_DIR="tests/benchmark_results"
GTIME="/usr/bin/time"

mkdir -p "$OUT_DIR"
echo "tool,wall_s,peak_rss_mb" > "$OUT_DIR/results.csv"

if [ ! -f "$GOLDEN" ]; then
  echo "ERROR: $GOLDEN not found — run: pixi run prepare-test-data"
  exit 1
fi
if [ ! -f "$SCX_BIN" ]; then
  echo "Building release binary..."
  cargo build --release --bin scx
fi

echo "============================================================"
echo "SCX benchmark — $(date)"
echo "Dataset: $GOLDEN"
echo "============================================================"
echo ""

# ─── Helper: run with GNU time, extract wall + RSS ────────────────────────────

run_measured() {
  local label="$1"; shift
  local cmd=("$@")
  local tmplog
  tmplog=$(mktemp)

  echo "--- $label ---"
  echo "  cmd: ${cmd[*]}"

  # GNU time writes to stderr; redirect to tmplog
  if "$GTIME" -v "${cmd[@]}" > /dev/null 2>"$tmplog"; then
    true
  else
    echo "  WARNING: command exited non-zero (may still have timing data)"
  fi

  local wall rss_kb rss_mb
  wall=$(grep "Elapsed (wall" "$tmplog" | grep -oP '[\d:]+\.\d+' | tail -1)
  rss_kb=$(grep "Maximum resident" "$tmplog" | grep -oP '\d+' | tail -1)
  rss_mb=$(( ${rss_kb:-0} / 1024 ))

  echo "  Wall time : $wall"
  echo "  Peak RSS  : ${rss_mb} MB"
  echo ""

  echo "$label,$wall,$rss_mb" >> "$OUT_DIR/results.csv"
  rm -f "$tmplog"
}

# ─── scx ──────────────────────────────────────────────────────────────────────

echo "=== scx (Rust, release) ==="

for chunk in 500 2700; do
  out="$OUT_DIR/scx_chunk${chunk}.h5ad"
  run_measured "scx_chunk${chunk}" \
    "$SCX_BIN" convert "$GOLDEN" "$out" --chunk-size "$chunk"
done

# ─── zellconverter ────────────────────────────────────────────────────────────

echo "=== zellconverter (R/Bioconductor) ==="

ZELL_SCRIPT=$(mktemp /tmp/zell_XXXXXX.R)
# Detect pixi-managed Python so reticulate doesn't try to install its own
PIXI_PYTHON=$(pixi run python -c "import sys; print(sys.executable)" 2>/dev/null || true)

cat > "$ZELL_SCRIPT" << 'REOF'
suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(Matrix)
  library(hdf5r)
})

# Load from same SCX HDF5 golden file as scx uses —
# reconstruct dgCMatrix (CSC, genes x cells) then write h5ad via zellconverter.
f <- H5File$new("tests/golden/pbmc3k.h5", mode = "r")
data_vals  <- f[["X/data"]][]
indices    <- f[["X/indices"]][]   # 0-based gene indices
indptr     <- f[["X/indptr"]][]
shape_vec  <- f[["X/shape"]][]
gene_names <- f[["var/index"]][]
cell_names <- f[["obs/index"]][]
f$close_all()

n_genes <- as.integer(shape_vec[1])
n_cells <- as.integer(shape_vec[2])

mat <- sparseMatrix(
  i    = indices + 1L,
  p    = indptr,
  x    = data_vals,
  dims = c(n_genes, n_cells),
  dimnames = list(gene_names, cell_names)
)

sce <- SingleCellExperiment(assays = list(counts = mat))
out <- "tests/benchmark_results/zellkonverter.h5ad"
writeH5AD(sce, file = out, verbose = FALSE)
message(sprintf("zellkonverter written %d cells x %d genes -> %s", n_cells, n_genes, out))
REOF

if pixi run Rscript -e "requireNamespace('zellkonverter', quietly=TRUE)" 2>/dev/null; then
  run_measured "zellkonverter" \
    env RETICULATE_PYTHON="${PIXI_PYTHON}" pixi run Rscript "$ZELL_SCRIPT"
else
  echo "  SKIP: zellkonverter not in pixi env — add bioconductor-zellkonverter to pixi.toml"
  echo ""
fi

rm -f "$ZELL_SCRIPT"

# ─── Summary ──────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Results:"
echo ""
column -t -s',' "$OUT_DIR/results.csv"
echo ""
echo "Full results: $OUT_DIR/results.csv"
echo "============================================================"
