#!/usr/bin/env bash
# h5ad -> BPCells matrix dir: scx vs BPCells (write_matrix_dir), time + peak RSS.
# Oracle: both outputs must be byte-identical, file by file.
#
#   scripts/bench_bpcells_dir.sh tests/golden/hlca_core.h5ad [reps]
#
# Needs: target/release/scx (cargo build --release -p scx-cli), pixi env `bpcells`.
set -euo pipefail
in=$1; reps=${2:-3}
out=${BENCH_OUT:-$(mktemp -d)}; mkdir -p "$out"
scx=${SCX:-target/release/scx}
TIME=/usr/bin/time

run() { # label cmd...  -> "label wall_s peak_mb"
  local label=$1; shift
  rm -rf "$out/$label.bpcells"
  $TIME -f "%e %M" -o "$out/time" "$@" >/dev/null 2>"$out/$label.log" || { cat "$out/$label.log"; exit 1; }
  read -r wall kb <"$out/time"
  printf '%-8s %8.2f s %8.0f MB\n' "$label" "$wall" "$((kb / 1024))"
}

cat >"$out/bpcells.R" <<'EOF'
a <- commandArgs(TRUE)
suppressPackageStartupMessages(library(BPCells))
invisible(write_matrix_dir(open_matrix_anndata_hdf5(a[1]), a[2]))
EOF
# R startup + library load alone, to subtract mentally from the BPCells rows.
echo 'suppressPackageStartupMessages(library(BPCells))' >"$out/rstart.R"

echo "input: $in   reps: $reps   out: $out"
rpath=$(pixi run -e bpcells which Rscript)
run r-start "$rpath" "$out/rstart.R"
for i in $(seq "$reps"); do
  run scx "$scx" convert "$in" "$out/scx.bpcells" --dtype f32 --exclude obsm,varm,uns,layers,obsp
  RAYON_NUM_THREADS=1 run scx-1t "$scx" convert "$in" "$out/scx-1t.bpcells" --dtype f32 --exclude obsm,varm,uns,layers,obsp
  run bpcells "$rpath" "$out/bpcells.R" "$in" "$out/bpcells.bpcells"
done

echo "oracle: byte-compare scx vs BPCells output"
for f in "$out/bpcells.bpcells"/*; do
  cmp "$f" "$out/scx.bpcells/$(basename "$f")" || { echo "MISMATCH: $(basename "$f")"; exit 1; }
done
du -sh "$out/scx.bpcells" "$out/bpcells.bpcells"
echo "identical"
