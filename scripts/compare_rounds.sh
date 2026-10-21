#!/usr/bin/env bash
# Interleaved A/B query rounds: vecsearch and hnswlib alternate full efSearch
# sweeps on their saved indexes, ROUNDS times. plot_results.py keeps the best
# QPS per (library, ef) across rounds, so both libraries are measured under
# the same (shared-machine) conditions. Run after run_benchmarks.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build/release}"
DATA="${DATA:-$ROOT/data}"
OUT="${OUT:-$ROOT/results}/rounds"
PY="${PYTHON:-python3}"
ROUNDS="${ROUNDS:-3}"
EFS="10,16,24,32,48,64,96,128,192,256,384,512"
MT_EFS="16,32,64,128,256"
mkdir -p "$OUT"

for ds in sift glove; do
  metric=l2; [ "$ds" = glove ] && metric=cosine
  ARGS=(--base "$DATA/$ds/base.fvecs" --query "$DATA/$ds/query.fvecs" --gt "$DATA/$ds/groundtruth.ivecs")
  [ -f "$DATA/$ds.vsi" ] && [ -f "$DATA/$ds.hnswlib" ] || { echo "run run_benchmarks.sh first" >&2; exit 1; }
  # MT_ONLY=1 skips the single-thread sweeps (note: BSD `seq 1 0` counts down).
  r=0
  while [ "${MT_ONLY:-0}" != 1 ] && [ "$r" -lt "$ROUNDS" ]; do
    r=$((r + 1))
    "$BUILD/ann_bench" --name "$ds" "${ARGS[@]}" --load "$DATA/$ds.vsi" --ef "$EFS" --runs 1 \
        --out "$OUT/${ds}_vecsearch_r$r.json"
    "$PY" "$ROOT/scripts/bench_hnswlib.py" --name "$ds" "${ARGS[@]}" --metric $metric \
        --index-path "$DATA/$ds.hnswlib" --ef "$EFS" --runs 1 --out "$OUT/${ds}_hnswlib_r$r.json"
  done
  # Multi-thread batch throughput, interleaved, best of 5 passes each.
  for r in 1 2; do
    "$BUILD/ann_bench" --name "$ds" "${ARGS[@]}" --load "$DATA/$ds.vsi" --ef none --mt-ef "$MT_EFS" \
        --runs 5 --out "$OUT/${ds}_vecsearch_mt_r$r.json"
    "$PY" "$ROOT/scripts/bench_hnswlib.py" --name "$ds" "${ARGS[@]}" --metric $metric \
        --index-path "$DATA/$ds.hnswlib" --ef "" --mt-ef "$MT_EFS" --runs 5 --out "$OUT/${ds}_hnswlib_mt_r$r.json"
  done
done
