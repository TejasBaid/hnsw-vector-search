#!/usr/bin/env bash
# Reproduces every number in the README.
#
#   scripts/run_benchmarks.sh            # uses build/release, writes results/*.json
#   PYTHON=.venv/bin/python THREADS=10 scripts/run_benchmarks.sh
#
# Each SIFT/GloVe build is done once and saved (data/*.vsi, data/*.hnswlib);
# later stages reload it.
# Prerequisites: cmake --preset release && cmake --build --preset release
#                python scripts/prepare_datasets.py --data-dir data
#                pip install numpy h5py matplotlib hnswlib
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build/release}"
DATA="${DATA:-$ROOT/data}"
OUT="${OUT:-$ROOT/results}"
PY="${PYTHON:-python3}"
THREADS="${THREADS:-$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )}"
RUNS="${RUNS:-3}"
EFS="10,16,24,32,48,64,96,128,192,256,384,512"
mkdir -p "$OUT"

run() { echo "+ $*" >&2; "$@"; }

# ---------------------------------------------------------------- SIFT-1M (L2)
SIFT=(--base "$DATA/sift/base.fvecs" --query "$DATA/sift/query.fvecs" --gt "$DATA/sift/groundtruth.ivecs")
# Build runs measure build time, memory, integrity, persistence and filtered
# search; the recall/QPS sweeps come from the interleaved rounds below.
rm -f "$DATA/sift.hnswlib" "$DATA/glove.hnswlib"   # force fresh hnswlib builds
run "$BUILD/ann_bench" --name sift-1m "${SIFT[@]}" --metric l2 --M 16 --efc 200 \
    --threads "$THREADS" --runs "$RUNS" --ef none --check \
    --persistence --index "$DATA/sift.vsi" \
    --filter 0.5,0.1,0.01,0.001 --filter-ef 16,64,256 --filter-queries 1000 \
    --out "$OUT/sift_vecsearch.json"
run "$PY" "$ROOT/scripts/bench_hnswlib.py" --name sift-1m "${SIFT[@]}" --metric l2 --M 16 --efc 200 \
    --threads "$THREADS" --runs "$RUNS" --ef "" --index-path "$DATA/sift.hnswlib" \
    --out "$OUT/sift_hnswlib.json"
# Same graph, scalar distance kernels: isolates the SIMD contribution.
run "$BUILD/ann_bench" --name sift-1m-scalar "${SIFT[@]}" --load "$DATA/sift.vsi" --simd scalar \
    --threads "$THREADS" --runs "$RUNS" --ef "$EFS" --out "$OUT/sift_vecsearch_scalar.json"
# Quantization on the same graph.
run "$BUILD/ann_bench" --name sift-1m-quant "${SIFT[@]}" --load "$DATA/sift.vsi" --ef none \
    --threads "$THREADS" --runs "$RUNS" --quant sq8,pq32,pq16 --quant-ef 16,32,64,128,256,512 \
    --out "$OUT/sift_quant.json"

# ---------------------------------------------------------- GloVe-100 (angular)
GLOVE=(--base "$DATA/glove/base.fvecs" --query "$DATA/glove/query.fvecs" --gt "$DATA/glove/groundtruth.ivecs")
run "$BUILD/ann_bench" --name glove-100 "${GLOVE[@]}" --metric cosine --M 16 --efc 200 \
    --threads "$THREADS" --runs "$RUNS" --ef none --check \
    --persistence --index "$DATA/glove.vsi" --out "$OUT/glove_vecsearch.json"
run "$PY" "$ROOT/scripts/bench_hnswlib.py" --name glove-100 "${GLOVE[@]}" --metric cosine --M 16 --efc 200 \
    --threads "$THREADS" --runs "$RUNS" --ef "" --index-path "$DATA/glove.hnswlib" \
    --out "$OUT/glove_hnswlib.json"

# ------------------------------------------------------------ kernel microbench
run "$BUILD/distance_bench" --benchmark_out="$OUT/distance_bench.json" --benchmark_out_format=json \
    --benchmark_repetitions=3 --benchmark_report_aggregates_only=true

# Interleaved A/B rounds on the saved indexes (headline comparison).
BUILD="$BUILD" DATA="$DATA" OUT="$OUT" PYTHON="$PY" run "$ROOT/scripts/compare_rounds.sh"

run "$PY" "$ROOT/scripts/plot_results.py" --results "$OUT" --out "$ROOT/docs/img"
