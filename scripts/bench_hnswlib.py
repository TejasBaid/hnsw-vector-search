#!/usr/bin/env python3
"""Baseline: run hnswlib with the same protocol as bench/ann_bench.cpp.

hnswlib is used ONLY as a comparison point; the engine does not depend on it.

Protocol: build with the same M / efConstruction / thread count, then for each
efSearch run the whole query set through `knn_query(..., num_threads=1)`
(a single-threaded C++ loop, so Python call overhead is not measured) after an
untimed warm-up pass. Recall@k is computed against the same ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

import hnswlib
import numpy as np


def read_vecs(path: pathlib.Path, dtype, max_rows: int = 0) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    mat = raw.reshape(-1, dim + 1)[:, 1:]
    if max_rows:
        mat = mat[:max_rows]
    return np.ascontiguousarray(mat.view(dtype))


def exact_gt(base: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    if metric == "cosine":
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    bn = (base * base).sum(1)
    for s in range(0, len(queries), 256):
        q = queries[s : s + 256]
        if metric == "l2":
            d = bn[None, :] - 2.0 * q @ base.T
        else:
            d = -(q @ base.T)
        out[s : s + 256] = np.argsort(d, axis=1)[:, :k]
    return out


def footprint_bytes() -> int:
    """macOS: phys_footprint (resident + compressed) via proc_pid_rusage; Linux: RSS."""
    if sys.platform == "darwin":
        import ctypes

        class RusageInfoV0(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
                (f, ctypes.c_uint64) for f in (
                    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
                    "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
                    "proc_exit_abstime")]

        info = RusageInfoV0()
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if libc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) == 0:
            return int(info.phys_footprint)
        return 0
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--query", type=pathlib.Path, required=True)
    ap.add_argument("--gt", type=pathlib.Path)
    ap.add_argument("--metric", default="l2", choices=["l2", "ip", "cosine"])
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", default="10,16,24,32,48,64,96,128,192,256,384,512")
    ap.add_argument("--mt-ef", default="")
    ap.add_argument("--max-base", type=int, default=0)
    ap.add_argument("--max-query", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3, help="timed repetitions per point (best kept)")
    ap.add_argument("--index-path", type=pathlib.Path,
                    help="load the hnswlib index from here if it exists, else build and save it")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    base = read_vecs(args.base, np.float32, args.max_base)
    queries = read_vecs(args.query, np.float32, args.max_query)
    n, dim = base.shape
    if args.gt and not args.max_base:
        gt = read_vecs(args.gt, np.int32, len(queries))[:, : args.k]
    else:
        gt = exact_gt(base, queries, args.k, args.metric)
    print(f"base {n} x {dim}, queries {len(queries)}", file=sys.stderr)

    load_start = os.getloadavg()[0]
    fp0 = footprint_bytes()
    index = hnswlib.Index(space=args.metric, dim=dim)
    build_s = None
    if args.index_path and args.index_path.exists():
        index.load_index(str(args.index_path), max_elements=n)
        print(f"loaded {args.index_path}", file=sys.stderr)
    else:
        index.init_index(max_elements=n, M=args.M, ef_construction=args.efc, random_seed=100)
        index.set_num_threads(args.threads)
        t0 = time.perf_counter()
        index.add_items(base, np.arange(n), num_threads=args.threads)
        build_s = time.perf_counter() - t0
        print(f"build: {build_s:.2f} s", file=sys.stderr)
        if args.index_path:
            index.save_index(str(args.index_path))
    fp1 = footprint_bytes()

    def recall(labels: np.ndarray) -> float:
        hits = sum(len(set(labels[i]) & set(gt[i])) for i in range(len(gt)))
        return hits / (len(gt) * args.k)

    sweep = []
    for ef in [int(x) for x in args.ef.split(",") if x]:
        index.set_ef(max(ef, args.k))
        index.knn_query(queries[:1000], k=args.k, num_threads=1)  # warm-up
        secs = float("inf")
        for _ in range(args.runs):
            t0 = time.perf_counter()
            labels, _ = index.knn_query(queries, k=args.k, num_threads=1)
            secs = min(secs, time.perf_counter() - t0)
        point = {"ef": ef, "recall": recall(labels), "qps": len(queries) / secs,
                 "mean_latency_us": secs * 1e6 / len(queries)}
        print(f"  ef={ef:4d} recall={point['recall']:.4f} qps={point['qps']:9.0f}", file=sys.stderr)
        sweep.append(point)

    mt = []
    for ef in [int(x) for x in args.mt_ef.split(",") if x]:
        index.set_ef(max(ef, args.k))
        index.knn_query(queries, k=args.k, num_threads=args.threads)
        secs = float("inf")
        for _ in range(args.runs):
            t0 = time.perf_counter()
            labels, _ = index.knn_query(queries, k=args.k, num_threads=args.threads)
            secs = min(secs, time.perf_counter() - t0)
        mt.append({"ef": ef, "threads": args.threads, "recall": recall(labels),
                   "qps": len(queries) / secs})
        print(f"  [{args.threads} threads] ef={ef:4d} qps={mt[-1]['qps']:9.0f}", file=sys.stderr)

    out = {
        "name": args.name,
        "runs_per_point": args.runs,
        "load_average_start": load_start,
        "load_average_end": os.getloadavg()[0],
        "library": f"hnswlib {getattr(hnswlib, '__version__', 'unknown')}",
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "params": {"n": n, "dim": dim, "queries": len(queries), "metric": args.metric,
                   "M": args.M, "ef_construction": args.efc, "k": args.k,
                   "build_threads": args.threads},
        "build": ({"seconds": build_s, "vectors_per_second": n / build_s,
                   "footprint_delta_bytes": max(0, fp1 - fp0)} if build_s else None),
        "sweep": sweep,
        "multithread": mt,
    }
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
