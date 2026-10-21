#!/usr/bin/env python3
"""Plot recall/QPS curves and print README tables from results/*.json.

    python scripts/plot_results.py --results results --out docs/img

Every chart is drawn from measured JSON; nothing here invents numbers.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Validated categorical palette (fixed order, never cycled) and chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]  # secondary encoding: identity is never colour-only
SEQ_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sequential, light -> dark
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load(results: pathlib.Path, name: str) -> dict | None:
    """Loads results/<name>; if interleaved rounds exist (results/rounds/<stem>_r*.json,
    from scripts/compare_rounds.sh) the sweep is replaced by the per-ef best QPS across
    rounds (recall is deterministic and identical across rounds)."""
    p = results / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rounds = sorted((results / "rounds").glob(p.stem + "_r[0-9]*.json"))
    if rounds:
        best: dict[int, dict] = {}
        for r in rounds:
            for pt in json.loads(r.read_text()).get("sweep", []):
                if pt["ef"] not in best or pt["qps"] > best[pt["ef"]]["qps"]:
                    best[pt["ef"]] = pt
        d["sweep"] = [best[e] for e in sorted(best)]
        d["rounds"] = len(rounds)
    mt_rounds = sorted((results / "rounds").glob(p.stem + "_mt_r*.json"))
    if mt_rounds:
        best_mt: dict[int, dict] = {}
        for r in mt_rounds:
            for pt in json.loads(r.read_text()).get("multithread", []):
                if pt["ef"] not in best_mt or pt["qps"] > best_mt[pt["ef"]]["qps"]:
                    best_mt[pt["ef"]] = pt
        d["multithread"] = [best_mt[e] for e in sorted(best_mt)]
    return d


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def curve(ax, points, i, label, color=None, marker=None, label_at=0.9, below=False):
    pts = sorted(points, key=lambda p: p["recall"])
    xs = [p["recall"] for p in pts]
    ys = [p["qps"] for p in pts]
    c = color or SERIES[i]
    ax.plot(xs, ys, "-", color=c, linewidth=2, marker=marker or MARKERS[i], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    # Direct label at a mid-recall point, where the curves are well separated
    # (they all converge near recall 1.0). Text stays in ink; the mark carries colour.
    j = min(range(len(xs)), key=lambda k: abs(xs[k] - label_at))
    ax.annotate(label, (xs[j], ys[j]), xytext=(4, -9 if below else 7), textcoords="offset points",
                fontsize=8.5, color=INK2, va="top" if below else "bottom", ha="left")


def finish(fig, ax, out: pathlib.Path, stem: str, xmin: float) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    lo, _ = ax.get_xlim()
    ax.set_xlim(max(lo, xmin), 1.0 + (1.0 - xmin) * 0.03)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / stem}.svg/.png")


def recall_qps_chart(results, out, dataset, files, title, xmin, label_at=0.9):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    any_series = False
    for i, (fname, label) in enumerate(files):
        d = load(results, fname)
        if d and d.get("sweep"):
            curve(ax, d["sweep"], i, label, label_at=label_at, below=label.startswith("hnswlib"))
            any_series = True
    if not any_series:
        plt.close(fig)
        return
    style(ax, title, "Recall@10", "Queries / second (1 thread, log scale)")
    finish(fig, ax, out, f"{dataset}_recall_qps", xmin)


def quant_chart(results, out):
    q = load(results, "sift_quant.json")
    base = load(results, "sift_vecsearch.json")
    if not q or not base:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    quant_efs = {p["ef"] for p in q["quantization"][0]["sweep_rerank"]}
    curve(ax, [p for p in base["sweep"] if p["ef"] in quant_efs], 0, "float32 (512 B/vec)",
          label_at=0.8, below=True)
    names = {"sq8": "SQ8 + rerank (128 B/vec)", "pq32": "PQ32 + rerank (32 B/vec)",
             "pq16": "PQ16 + rerank (16 B/vec)"}
    for i, entry in enumerate(q["quantization"], start=1):
        curve(ax, entry["sweep_rerank"], i, names.get(entry["quantizer"], entry["quantizer"]),
              label_at={1: 0.8, 2: 0.73, 3: 0.61}.get(i, 0.8))
    style(ax, "SIFT-1M: traversal on compressed codes, exact re-rank",
          "Recall@10", "Queries / second (1 thread, log scale)")
    finish(fig, ax, out, "sift_quantization", 0.6)


def filter_chart(results, out, ef=64):
    """QPS vs. filter selectivity at a fixed efSearch for the three strategies."""
    d = load(results, "sift_vecsearch.json")
    if not d or not d.get("filtered"):
        return
    rows = sorted(d["filtered"], key=lambda f: f["selectivity"])
    sel = [f["selectivity"] * 100 for f in rows]

    def at_ef(sweep):
        return next(p for p in sweep if p["ef"] == ef)

    graph = [at_ef(f["sweep_graph"])["qps"] for f in rows]
    planner = [at_ef(f["sweep_planner"])["qps"] for f in rows]
    flat = [f["flat_qps"] for f in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    series = [(graph, f"graph traversal only (ef={ef})"), (flat, "brute force (exact)"),
              (planner, f"planner (ef={ef}): graph or exact scan")]
    for i, (ys, label) in enumerate(series):
        ax.plot(sel, ys, "-", color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3 + (i == 2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    style(ax, f"SIFT-1M filtered search: throughput vs. selectivity (ef={ef})",
          "Fraction of vectors matching the filter", "Queries / second (1 thread, log scale)")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.annotate("graph only", (sel[0], graph[0]), xytext=(6, -2), textcoords="offset points",
                fontsize=8.5, color=INK2, va="top")
    ax.annotate("brute force", (sel[-1], flat[-1]), xytext=(-6, -6), textcoords="offset points",
                fontsize=8.5, color=INK2, va="top", ha="right")
    ax.annotate("planner", (sel[1], planner[1]), xytext=(4, 8), textcoords="offset points",
                fontsize=8.5, color=INK2, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out / f"sift_filtered.{ext}", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / 'sift_filtered'}.svg/.png")


# ------------------------------------------------------------------ tables

def fmt_qps(v: float) -> str:
    return f"{v:,.0f}"


def nearest(sweep, target):
    """Cheapest point reaching `target` recall (None if never reached)."""
    ok = [p for p in sweep if p["recall"] >= target]
    return max(ok, key=lambda p: p["qps"]) if ok else None


def tables(results: pathlib.Path) -> None:
    for ds, ours, theirs, scalar in [
        ("SIFT-1M", "sift_vecsearch.json", "sift_hnswlib.json", "sift_vecsearch_scalar.json"),
        ("GloVe-100", "glove_vecsearch.json", "glove_hnswlib.json", None),
    ]:
        a, b = load(results, ours), load(results, theirs)
        c = load(results, scalar) if scalar else None
        if not a:
            continue
        print(f"\n### {ds}: per-ef sweep (rounds: vecsearch {a.get('rounds', 1)}, "
              f"hnswlib {b.get('rounds', 1) if b else 0})\n")
        hdr = "| efSearch | recall@10 | QPS | p50 µs | p99 µs | dist. evals |"
        sep = "|---:|---:|---:|---:|---:|---:|"
        if b:
            hdr += " hnswlib recall | hnswlib QPS | speed-up |"
            sep += "---:|---:|---:|"
        if c:
            hdr += " scalar-kernel QPS |"
            sep += "---:|"
        print(hdr)
        print(sep)
        bmap = {p["ef"]: p for p in b["sweep"]} if b else {}
        cmap = {p["ef"]: p for p in c["sweep"]} if c else {}
        for p in a["sweep"]:
            row = (f"| {p['ef']} | {p['recall']:.4f} | {fmt_qps(p['qps'])} | {p['p50_latency_us']:.0f} | "
                   f"{p['p99_latency_us']:.0f} | {p['mean_distance_computations']:,.0f} |")
            if b and p["ef"] in bmap:
                q = bmap[p["ef"]]
                row += f" {q['recall']:.4f} | {fmt_qps(q['qps'])} | {p['qps'] / q['qps']:.2f}x |"
            if c and p["ef"] in cmap:
                row += f" {fmt_qps(cmap[p['ef']]['qps'])} |"
            print(row)
        print(f"\nQPS at fixed recall ({ds}):\n")
        print("| target recall@10 | vecsearch QPS (ef) | hnswlib QPS (ef) |")
        print("|---:|---:|---:|")
        for t in (0.90, 0.95, 0.99):
            pa = nearest(a["sweep"], t)
            pb = nearest(b["sweep"], t) if b else None
            fa = f"{fmt_qps(pa['qps'])} ({pa['ef']})" if pa else "n/a"
            fb = f"{fmt_qps(pb['qps'])} ({pb['ef']})" if pb else "n/a"
            print(f"| {t:.2f} | {fa} | {fb} |")
        print(f"\nBuild / memory ({ds}):\n")
        m = a["build"]["memory"]
        print(f"- vecsearch build: {a['build']['seconds']:.1f} s with {a['params']['build_threads']} threads "
              f"({a['build']['vectors_per_second']:,.0f} vec/s); footprint delta {a['build'].get('footprint_delta_bytes', 0) / 2**20:,.0f} MiB; "
              f"accounted {m['total_bytes'] / 2**20:,.0f} MiB (vectors {m['vectors_bytes'] / 2**20:,.0f}, "
              f"graph {m['graph_bytes'] / 2**20:,.0f}, metadata {m['metadata_bytes'] / 2**20:,.0f})")
        if b and b.get("build"):
            print(f"- hnswlib build: {b['build']['seconds']:.1f} s with {b['params']['build_threads']} threads "
                  f"({b['build']['vectors_per_second']:,.0f} vec/s); footprint delta "
                  f"{b['build']['footprint_delta_bytes'] / 2**20:,.0f} MiB")
        for name, d in (("vecsearch", a), ("hnswlib", b)):
            if d and d.get("multithread"):
                pts = ", ".join(f"ef={p['ef']}: {fmt_qps(p['qps'])} QPS (recall {p['recall']:.3f})"
                                for p in d["multithread"])
                print(f"- {name} {d['multithread'][0]['threads']}-thread batch: {pts}")
        if "persistence" in a:
            pj = a["persistence"]
            print(f"- persistence: file {pj['file_bytes'] / 2**20:,.0f} MiB, save {pj['save_seconds']:.2f} s, "
                  f"load (copy) {pj['load_copy_seconds']:.2f} s, load (mmap) {pj['load_mmap_seconds'] * 1000:.0f} ms, "
                  f"identical results: {pj['results_identical']}")
        if "integrity" in a:
            it = a["integrity"]
            print(f"- integrity: ok={it['ok']}, reachable {it['reachable_level0']:,}/{a['params']['n']:,}, "
                  f"max level {it['max_level']}, avg layer-0 degree {it['avg_degree_level0']:.1f}")
        for name, d in (("vecsearch", a), ("hnswlib", b)):
            if d:
                print(f"- {name} load average start/end: {d.get('load_average_start', -1):.1f} / "
                      f"{d.get('load_average_end', -1):.1f}")

    q = load(results, "sift_quant.json")
    if q:
        print("\n### SIFT-1M quantization\n")
        print("| codes | bytes/vec | train+encode s | ef | recall (no rerank) | QPS (no rerank) | recall (rerank) | QPS (rerank) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for e in q["quantization"]:
            for a_, b_ in zip(e["sweep_no_rerank"], e["sweep_rerank"]):
                print(f"| {e['quantizer']} | {e['code_bytes_per_vector']} | {e['train_seconds']:.1f} | {a_['ef']} | "
                      f"{a_['recall']:.4f} | {fmt_qps(a_['qps'])} | {b_['recall']:.4f} | {fmt_qps(b_['qps'])} |")
    a = load(results, "sift_vecsearch.json")
    if a and a.get("filtered"):
        print("\n### SIFT-1M filtered search\n")
        print("| selectivity | ef | graph recall | graph QPS | planner | planner recall | planner QPS | brute-force QPS |")
        print("|---:|---:|---:|---:|---|---:|---:|---:|")
        for f in a["filtered"]:
            for g, p in zip(f["sweep_graph"], f["sweep_planner"]):
                print(f"| {f['selectivity']:g} | {g['ef']} | {g['recall']:.4f} | {fmt_qps(g['qps'])} | "
                      f"{'scan' if p.get('exact_scan') else 'graph'} | {p['recall']:.4f} | {fmt_qps(p['qps'])} | "
                      f"{fmt_qps(f['flat_qps'])} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/img"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none"})
    recall_qps_chart(args.results, args.out, "sift",
                     [("sift_vecsearch.json", "vecsearch (NEON)"), ("sift_hnswlib.json", "hnswlib 0.8.0"),
                      ("sift_vecsearch_scalar.json", "vecsearch (scalar kernels)")],
                     "SIFT-1M (128-d, L2): recall vs. throughput", 0.75)
    recall_qps_chart(args.results, args.out, "glove",
                     [("glove_vecsearch.json", "vecsearch (NEON)"), ("glove_hnswlib.json", "hnswlib 0.8.0")],
                     "GloVe-100 (angular): recall vs. throughput", 0.45, label_at=0.7)
    quant_chart(args.results, args.out)
    filter_chart(args.results, args.out)
    tables(args.results)


if __name__ == "__main__":
    main()
