// End-to-end ANN benchmark driver.
//
// Builds an HNSW index over a base set, then sweeps efSearch and reports
// recall@k against exact ground truth together with single-thread QPS,
// latency percentiles, multi-thread QPS, build time and memory. Optionally
// evaluates SQ8/PQ quantization, filtered search and persistence. Emits JSON
// consumed by scripts/plot_results.py.
//
// Protocol (mirrors ann-benchmarks): queries are issued one at a time from a
// single thread for the QPS/latency numbers; the whole query set is searched
// once untimed for warm-up first.
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/io.hpp"
#include "vecsearch/parallel.hpp"
#include "vecsearch/system.hpp"

namespace {

using json = nlohmann::json;
using namespace vs;

struct Options {
  std::string name = "dataset";
  std::filesystem::path base, query, gt, out, index_path, load_path;
  Metric metric = Metric::L2;
  std::size_t M = 16;
  std::size_t efc = 200;
  std::size_t threads = 0;
  std::size_t k = 10;
  std::size_t max_base = 0;
  std::size_t max_query = 0;
  std::vector<std::size_t> efs{10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
  std::vector<std::size_t> mt_efs{};
  std::vector<std::size_t> quant_efs{};
  std::vector<std::size_t> filter_efs{};
  std::string simd = "auto";
  std::vector<std::string> quant;          // e.g. sq8, pq16, pq32
  std::vector<double> filter_selectivity;  // e.g. 0.5 0.1 0.01
  std::size_t runs = 3;                    // timed repetitions per point; the fastest is kept
  std::size_t filter_queries = 1000;
  bool persistence = false;
  bool check = false;
};

std::size_t g_runs = 3;

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= s.size()) {
    const std::size_t comma = s.find(',', start);
    const std::size_t end = comma == std::string::npos ? s.size() : comma;
    if (end > start) out.push_back(s.substr(start, end - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

std::size_t to_size(const std::string& s) {
  std::size_t v = 0;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc{} || p != s.data() + s.size()) throw Error("not a number: " + s);
  return v;
}

/// Comma-separated sizes; "none" yields an empty list.
std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> out;
  if (s == "none") return out;
  for (const auto& part : split(s)) out.push_back(to_size(part));
  return out;
}

[[noreturn]] void usage() {
  std::cerr << R"(usage: ann_bench --base B.fvecs --query Q.fvecs [--gt G.ivecs] [options]
  --name NAME            label for the output
  --metric l2|ip|cosine  distance (default l2)
  --M N --efc N          graph parameters (default 16 / 200)
  --threads N            build / multi-thread query threads (default: all cores)
  --k N                  neighbours per query (default 10)
  --ef a,b,c|none        efSearch sweep
  --quant-ef / --filter-ef   sweeps for those sections (default: --ef)
  --load PATH            load a saved index instead of building
  --mt-ef a,b            efSearch values for the multi-thread QPS pass
  --simd auto|scalar     distance kernels
  --quant sq8,pq16       quantizers to evaluate (re-ranked and not)
  --filter s1,s2         filtered-search selectivities to evaluate
  --filter-queries N     queries used for filtered runs (default 1000)
  --persistence          measure save / load (copy and mmap) times
  --index PATH           where to save the index (with --persistence; kept)
  --check                run the graph integrity check after build
  --runs N               timed repetitions per point, best kept (default 3)
  --max-base N --max-query N   subsample for quick runs
  --out results.json
)";
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--name")
      o.name = next();
    else if (a == "--base")
      o.base = next();
    else if (a == "--query")
      o.query = next();
    else if (a == "--gt")
      o.gt = next();
    else if (a == "--out")
      o.out = next();
    else if (a == "--index")
      o.index_path = next();
    else if (a == "--load")
      o.load_path = next();
    else if (a == "--metric")
      o.metric = parse_metric(next());
    else if (a == "--M")
      o.M = to_size(next());
    else if (a == "--efc")
      o.efc = to_size(next());
    else if (a == "--threads")
      o.threads = to_size(next());
    else if (a == "--k")
      o.k = to_size(next());
    else if (a == "--max-base")
      o.max_base = to_size(next());
    else if (a == "--max-query")
      o.max_query = to_size(next());
    else if (a == "--runs")
      o.runs = std::max<std::size_t>(1, to_size(next()));
    else if (a == "--filter-queries")
      o.filter_queries = to_size(next());
    else if (a == "--simd")
      o.simd = next();
    else if (a == "--persistence")
      o.persistence = true;
    else if (a == "--check")
      o.check = true;
    else if (a == "--ef") {
      o.efs = parse_list(next());
    } else if (a == "--mt-ef") {
      o.mt_efs = parse_list(next());
    } else if (a == "--quant-ef") {
      o.quant_efs = parse_list(next());
    } else if (a == "--filter-ef") {
      o.filter_efs = parse_list(next());
    } else if (a == "--quant") {
      o.quant = split(next());
    } else if (a == "--filter") {
      for (const auto& s : split(next())) o.filter_selectivity.push_back(std::stod(s));
    } else {
      usage();
    }
  }
  if (o.base.empty() || o.query.empty()) usage();
  if (o.threads == 0) o.threads = hardware_threads();
  return o;
}

using Truth = std::vector<std::vector<label_t>>;

Truth truth_from_ivecs(const Matrix<std::int32_t>& gt, std::size_t nq, std::size_t k) {
  if (gt.cols < k) throw Error("ground truth has fewer than k columns");
  Truth t(nq);
  for (std::size_t q = 0; q < nq; ++q) {
    for (std::size_t i = 0; i < k; ++i) t[q].push_back(static_cast<label_t>(gt.row(q)[i]));
  }
  return t;
}

Truth truth_from_results(const std::vector<std::vector<SearchResult>>& res) {
  Truth t(res.size());
  for (std::size_t q = 0; q < res.size(); ++q) {
    for (const auto& r : res[q]) t[q].push_back(r.label);
  }
  return t;
}

double recall(const std::vector<std::vector<SearchResult>>& res, const Truth& truth,
              std::size_t k) {
  double hits = 0.0;
  double total = 0.0;
  for (std::size_t q = 0; q < truth.size(); ++q) {
    const std::size_t kk = std::min(k, truth[q].size());
    std::unordered_set<label_t> t(truth[q].begin(),
                                  truth[q].begin() + static_cast<std::ptrdiff_t>(kk));
    for (std::size_t i = 0; i < std::min(k, res[q].size()); ++i)
      hits += t.contains(res[q][i].label);
    total += static_cast<double>(kk);
  }
  return total == 0.0 ? 1.0 : hits / total;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  const auto idx = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx), v.end());
  return v[idx];
}

/// Single-threaded sweep point: one query at a time, per-query latency.
json run_point(const HnswIndex& index, const Matrix<float>& queries, const Truth& truth,
               SearchParams params) {
  const std::size_t nq = queries.rows;
  std::vector<std::vector<SearchResult>> res(nq);
  std::vector<double> lat(nq);
  std::size_t dist_total = 0;
  SearchStats stats;
  params.stats = &stats;
  // Warm-up pass (untimed) to fault in pages and train branch predictors.
  for (std::size_t q = 0; q < std::min<std::size_t>(nq, 1000); ++q)
    (void)index.search(queries.row(q), params);
  // Best of g_runs timed passes (as in ann-benchmarks): on a shared machine
  // the fastest pass is the least perturbed estimate of the true cost.
  double secs = 1e300;
  std::vector<double> run_lat(nq);
  for (std::size_t run = 0; run < g_runs; ++run) {
    dist_total = 0;
    const Stopwatch total;
    for (std::size_t q = 0; q < nq; ++q) {
      const Stopwatch sw;
      res[q] = index.search(queries.row(q), params);
      run_lat[q] = sw.micros();
      dist_total += stats.distance_computations;
    }
    const double t = total.seconds();
    if (t < secs) {
      secs = t;
      lat = run_lat;
    }
  }
  json j;
  j["ef"] = params.ef;
  j["recall"] = recall(res, truth, params.k);
  j["qps"] = static_cast<double>(nq) / secs;
  j["mean_latency_us"] = secs * 1e6 / static_cast<double>(nq);
  j["p50_latency_us"] = percentile(lat, 0.50);
  j["p99_latency_us"] = percentile(lat, 0.99);
  j["mean_distance_computations"] = static_cast<double>(dist_total) / static_cast<double>(nq);
  return j;
}

json run_mt_point(const HnswIndex& index, const Matrix<float>& queries, const Truth& truth,
                  const SearchParams& params, std::size_t threads) {
  (void)index.search_batch(queries.data.span(), params, threads);  // warm-up
  double secs = 1e300;
  std::vector<std::vector<SearchResult>> res;
  for (std::size_t run = 0; run < g_runs; ++run) {
    const Stopwatch sw;
    res = index.search_batch(queries.data.span(), params, threads);
    secs = std::min(secs, sw.seconds());
  }
  return {{"ef", params.ef},
          {"threads", threads},
          {"recall", recall(res, truth, params.k)},
          {"qps", static_cast<double>(queries.rows) / secs}};
}

json memory_json(const MemoryUsage& m) {
  return {{"vectors_bytes", m.vectors},   {"graph_bytes", m.graph},   {"codes_bytes", m.codes},
          {"metadata_bytes", m.metadata}, {"total_bytes", m.total()}, {"mapped_bytes", m.mapped}};
}

double load_average() {
  double avg[1] = {0.0};
  return getloadavg(avg, 1) == 1 ? avg[0] : -1.0;
}

void log(const std::string& s) { std::cerr << s << std::endl; }

}  // namespace

int main(int argc, char** argv) try {
  const Options o = parse(argc, argv);
  g_runs = o.runs;
  json out;
  out["name"] = o.name;
  out["runs_per_point"] = o.runs;
  out["load_average_start"] = load_average();
  out["machine"] = {{"hardware_threads", hardware_threads()},
                    {"simd", std::string(to_string(parse_backend(o.simd)))}};

  log("loading " + o.base.string());
  const Matrix<float> base = read_fvecs(o.base, o.max_base);
  const Matrix<float> queries = read_fvecs(o.query, o.max_query);
  if (base.cols != queries.cols) throw Error("base/query dimension mismatch");
  const std::size_t n = base.rows;
  const std::size_t dim = base.cols;
  log("base " + std::to_string(n) + " x " + std::to_string(dim) + ", queries " +
      std::to_string(queries.rows));

  Truth truth;
  if (!o.gt.empty() && o.max_base == 0) {
    truth = truth_from_ivecs(read_ivecs(o.gt, queries.rows), queries.rows, o.k);
  } else {
    log("computing exact ground truth with FlatIndex");
    FlatIndex flat(dim, o.metric);
    std::vector<label_t> labels(n);
    for (std::size_t i = 0; i < n; ++i) labels[i] = i;
    flat.add(base.data.span(), labels);
    truth = truth_from_results(flat.search_batch(queries.data.span(), o.k, o.threads));
  }

  out["params"] = {{"n", n},
                   {"dim", dim},
                   {"queries", queries.rows},
                   {"metric", std::string(to_string(o.metric))},
                   {"M", o.M},
                   {"ef_construction", o.efc},
                   {"k", o.k},
                   {"build_threads", o.threads}};

  // ---- tags for filtered search ------------------------------------------------
  // Bit i is set independently with probability filter_selectivity[i]. Tags
  // do not influence graph construction, so one index serves every
  // selectivity (and the unfiltered sweep). Deterministic, so a --load run
  // regenerates the same tags for its ground truth.
  std::vector<std::uint64_t> tags;
  if (!o.filter_selectivity.empty()) {
    if (o.filter_selectivity.size() > 63) throw Error("at most 63 filter selectivities");
    tags.assign(n, 0);
    std::mt19937_64 rng(12345);
    for (std::size_t b = 0; b < o.filter_selectivity.size(); ++b) {
      std::bernoulli_distribution coin(o.filter_selectivity[b]);
      for (auto& t : tags) t |= coin(rng) ? (std::uint64_t{1} << b) : 0U;
    }
  }

  // ---- build (or load) ---------------------------------------------------------
  std::vector<label_t> labels(n);
  for (std::size_t i = 0; i < n; ++i) labels[i] = i;
  std::unique_ptr<HnswIndex> index_ptr;
  if (!o.load_path.empty()) {
    const Stopwatch sw;
    index_ptr = std::make_unique<HnswIndex>(
        HnswIndex::load(o.load_path, LoadMode::Copy, parse_backend(o.simd)));
    log("loaded " + o.load_path.string() + " in " + std::to_string(sw.seconds()) + " s");
    if (index_ptr->size() != n || index_ptr->dim() != dim)
      throw Error("loaded index does not match base set");
    out["loaded_from"] = o.load_path.string();
    out["params"]["M"] = index_ptr->config().M;
    out["params"]["ef_construction"] = index_ptr->config().ef_construction;
    out["build"] = {{"memory", memory_json(index_ptr->memory_usage())}};
  } else {
    const std::size_t rss_before = current_rss_bytes();
    const std::size_t fp_before = memory_footprint_bytes();
    index_ptr = std::make_unique<HnswIndex>(HnswConfig{.dim = dim,
                                                       .metric = o.metric,
                                                       .M = o.M,
                                                       .ef_construction = o.efc,
                                                       .capacity = n,
                                                       .simd = parse_backend(o.simd)});
    log("building (M=" + std::to_string(o.M) + ", efC=" + std::to_string(o.efc) +
        ", threads=" + std::to_string(o.threads) + ")");
    const Stopwatch build_sw;
    index_ptr->add_batch(base.data.span(), labels, o.threads, tags);
    const double build_s = build_sw.seconds();
    const std::size_t rss_after = current_rss_bytes();
    const std::size_t fp_after = memory_footprint_bytes();
    log("build: " + std::to_string(build_s) + " s");
    out["build"] = {{"seconds", build_s},
                    {"vectors_per_second", static_cast<double>(n) / build_s},
                    {"rss_delta_bytes", rss_after > rss_before ? rss_after - rss_before : 0},
                    {"footprint_delta_bytes", fp_after > fp_before ? fp_after - fp_before : 0},
                    {"memory", memory_json(index_ptr->memory_usage())}};
  }
  HnswIndex& index = *index_ptr;
  if (o.check) {
    const auto r = index.check_integrity();
    out["integrity"] = {{"ok", r.ok},
                        {"reachable_level0", r.reachable_level0},
                        {"max_level", r.max_level},
                        {"avg_degree_level0", r.avg_degree_level0}};
    log(std::string("integrity: ") + (r.ok ? "ok" : "FAILED") + ", reachable " +
        std::to_string(r.reachable_level0) + "/" + std::to_string(n));
  }

  // ---- float sweep ------------------------------------------------------------
  json sweep = json::array();
  for (const std::size_t ef : o.efs) {
    json p = run_point(index, queries, truth, {.k = o.k, .ef = ef});
    std::fprintf(stderr, "  ef=%4zu recall=%.4f qps=%9.0f p50=%7.1fus p99=%7.1fus dists=%.0f\n", ef,
                 p["recall"].get<double>(), p["qps"].get<double>(),
                 p["p50_latency_us"].get<double>(), p["p99_latency_us"].get<double>(),
                 p["mean_distance_computations"].get<double>());
    sweep.push_back(p);
  }
  out["sweep"] = sweep;

  json mt = json::array();
  for (const std::size_t ef : o.mt_efs) {
    json p = run_mt_point(index, queries, truth, {.k = o.k, .ef = ef}, o.threads);
    std::fprintf(stderr, "  [%zu threads] ef=%4zu recall=%.4f qps=%9.0f\n", o.threads, ef,
                 p["recall"].get<double>(), p["qps"].get<double>());
    mt.push_back(p);
  }
  out["multithread"] = mt;

  // ---- persistence -----------------------------------------------------------
  if (o.persistence) {
    const auto path = o.index_path.empty()
                          ? std::filesystem::temp_directory_path() / "ann_bench.vsi"
                          : o.index_path;
    Stopwatch sw;
    index.save(path);
    const double save_s = sw.seconds();
    sw.reset();
    const HnswIndex copy = HnswIndex::load(path, LoadMode::Copy);
    const double load_copy_s = sw.seconds();
    sw.reset();
    const HnswIndex mapped = HnswIndex::load(path, LoadMode::Mmap);
    const double load_mmap_s = sw.seconds();
    bool identical = true;
    for (std::size_t q = 0; q < std::min<std::size_t>(queries.rows, 200); ++q) {
      const SearchParams p{.k = o.k, .ef = 64};
      identical = identical &&
                  index.search(queries.row(q), p) == mapped.search(queries.row(q), p) &&
                  index.search(queries.row(q), p) == copy.search(queries.row(q), p);
    }
    json pj = {{"file_bytes", std::filesystem::file_size(path)},
               {"save_seconds", save_s},
               {"load_copy_seconds", load_copy_s},
               {"load_mmap_seconds", load_mmap_s},
               {"results_identical", identical}};
    pj["mmap_sweep_point"] = run_point(mapped, queries, truth, {.k = o.k, .ef = 64});
    out["persistence"] = pj;
    std::fprintf(stderr, "  save %.2fs, load copy %.2fs, load mmap %.3fs, identical=%d\n", save_s,
                 load_copy_s, load_mmap_s, identical ? 1 : 0);
    if (o.index_path.empty()) std::filesystem::remove(path);
  }

  // ---- filtered search ---------------------------------------------------------
  if (!o.filter_selectivity.empty()) {
    json fj = json::array();
    // Filtered experiments use the first --filter-queries queries (graph
    // traversal under a very selective filter is deliberately expensive).
    const std::size_t nfq = std::min(o.filter_queries, queries.rows);
    Matrix<float> fqueries(nfq, dim);
    std::copy_n(queries.data.data(), nfq * dim, fqueries.data.data());
    FlatIndex flat(dim, o.metric);
    flat.add(base.data.span(), labels, tags);
    for (std::size_t b = 0; b < o.filter_selectivity.size(); ++b) {
      const double s = o.filter_selectivity[b];
      const TagFilter filter{.any_of = std::uint64_t{1} << b};
      const Truth ftruth =
          truth_from_results(flat.search_batch(fqueries.data.span(), o.k, o.threads, filter));
      json graph_points = json::array();
      json planner_points = json::array();
      for (const std::size_t ef : o.filter_efs.empty() ? o.efs : o.filter_efs) {
        // Pure filtered graph traversal, then with the exact-scan planner on.
        json g = run_point(index, fqueries, ftruth,
                           {.k = o.k, .ef = ef, .filter = filter, .allow_exact_scan = false});
        SearchStats st;
        (void)index.search(fqueries.row(0), {.k = o.k, .ef = ef, .filter = filter, .stats = &st});
        json pl = run_point(index, fqueries, ftruth, {.k = o.k, .ef = ef, .filter = filter});
        pl["exact_scan"] = st.exact_scan;
        std::fprintf(stderr,
                     "  [filter s=%.3f] ef=%4zu graph: recall=%.4f qps=%8.0f | planner(%s): "
                     "recall=%.4f qps=%8.0f\n",
                     s, ef, g["recall"].get<double>(), g["qps"].get<double>(),
                     st.exact_scan ? "scan" : "graph", pl["recall"].get<double>(),
                     pl["qps"].get<double>());
        graph_points.push_back(g);
        planner_points.push_back(pl);
      }
      // Brute-force baseline over the filtered subset, single-threaded, on
      // the first 500 queries.
      const std::size_t nflat = std::min<std::size_t>(500, fqueries.rows);
      const Stopwatch sw;
      for (std::size_t q = 0; q < nflat; ++q) (void)flat.search(fqueries.row(q), o.k, filter);
      const double flat_qps = static_cast<double>(nflat) / sw.seconds();
      fj.push_back(
          {{"selectivity", s},
           {"measured_fraction", static_cast<double>(std::count_if(
                                     tags.begin(), tags.end(),
                                     [&](std::uint64_t t) { return (t & filter.any_of) != 0; })) /
                                     static_cast<double>(n)},
           {"queries", fqueries.rows},
           {"sweep_graph", graph_points},
           {"sweep_planner", planner_points},
           {"flat_qps", flat_qps}});
      std::fprintf(stderr, "  [filter s=%.3f] brute-force qps=%.0f\n", s, flat_qps);
    }
    out["filtered"] = fj;
  }

  // ---- quantization ----------------------------------------------------------
  if (!o.quant.empty()) {
    json qj = json::array();
    for (const std::string& qname : o.quant) {
      QuantizerConfig qc;
      if (qname == "sq8") {
        qc.type = QuantizerType::SQ8;
      } else if (qname.rfind("pq", 0) == 0) {
        qc.type = QuantizerType::PQ;
        qc.pq_subspaces = to_size(qname.substr(2));
      } else {
        throw Error("unknown quantizer " + qname);
      }
      const Stopwatch sw;
      index.train_quantizer(qc, o.threads);
      const double train_s = sw.seconds();
      const MemoryUsage mem = index.memory_usage();
      log("quantizer " + qname + " trained+encoded in " + std::to_string(train_s) + " s");
      json no_rr = json::array();
      json rr = json::array();
      for (const std::size_t ef : o.quant_efs.empty() ? o.efs : o.quant_efs) {
        json a = run_point(index, queries, truth,
                           {.k = o.k, .ef = ef, .use_quantized = true, .rerank = 0});
        json b = run_point(index, queries, truth, {.k = o.k, .ef = ef, .use_quantized = true});
        std::fprintf(stderr, "  [%s] ef=%4zu recall=%.4f (rerank %.4f) qps=%9.0f (rerank %9.0f)\n",
                     qname.c_str(), ef, a["recall"].get<double>(), b["recall"].get<double>(),
                     a["qps"].get<double>(), b["qps"].get<double>());
        no_rr.push_back(a);
        rr.push_back(b);
      }
      qj.push_back(
          {{"quantizer", qname},
           {"code_bytes_per_vector", mem.codes / std::max<std::size_t>(index.capacity(), 1)},
           {"train_seconds", train_s},
           {"memory", memory_json(mem)},
           {"sweep_no_rerank", no_rr},
           {"sweep_rerank", rr}});
    }
    out["quantization"] = qj;
  }

  out["peak_rss_bytes"] = peak_rss_bytes();
  out["load_average_end"] = load_average();
  if (!o.out.empty()) {
    std::ofstream f(o.out);
    f << out.dump(2) << '\n';
    log("wrote " + o.out.string());
  } else {
    std::cout << out.dump(2) << '\n';
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
