// vsearch: offline command-line tool.
//
//   vsearch build --data base.fvecs --out index.vsi [--metric l2] [--M 16] [--efc 200]
//                 [--threads N] [--quant sq8|pq16] [--max N]
//   vsearch query --index index.vsi --queries q.fvecs [--gt gt.ivecs] [--k 10] [--ef 64]
//                 [--mmap] [--quantized] [--rerank N] [--show N]
//   vsearch info  --index index.vsi [--check] [--mmap]
#include <cstdio>
#include <iostream>
#include <unordered_set>

#include "args.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/io.hpp"
#include "vecsearch/parallel.hpp"
#include "vecsearch/system.hpp"

namespace {

using namespace vs;

void usage() {
  std::cerr << R"(usage:
  vsearch build --data BASE.fvecs --out INDEX [--metric l2|ip|cosine] [--M 16] [--efc 200]
                [--threads N] [--quant sq8|pqN] [--max N]
  vsearch query --index INDEX --queries Q.fvecs [--gt GT.ivecs] [--k 10] [--ef 64]
                [--mmap] [--quantized] [--rerank N] [--show N]
  vsearch info  --index INDEX [--check] [--mmap]
)";
}

std::string human_bytes(std::size_t b) {
  char buf[32];
  const double v = static_cast<double>(b);
  if (b >= (1ULL << 30)) {
    std::snprintf(buf, sizeof buf, "%.2f GiB", v / static_cast<double>(1ULL << 30));
  } else if (b >= (1ULL << 20)) {
    std::snprintf(buf, sizeof buf, "%.1f MiB", v / static_cast<double>(1ULL << 20));
  } else {
    std::snprintf(buf, sizeof buf, "%.1f KiB", v / 1024.0);
  }
  return buf;
}

void print_memory(const MemoryUsage& m) {
  std::printf("memory     : total %s (vectors %s, graph %s, codes %s, metadata %s, mmap'd %s)\n",
              human_bytes(m.total()).c_str(), human_bytes(m.vectors).c_str(),
              human_bytes(m.graph).c_str(), human_bytes(m.codes).c_str(),
              human_bytes(m.metadata).c_str(), human_bytes(m.mapped).c_str());
}

int cmd_build(const apps::Args& a) {
  const Matrix<float> data = read_fvecs(a.required("data"), a.size("max", 0));
  HnswConfig cfg;
  cfg.dim = data.cols;
  cfg.metric = parse_metric(a.str("metric", "l2"));
  cfg.M = a.size("M", 16);
  cfg.ef_construction = a.size("efc", 200);
  cfg.capacity = data.rows;
  const std::size_t threads = a.size("threads", 0);
  HnswIndex index(cfg);
  std::vector<label_t> labels(data.rows);
  for (std::size_t i = 0; i < data.rows; ++i) labels[i] = i;

  const Stopwatch sw;
  index.add_batch(data.data.span(), labels, threads);
  std::printf("built %zu x %zu in %.2f s (%.0f vectors/s, %zu threads)\n", data.rows, data.cols,
              sw.seconds(), static_cast<double>(data.rows) / sw.seconds(),
              threads == 0 ? hardware_threads() : threads);

  if (a.has("quant")) {
    const std::string q = a.str("quant");
    QuantizerConfig qc;
    if (q == "sq8") {
      qc.type = QuantizerType::SQ8;
    } else if (q.rfind("pq", 0) == 0) {
      qc.type = QuantizerType::PQ;
      qc.pq_subspaces = std::stoul(q.substr(2));
    } else {
      throw Error("unknown quantizer '" + q + "'");
    }
    const Stopwatch qs;
    index.train_quantizer(qc, threads);
    std::printf("trained %s quantizer in %.2f s\n", q.c_str(), qs.seconds());
  }
  print_memory(index.memory_usage());
  const Stopwatch ss;
  index.save(a.required("out"));
  std::printf("saved to %s in %.2f s\n", a.required("out").c_str(), ss.seconds());
  return 0;
}

int cmd_query(const apps::Args& a) {
  const Stopwatch ls;
  const HnswIndex index =
      HnswIndex::load(a.required("index"), a.has("mmap") ? LoadMode::Mmap : LoadMode::Copy);
  std::printf("loaded %zu vectors in %.3f s%s\n", index.size(), ls.seconds(),
              a.has("mmap") ? " (mmap)" : "");
  const Matrix<float> queries = read_fvecs(a.required("queries"));
  SearchParams p;
  p.k = a.size("k", 10);
  p.ef = a.size("ef", 64);
  p.use_quantized = a.has("quantized");
  p.rerank = a.size("rerank", kRerankAll);

  std::vector<std::vector<SearchResult>> results(queries.rows);
  const Stopwatch sw;
  for (std::size_t q = 0; q < queries.rows; ++q) results[q] = index.search(queries.row(q), p);
  const double secs = sw.seconds();
  std::printf("%zu queries, k=%zu, ef=%zu: %.0f QPS (single thread), %.1f us/query\n", queries.rows,
              p.k, p.ef, static_cast<double>(queries.rows) / secs,
              secs * 1e6 / static_cast<double>(queries.rows));

  if (a.has("gt")) {
    const auto gt = read_ivecs(a.str("gt"), queries.rows);
    double hits = 0;
    for (std::size_t q = 0; q < queries.rows; ++q) {
      std::unordered_set<label_t> truth;
      for (std::size_t i = 0; i < p.k && i < gt.cols; ++i)
        truth.insert(static_cast<label_t>(gt.row(q)[i]));
      for (const auto& r : results[q]) hits += truth.contains(r.label) ? 1 : 0;
    }
    std::printf("recall@%zu = %.4f\n", p.k, hits / static_cast<double>(queries.rows * p.k));
  }
  const std::size_t show = a.size("show", 0);
  for (std::size_t q = 0; q < std::min(show, queries.rows); ++q) {
    std::printf("query %zu:", q);
    for (const auto& r : results[q])
      std::printf(" %llu(%.4g)", static_cast<unsigned long long>(r.label),
                  static_cast<double>(r.distance));
    std::printf("\n");
  }
  return 0;
}

int cmd_info(const apps::Args& a) {
  const HnswIndex index =
      HnswIndex::load(a.required("index"), a.has("mmap") ? LoadMode::Mmap : LoadMode::Copy);
  const HnswConfig& c = index.config();
  std::printf("vectors    : %zu live, %zu deleted, capacity %zu\n", index.size(),
              index.deleted_count(), index.capacity());
  std::printf("dimension  : %zu, metric %s\n", index.dim(),
              std::string(to_string(c.metric)).c_str());
  std::printf("graph      : M=%zu (layer0 %zu), efConstruction=%zu\n", c.M, 2 * c.M,
              c.ef_construction);
  const char* q = index.quantizer_type() == QuantizerType::SQ8  ? "sq8"
                  : index.quantizer_type() == QuantizerType::PQ ? "pq"
                                                                : "none";
  std::printf("quantizer  : %s\n", q);
  print_memory(index.memory_usage());
  if (a.has("check")) {
    const IntegrityReport r = index.check_integrity();
    std::printf("integrity  : %s, max level %zu, avg layer-0 degree %.2f, reachable %zu/%zu\n",
                r.ok ? "OK" : "FAILED", r.max_level, r.avg_degree_level0, r.reachable_level0,
                r.nodes);
    for (const auto& e : r.errors) std::printf("  error: %s\n", e.c_str());
    return r.ok ? 0 : 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string cmd = argv[1];
  const apps::Args args(argc, argv, 2, {"mmap", "check", "quantized"});
  if (cmd == "build") return cmd_build(args);
  if (cmd == "query") return cmd_query(args);
  if (cmd == "info") return cmd_info(args);
  usage();
  return 2;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
