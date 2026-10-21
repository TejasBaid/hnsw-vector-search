#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include "test_util.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"

namespace vs {
namespace {

constexpr std::size_t kDim = 32;
constexpr std::size_t kN = 10'000;
constexpr std::size_t kQueries = 200;

struct Fixture {
  std::vector<float> data;
  std::vector<float> queries;
  std::vector<label_t> labels;
};

// Inner-product search is only well-posed for graph indexes when norms are
// comparable (MIPS on raw data is a different problem), so the IP fixture uses
// unit-norm data.
const Fixture& fixture(Metric metric = Metric::L2) {
  auto make = [](bool unit) {
    Fixture x;
    auto split = test::clustered_split(kN, kQueries, kDim, 42, unit);
    x.data = std::move(split.base);
    x.queries = std::move(split.queries);
    x.labels = test::iota_labels(kN);
    return x;
  };
  static const Fixture plain = make(false);
  static const Fixture unit = make(true);
  return metric == Metric::InnerProduct ? unit : plain;
}

std::vector<std::vector<SearchResult>> exact(Metric metric, std::size_t k) {
  const auto& f = fixture(metric);
  FlatIndex flat(kDim, metric);
  flat.add(f.data, f.labels);
  return flat.search_batch(f.queries, k, 0);
}

HnswIndex build(Metric metric, std::size_t threads = 4) {
  const auto& f = fixture(metric);
  HnswConfig cfg{.dim = kDim, .metric = metric, .M = 16, .ef_construction = 100, .capacity = kN};
  HnswIndex index(cfg);
  index.add_batch(f.data, f.labels, threads);
  return index;
}

class HnswRecall : public ::testing::TestWithParam<Metric> {};

TEST_P(HnswRecall, HighRecallAgainstBruteForce) {
  const Metric metric = GetParam();
  const auto index = build(metric);
  const auto truth = exact(metric, 10);
  SearchParams p{.k = 10, .ef = 128};
  const auto& queries = fixture(metric).queries;
  const auto approx = index.search_batch(queries, p, 0);
  const double recall = test::recall_at_k(approx, truth, 10);
  EXPECT_GE(recall, 0.97) << "metric=" << to_string(metric);

  // Recall must be monotone (within noise) in ef and near-perfect at large ef.
  p.ef = 16;
  const double low = test::recall_at_k(index.search_batch(queries, p, 0), truth, 10);
  p.ef = 512;
  const double high = test::recall_at_k(index.search_batch(queries, p, 0), truth, 10);
  EXPECT_LE(low, recall + 1e-9);
  EXPECT_GE(high, 0.995);
}

INSTANTIATE_TEST_SUITE_P(Metrics, HnswRecall,
                         ::testing::Values(Metric::L2, Metric::InnerProduct, Metric::Cosine),
                         [](const auto& p) { return std::string(to_string(p.param)); });

TEST(Hnsw, GraphInvariantsHold) {
  const auto index = build(Metric::L2, 8);
  const IntegrityReport r = index.check_integrity();
  for (const auto& e : r.errors) ADD_FAILURE() << e;
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.nodes, kN);
  EXPECT_GE(r.reachable_level0, kN * 999 / 1000) << "layer 0 should be (almost) fully connected";
  EXPECT_GT(r.max_level, 0U);
  EXPECT_GT(r.avg_degree_level0, 8.0);
  EXPECT_LE(r.avg_degree_level0, 32.0);
}

TEST(Hnsw, SingleThreadedBuildIsDeterministic) {
  const auto a = build(Metric::L2, 1);
  const auto b = build(Metric::L2, 1);
  const SearchParams p{.k = 10, .ef = 50};
  for (std::size_t q = 0; q < 20; ++q) {
    const std::span<const float> query(fixture().queries.data() + q * kDim, kDim);
    EXPECT_EQ(a.search(query, p), b.search(query, p));
  }
}

TEST(Hnsw, ResultsAreSortedAndExactDistances) {
  const auto index = build(Metric::L2);
  const std::span<const float> q(fixture().queries.data(), kDim);
  const auto res = index.search(q, {.k = 20, .ef = 64});
  ASSERT_EQ(res.size(), 20U);
  for (std::size_t i = 1; i < res.size(); ++i) EXPECT_LE(res[i - 1].distance, res[i].distance);
  for (const auto& r : res) {
    const std::span<const float> v(fixture().data.data() + r.label * kDim, kDim);
    EXPECT_NEAR(r.distance, reference_distance(Metric::L2, q, v), 1e-3);
  }
}

TEST(Hnsw, EmptyAndTinyIndexes) {
  HnswIndex index({.dim = 3, .capacity = 4});
  const std::vector<float> q{1, 2, 3};
  EXPECT_TRUE(index.search(q, {}).empty());
  EXPECT_TRUE(index.check_integrity().ok);
  index.add(7, q);
  const auto r = index.search(q, {.k = 5});
  ASSERT_EQ(r.size(), 1U);
  EXPECT_EQ(r[0].label, 7U);
  EXPECT_FLOAT_EQ(r[0].distance, 0.0F);
}

TEST(Hnsw, RejectsBadInput) {
  EXPECT_THROW(HnswIndex({.dim = 0}), Error);
  EXPECT_THROW(HnswIndex({.dim = 4, .M = 1}), Error);
  HnswIndex index({.dim = 4, .capacity = 2});
  EXPECT_THROW(index.add(1, std::vector<float>{1, 2, 3}), Error);
  EXPECT_THROW((void)index.search(std::vector<float>{1, 2}, {}), Error);
  index.add(1, std::vector<float>{1, 2, 3, 4});
  EXPECT_THROW((void)index.search(std::vector<float>{1, 2, 3, 4}, {.use_quantized = true}), Error);
}

TEST(Hnsw, CapacityErrorAndReserve) {
  HnswIndex index({.dim = 2, .capacity = 2});
  index.add(1, std::vector<float>{0, 0});
  index.add(2, std::vector<float>{1, 1});
  EXPECT_THROW(index.add(3, std::vector<float>{2, 2}), CapacityError);
  index.reserve(100);
  EXPECT_EQ(index.capacity(), 100U);
  index.add(3, std::vector<float>{2, 2});
  EXPECT_EQ(index.size(), 3U);
  EXPECT_EQ(index.search(std::vector<float>{2.1F, 2.1F}, {.k = 1})[0].label, 3U);
  EXPECT_TRUE(index.check_integrity().ok);
}

TEST(Hnsw, DeleteHidesLabelsButKeepsGraphNavigable) {
  auto index = build(Metric::L2);
  const auto& f = fixture();
  // Delete every 3rd vector.
  for (label_t l = 0; l < kN; l += 3) ASSERT_TRUE(index.remove(l));
  EXPECT_FALSE(index.remove(0));  // already gone
  EXPECT_FALSE(index.contains(3));
  EXPECT_TRUE(index.contains(4));
  EXPECT_EQ(index.deleted_count(), (kN + 2) / 3);
  EXPECT_EQ(index.size(), kN - (kN + 2) / 3);

  // Ground truth over the surviving vectors only.
  FlatIndex flat(kDim, Metric::L2);
  std::vector<float> live;
  std::vector<label_t> live_labels;
  for (label_t l = 0; l < kN; ++l) {
    if (l % 3 == 0) continue;
    live.insert(live.end(), f.data.begin() + static_cast<std::ptrdiff_t>(l * kDim),
                f.data.begin() + static_cast<std::ptrdiff_t>((l + 1) * kDim));
    live_labels.push_back(l);
  }
  flat.add(live, live_labels);
  const auto truth = flat.search_batch(f.queries, 10, 0);
  const auto approx = index.search_batch(f.queries, {.k = 10, .ef = 128}, 0);
  for (const auto& res : approx) {
    for (const auto& r : res) EXPECT_NE(r.label % 3, 0U) << "deleted label returned";
  }
  EXPECT_GE(test::recall_at_k(approx, truth, 10), 0.95);
  EXPECT_TRUE(index.check_integrity().ok);

  // Compaction drops tombstones and preserves search quality.
  const HnswIndex compacted = index.compact(4);
  EXPECT_EQ(compacted.node_count(), live_labels.size());
  EXPECT_EQ(compacted.deleted_count(), 0U);
  const auto r = compacted.check_integrity();
  EXPECT_TRUE(r.ok);
  EXPECT_GE(r.reachable_level0, live_labels.size() * 999 / 1000);
  const auto approx2 = compacted.search_batch(f.queries, {.k = 10, .ef = 128}, 0);
  EXPECT_GE(test::recall_at_k(approx2, truth, 10), 0.97);
}

TEST(Hnsw, UpsertReplacesVector) {
  HnswIndex index({.dim = 2, .capacity = 16});
  index.add(1, std::vector<float>{0, 0});
  index.add(2, std::vector<float>{10, 10});
  index.add(1, std::vector<float>{9, 9});  // move label 1 next to label 2
  EXPECT_EQ(index.size(), 2U);
  EXPECT_EQ(index.node_count(), 3U);
  EXPECT_EQ(index.deleted_count(), 1U);
  EXPECT_EQ(index.get_vector(1), (std::vector<float>{9, 9}));
  const auto res = index.search(std::vector<float>{0, 0}, {.k = 2});
  ASSERT_EQ(res.size(), 2U);
  EXPECT_EQ(res[0].label, 1U);
  EXPECT_FLOAT_EQ(res[0].distance, 162.0F);
  EXPECT_TRUE(index.check_integrity().ok);
}

TEST(Hnsw, FilteredSearchOnlyReturnsMatchingNodes) {
  const auto& f = fixture();
  std::vector<std::uint64_t> tags(kN);
  for (std::size_t i = 0; i < kN; ++i) tags[i] = std::uint64_t{1} << (i % 10);  // 10% each
  HnswIndex index({.dim = kDim, .M = 16, .ef_construction = 100, .capacity = kN});
  index.add_batch(f.data, f.labels, 4, tags);
  FlatIndex flat(kDim, Metric::L2);
  flat.add(f.data, f.labels, tags);

  const TagFilter filter{.any_of = 1U << 3};
  const auto truth = flat.search_batch(f.queries, 10, 0, filter);
  // Graph traversal with the predicate applied during the walk.
  const auto approx = index.search_batch(
      f.queries, {.k = 10, .ef = 200, .filter = filter, .allow_exact_scan = false}, 0);
  for (const auto& res : approx) {
    EXPECT_EQ(res.size(), 10U);
    for (const auto& r : res) EXPECT_EQ(r.label % 10, 3U);
  }
  EXPECT_GE(test::recall_at_k(approx, truth, 10), 0.95);

  // The planner: a 10% filter on 10k nodes is cheaper to scan exactly.
  SearchStats stats;
  const std::span<const float> q0(f.queries.data(), kDim);
  const auto planned = index.search(q0, {.k = 10, .ef = 200, .filter = filter, .stats = &stats});
  EXPECT_TRUE(stats.exact_scan);
  EXPECT_EQ(planned, flat.search(q0, 10, filter));
  // ...while an unselective filter (bits 0-8 = 90%) stays on the graph.
  const TagFilter broad{.any_of = 0x1FF};
  (void)index.search(q0, {.k = 10, .ef = 64, .filter = broad, .stats = &stats});
  EXPECT_FALSE(stats.exact_scan);
  // Tag counters track deletes: after removing every tag-3 node the planner
  // estimates zero matches and the scan returns nothing.
  for (label_t l = 3; l < kN; l += 10) index.remove(l);
  EXPECT_TRUE(index.search(q0, {.k = 10, .ef = 64, .filter = filter, .stats = &stats}).empty());
  EXPECT_TRUE(stats.exact_scan);

  // Label predicate path.
  auto pred = [](label_t l) { return l >= 5000; };
  const auto res = index.search({f.queries.data(), kDim}, {.k = 10, .ef = 100, .predicate = pred});
  ASSERT_EQ(res.size(), 10U);
  for (const auto& r : res) EXPECT_GE(r.label, 5000U);

  // A filter that matches nothing returns nothing (both execution paths).
  for (const bool allow_scan : {true, false}) {
    const auto none = index.search(q0, {.k = 10,
                                        .ef = 50,
                                        .filter = TagFilter{.any_of = 1ULL << 40},
                                        .allow_exact_scan = allow_scan});
    EXPECT_TRUE(none.empty());
  }
}

TEST(Hnsw, ScalarBackendGivesSameResults) {
  const auto& f = fixture();
  HnswConfig cfg{.dim = kDim, .M = 12, .ef_construction = 64, .capacity = 2000};
  HnswIndex simd(cfg);
  cfg.simd = SimdBackend::Scalar;
  HnswIndex scalar(cfg);
  const std::span<const float> d(f.data.data(), 2000 * kDim);
  const std::span<const label_t> l(f.labels.data(), 2000);
  simd.add_batch(d, l, 1);
  scalar.add_batch(d, l, 1);
  const auto ra = simd.search_batch(f.queries, {.k = 10, .ef = 64}, 0);
  const auto rb = scalar.search_batch(f.queries, {.k = 10, .ef = 64}, 0);
  // Floating-point reassociation can flip near-ties, so compare as recall.
  EXPECT_GE(test::recall_at_k(ra, rb, 10), 0.99);
}

TEST(Hnsw, SearchStatsAreReported) {
  const auto index = build(Metric::L2);
  SearchStats stats;
  (void)index.search({fixture().queries.data(), kDim}, {.k = 10, .ef = 64, .stats = &stats});
  EXPECT_GT(stats.distance_computations, 64U);
  EXPECT_LT(stats.distance_computations, kN / 2) << "should be far fewer than brute force";
  EXPECT_GT(stats.hops, 0U);
}

TEST(Hnsw, MemoryUsageIsPlausible) {
  const auto index = build(Metric::L2);
  const MemoryUsage m = index.memory_usage();
  EXPECT_EQ(m.vectors, kN * kDim * sizeof(float));
  EXPECT_GE(m.graph, kN * (1 + 32) * sizeof(std::uint32_t));
  EXPECT_EQ(m.codes, 0U);
  EXPECT_EQ(m.mapped, 0U);
}

}  // namespace
}  // namespace vs
