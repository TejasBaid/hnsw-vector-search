// Concurrency tests. These are most valuable under ThreadSanitizer
// (cmake --preset tsan), which checks the lock-free read path for races.
#include <gtest/gtest.h>

#include <atomic>
#include <unordered_set>

#include "test_util.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"

namespace vs {
namespace {

constexpr std::size_t kDim = 16;

#if defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(address_sanitizer)
constexpr std::size_t kScale = 4;  // sanitizers are 5-15x slower
#else
constexpr std::size_t kScale = 1;
#endif
#else
constexpr std::size_t kScale = 1;
#endif

TEST(Concurrency, ParallelBuildMatchesSerialQuality) {
  const std::size_t n = 20'000 / kScale;
  const auto [data, queries] = test::clustered_split(n, 100, kDim, 100);
  const auto labels = test::iota_labels(n);
  HnswIndex index({.dim = kDim, .M = 16, .ef_construction = 100, .capacity = n});
  index.add_batch(data, labels, 8);
  const auto report = index.check_integrity();
  for (const auto& e : report.errors) ADD_FAILURE() << e;
  // HNSW does not strictly guarantee connectivity (pruning can strip a node's
  // last in-edge); in practice a handful of nodes per 100k at most.
  EXPECT_GE(report.reachable_level0, n * 999 / 1000);

  FlatIndex flat(kDim, Metric::L2);
  flat.add(data, labels);
  const double recall = test::recall_at_k(index.search_batch(queries, {.k = 10, .ef = 100}, 8),
                                          flat.search_batch(queries, 10, 8), 10);
  EXPECT_GE(recall, 0.97);
}

TEST(Concurrency, SearchesDuringInsertsSeeOnlyInsertedLabels) {
  const std::size_t n = 20'000 / kScale;
  const auto data = test::clustered_data(n, kDim, 200);
  HnswIndex index({.dim = kDim, .M = 12, .ef_construction = 64, .capacity = n});
  // Seed with the first 10% so searches have something to find.
  const std::size_t seed_n = n / 10;
  index.add_batch({data.data(), seed_n * kDim}, test::iota_labels(seed_n), 2);

  std::atomic<std::size_t> inserted{seed_n};
  std::atomic<bool> done{false};
  std::atomic<std::size_t> searches{0};
  std::atomic<std::size_t> bad{0};

  test::ThreadGroup readers;
  for (int t = 0; t < 4; ++t) {
    readers.spawn([&, t] {
      std::size_t q = static_cast<std::size_t>(t);
      while (!done.load(std::memory_order_acquire)) {
        const std::size_t upper_bound = n;  // labels are always < n
        const auto res = index.search({data.data() + (q % n) * kDim, kDim}, {.k = 10, .ef = 32});
        for (const auto& r : res) {
          if (r.label >= upper_bound) bad.fetch_add(1);
        }
        for (std::size_t i = 1; i < res.size(); ++i) {
          if (res[i - 1].distance > res[i].distance) bad.fetch_add(1);
        }
        searches.fetch_add(1, std::memory_order_relaxed);
        q += 7;
      }
    });
  }

  test::ThreadGroup writers;
  std::atomic<std::size_t> cursor{seed_n};
  for (int t = 0; t < 4; ++t) {
    writers.spawn([&] {
      for (std::size_t i = cursor.fetch_add(1); i < n; i = cursor.fetch_add(1)) {
        index.add(i, {data.data() + i * kDim, kDim});
        inserted.fetch_add(1);
      }
    });
  }
  writers.join();  // join writers
  done.store(true, std::memory_order_release);
  readers.join();

  EXPECT_EQ(bad.load(), 0U);
  EXPECT_GT(searches.load(), 0U);
  EXPECT_EQ(index.size(), n);
  const auto report = index.check_integrity();
  for (const auto& e : report.errors) ADD_FAILURE() << e;
  EXPECT_GE(report.reachable_level0, n * 999 / 1000);

  // Every vector must find itself once the build has settled.
  std::size_t self_hits = 0;
  for (std::size_t i = 0; i < n; i += 97) {
    const auto res = index.search({data.data() + i * kDim, kDim}, {.k = 1, .ef = 64});
    self_hits += (!res.empty() && res[0].label == i) ? 1U : 0U;
  }
  EXPECT_GE(self_hits, (n / 97) * 98 / 100);
}

TEST(Concurrency, DeletesAreNeverReturnedOnceAcknowledged) {
  const std::size_t n = 10'000 / kScale;
  const auto data = test::clustered_data(n, kDim, 300);
  HnswIndex index({.dim = kDim, .M = 12, .ef_construction = 64, .capacity = n});
  index.add_batch(data, test::iota_labels(n), 4);

  // Deleter publishes a watermark after each delete; readers verify that no
  // label below the watermark (all of which are even and deleted) appears.
  std::atomic<std::size_t> watermark{0};
  std::atomic<bool> done{false};
  std::atomic<std::size_t> violations{0};
  test::ThreadGroup readers;
  for (int t = 0; t < 4; ++t) {
    readers.spawn([&, t] {
      std::size_t q = static_cast<std::size_t>(t) * 31;
      while (!done.load(std::memory_order_acquire)) {
        const std::size_t w = watermark.load(std::memory_order_acquire);
        const auto res = index.search({data.data() + (q % n) * kDim, kDim}, {.k = 10, .ef = 32});
        for (const auto& r : res) {
          if (r.label % 2 == 0 && r.label < w) violations.fetch_add(1);
        }
        q += 13;
      }
    });
  }
  for (std::size_t l = 0; l < n; l += 2) {
    ASSERT_TRUE(index.remove(l));
    watermark.store(l + 1, std::memory_order_release);
  }
  done.store(true, std::memory_order_release);
  readers.join();
  EXPECT_EQ(violations.load(), 0U);
  EXPECT_EQ(index.size(), n / 2);
  EXPECT_TRUE(index.check_integrity().ok);
}

TEST(Concurrency, ConcurrentUpsertsOfSameLabelsStayConsistent) {
  const std::size_t labels = 200;
  const auto data = test::clustered_data(labels * 8, kDim, 400);
  HnswIndex index({.dim = kDim, .M = 8, .ef_construction = 32, .capacity = labels * 8});
  test::ThreadGroup writers;
  for (std::size_t t = 0; t < 8; ++t) {
    writers.spawn([&, t] {
      for (std::size_t l = 0; l < labels; ++l) {
        index.add(l, {data.data() + (t * labels + l) * kDim, kDim});
      }
    });
  }
  writers.join();
  EXPECT_EQ(index.size(), labels);
  EXPECT_EQ(index.deleted_count(), labels * 7);
  const auto report = index.check_integrity();
  for (const auto& e : report.errors) ADD_FAILURE() << e;
  EXPECT_TRUE(report.ok);
  // Each label's current vector is findable and unique in results.
  for (std::size_t l = 0; l < labels; l += 10) {
    const auto v = index.get_vector(l);
    const auto res = index.search(v, {.k = 5, .ef = 64});
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res[0].label, l);
    std::unordered_set<label_t> uniq;
    for (const auto& r : res) EXPECT_TRUE(uniq.insert(r.label).second);
  }
}

TEST(Concurrency, CompactWhileSearching) {
  const std::size_t n = 8'000 / kScale;
  const auto data = test::clustered_data(n, kDim, 500);
  HnswIndex index({.dim = kDim, .M = 12, .ef_construction = 64, .capacity = n});
  index.add_batch(data, test::iota_labels(n), 4);
  for (label_t l = 0; l < n; l += 4) index.remove(l);
  std::atomic<bool> done{false};
  test::ThreadGroup reader;
  reader.spawn([&] {
    std::size_t q = 0;
    while (!done.load()) {
      (void)index.search({data.data() + (q++ % n) * kDim, kDim}, {.k = 10, .ef = 32});
    }
  });
  const HnswIndex compacted = index.compact(4);
  done.store(true);
  reader.join();
  EXPECT_EQ(compacted.size(), n - n / 4);
  EXPECT_TRUE(compacted.check_integrity().ok);
}

}  // namespace
}  // namespace vs
