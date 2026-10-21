#include <gtest/gtest.h>

#include <algorithm>

#include "test_util.hpp"
#include "vecsearch/flat_index.hpp"

namespace vs {
namespace {

TEST(FlatIndex, ReturnsExactNearestNeighboursInOrder) {
  constexpr std::size_t dim = 8;
  constexpr std::size_t n = 500;
  const auto data = test::clustered_data(n, dim, 1);
  FlatIndex index(dim, Metric::L2);
  index.add(data, test::iota_labels(n));
  ASSERT_EQ(index.size(), n);

  const std::span<const float> q(data.data() + 17 * dim, dim);
  const auto res = index.search(q, 10);
  ASSERT_EQ(res.size(), 10U);
  EXPECT_EQ(res[0].label, 17U);
  EXPECT_FLOAT_EQ(res[0].distance, 0.0F);
  EXPECT_TRUE(std::is_sorted(res.begin(), res.end(),
                             [](auto& a, auto& b) { return a.distance < b.distance; }));

  // Cross-check against a full sort by reference distance.
  std::vector<std::pair<double, label_t>> all;
  for (std::size_t i = 0; i < n; ++i) {
    all.emplace_back(reference_distance(Metric::L2, q, {data.data() + i * dim, dim}), i);
  }
  std::sort(all.begin(), all.end());
  for (std::size_t i = 0; i < 10; ++i) EXPECT_EQ(res[i].label, all[i].second);
}

TEST(FlatIndex, TagFilterAndPredicate) {
  constexpr std::size_t dim = 4;
  const auto data = test::clustered_data(200, dim, 2);
  std::vector<std::uint64_t> tags(200);
  for (std::size_t i = 0; i < 200; ++i) tags[i] = (i % 2 == 0) ? 0b01U : 0b10U;
  FlatIndex index(dim, Metric::L2);
  index.add(data, test::iota_labels(200), tags);

  TagFilter even{.any_of = 0b01U};
  for (const auto& r : index.search({data.data(), dim}, 20, even)) EXPECT_EQ(r.label % 2, 0U);

  auto below_50 = [](label_t l) { return l < 50; };
  for (const auto& r : index.search({data.data(), dim}, 20, {}, below_50)) EXPECT_LT(r.label, 50U);
}

TEST(FlatIndex, KLargerThanSizeReturnsEverything) {
  FlatIndex index(2, Metric::Cosine);
  const std::vector<float> v{1, 0, 0, 1, 1, 1};
  index.add(v, test::iota_labels(3));
  EXPECT_EQ(index.search(std::vector<float>{1, 0}, 10).size(), 3U);
}

}  // namespace
}  // namespace vs
