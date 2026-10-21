#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "test_util.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/quantization.hpp"

namespace vs {
namespace {

constexpr std::size_t kDim = 32;

TEST(ScalarQuantizer, RoundTripErrorIsBoundedByHalfStep) {
  const auto data = test::clustered_data(2000, kDim, 5);
  const auto sq = ScalarQuantizer::train(data, kDim, Metric::L2);
  std::vector<std::uint8_t> code(kDim);
  std::vector<float> back(kDim);
  float max_range = 0.0F;
  for (std::size_t j = 0; j < kDim; ++j) {
    float lo = 1e9F;
    float hi = -1e9F;
    for (std::size_t i = 0; i < 2000; ++i) {
      lo = std::min(lo, data[i * kDim + j]);
      hi = std::max(hi, data[i * kDim + j]);
    }
    max_range = std::max(max_range, hi - lo);
  }
  const float half_step = max_range / 255.0F / 2.0F + 1e-6F;
  for (std::size_t i = 0; i < 2000; ++i) {
    sq.encode(&data[i * kDim], code.data());
    sq.decode(code.data(), back.data());
    for (std::size_t j = 0; j < kDim; ++j)
      EXPECT_LE(std::abs(back[j] - data[i * kDim + j]), half_step);
  }
}

TEST(ScalarQuantizer, AsymmetricDistanceMatchesDecodedDistance) {
  const auto data = test::clustered_data(1000, kDim, 6);
  for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
    const auto sq = ScalarQuantizer::train(data, kDim, metric);
    const auto q = sq.prepare(data.data(), compiled_backend());
    std::vector<std::uint8_t> code(kDim);
    std::vector<float> back(kDim);
    for (std::size_t i = 1; i < 100; ++i) {
      sq.encode(&data[i * kDim], code.data());
      sq.decode(code.data(), back.data());
      const double expected = reference_distance(metric, {data.data(), kDim}, back);
      EXPECT_NEAR(q.distance(code.data()), expected, 1e-3 * std::max(1.0, std::abs(expected)));
    }
  }
}

TEST(ScalarQuantizer, ConstantDimensionsAreHandled) {
  std::vector<float> data(100 * 4);
  for (std::size_t i = 0; i < 100; ++i) {
    data[i * 4 + 0] = static_cast<float>(i);
    data[i * 4 + 1] = 5.0F;  // constant
    data[i * 4 + 2] = -static_cast<float>(i);
    data[i * 4 + 3] = 0.0F;  // constant
  }
  const auto sq = ScalarQuantizer::train(data, 4, Metric::L2);
  const std::vector<float> query{10.0F, 7.0F, -10.0F, 1.0F};
  const auto q = sq.prepare(query.data(), compiled_backend());
  std::vector<std::uint8_t> code(4);
  sq.encode(&data[10 * 4], code.data());
  // (7-5)^2 + (1-0)^2 = 5 from the constant dimensions, ~0 from the others.
  EXPECT_NEAR(q.distance(code.data()), 5.0F, 0.1F);
}

TEST(ProductQuantizer, ReconstructionBeatsRandomCodebook) {
  const auto data = test::clustered_data(5000, kDim, 7);
  QuantizerConfig cfg{.type = QuantizerType::PQ, .pq_subspaces = 8, .kmeans_iters = 10};
  const auto pq = ProductQuantizer::train(data, kDim, Metric::L2, cfg, 4);
  EXPECT_EQ(pq.code_size(), 8U);
  std::vector<std::uint8_t> code(8);
  std::vector<float> back(kDim);
  double err = 0.0;
  double energy = 0.0;
  for (std::size_t i = 0; i < 5000; ++i) {
    pq.encode(&data[i * kDim], code.data());
    pq.decode(code.data(), back.data());
    err += reference_distance(Metric::L2, {&data[i * kDim], kDim}, back);
    energy +=
        reference_distance(Metric::L2, {&data[i * kDim], kDim}, std::vector<float>(kDim, 0.0F));
  }
  EXPECT_LT(err / energy, 0.05) << "relative reconstruction error";
}

TEST(ProductQuantizer, AdcMatchesDecodedDistance) {
  const auto data = test::clustered_data(3000, kDim, 8);
  for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
    QuantizerConfig cfg{.type = QuantizerType::PQ, .pq_subspaces = 4, .kmeans_iters = 5};
    const auto pq = ProductQuantizer::train(data, kDim, metric, cfg, 2);
    const auto q = pq.prepare(data.data());
    std::vector<std::uint8_t> code(4);
    std::vector<float> back(kDim);
    for (std::size_t i = 1; i < 50; ++i) {
      pq.encode(&data[i * kDim], code.data());
      pq.decode(code.data(), back.data());
      const double expected = reference_distance(metric, {data.data(), kDim}, back);
      EXPECT_NEAR(q.distance(code.data()), expected, 1e-3 * std::max(1.0, std::abs(expected)));
    }
  }
}

TEST(ProductQuantizer, RejectsBadGeometry) {
  const auto data = test::clustered_data(300, 30, 9);
  QuantizerConfig cfg{.type = QuantizerType::PQ, .pq_subspaces = 7};
  EXPECT_THROW((void)ProductQuantizer::train(data, 30, Metric::L2, cfg, 1), Error);
}

struct QuantCase {
  QuantizerType type;
  std::size_t subspaces;
  double min_recall_rerank;
};

class HnswQuantized : public ::testing::TestWithParam<QuantCase> {};

TEST_P(HnswQuantized, RerankingRecoversRecall) {
  const auto [type, subspaces, min_recall] = GetParam();
  constexpr std::size_t n = 8000;
  const auto [data, queries] = test::clustered_split(n, 100, kDim, 10);
  const auto labels = test::iota_labels(n);
  HnswIndex index({.dim = kDim, .M = 16, .ef_construction = 100, .capacity = n});
  index.add_batch(data, labels, 4);
  index.train_quantizer({.type = type, .pq_subspaces = subspaces, .kmeans_iters = 10}, 4);
  EXPECT_EQ(index.quantizer_type(), type);
  EXPECT_GT(index.memory_usage().codes, 0U);

  FlatIndex flat(kDim, Metric::L2);
  flat.add(data, labels);
  const auto truth = flat.search_batch(queries, 10, 0);

  const auto no_rerank =
      index.search_batch(queries, {.k = 10, .ef = 128, .use_quantized = true, .rerank = 0}, 0);
  const auto rerank = index.search_batch(queries, {.k = 10, .ef = 128, .use_quantized = true}, 0);
  const double r0 = test::recall_at_k(no_rerank, truth, 10);
  const double r1 = test::recall_at_k(rerank, truth, 10);
  EXPECT_GE(r1, min_recall);
  EXPECT_GE(r1 + 1e-9, r0) << "re-ranking must not hurt recall";

  // Codes are maintained for vectors inserted after training.
  const auto extra = test::clustered_data(1, kDim, 12);
  index.reserve(n + 1);
  index.add(999999, extra);
  const auto hit = index.search(extra, {.k = 1, .ef = 64, .use_quantized = true});
  ASSERT_EQ(hit.size(), 1U);
  EXPECT_EQ(hit[0].label, 999999U);
}

INSTANTIATE_TEST_SUITE_P(Types, HnswQuantized,
                         ::testing::Values(QuantCase{QuantizerType::SQ8, 0, 0.97},
                                           QuantCase{QuantizerType::PQ, 8, 0.90}),
                         [](const auto& p) {
                           return p.param.type == QuantizerType::SQ8 ? std::string("SQ8")
                                                                     : std::string("PQ");
                         });

}  // namespace
}  // namespace vs
