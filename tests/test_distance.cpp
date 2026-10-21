#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/types.hpp"

namespace vs {
namespace {

std::vector<float> random_vec(std::size_t dim, std::mt19937& rng, float lo = -2.0F,
                              float hi = 2.0F) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(dim);
  for (float& x : v) x = d(rng);
  return v;
}

std::vector<SimdBackend> available_backends() {
  std::vector<SimdBackend> out{SimdBackend::Scalar};
  if (compiled_backend() != SimdBackend::Scalar) out.push_back(compiled_backend());
  return out;
}

// Tolerance for float32 sums of `dim` terms vs. a double-precision reference.
double tolerance(double magnitude, std::size_t dim) {
  return 1e-5 * static_cast<double>(dim) * std::max(1.0, std::abs(magnitude));
}

TEST(Distance, CompiledBackendIsReported) {
  const auto b = compiled_backend();
  EXPECT_FALSE(to_string(b).empty());
#if defined(__aarch64__)
  EXPECT_EQ(b, SimdBackend::Neon);
#endif
  EXPECT_EQ(parse_backend("scalar"), SimdBackend::Scalar);
  EXPECT_EQ(parse_backend("auto"), b);
  EXPECT_THROW((void)parse_backend("sse9"), Error);
}

// Every dimension from 1 to 300 exercises all main-loop / remainder paths of
// the 16-wide, 8-wide and 4-wide SIMD loops.
TEST(Distance, SimdMatchesScalarAndReferenceForAllTailLengths) {
  std::mt19937 rng(7);
  for (const SimdBackend backend : available_backends()) {
    const KernelTable& k = kernels(backend);
    for (std::size_t dim = 1; dim <= 300; ++dim) {
      const auto a = random_vec(dim, rng);
      const auto b = random_vec(dim, rng);
      const double ref_l2 = reference_distance(Metric::L2, a, b);
      const double ref_dot = 1.0 - reference_distance(Metric::InnerProduct, a, b);
      EXPECT_NEAR(k.l2sq(a.data(), b.data(), dim), ref_l2, tolerance(ref_l2, dim))
          << to_string(backend) << " dim=" << dim;
      EXPECT_NEAR(k.dot(a.data(), b.data(), dim), ref_dot, tolerance(ref_dot, dim))
          << to_string(backend) << " dim=" << dim;
      EXPECT_NEAR(k.l2sq(a.data(), b.data(), dim), kernels_scalar::l2sq(a.data(), b.data(), dim),
                  tolerance(ref_l2, dim));
      EXPECT_NEAR(k.dot(a.data(), b.data(), dim), kernels_scalar::dot(a.data(), b.data(), dim),
                  tolerance(ref_dot, dim));
    }
  }
}

TEST(Distance, Sq8KernelsMatchScalar) {
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> byte(0, 255);
  for (const SimdBackend backend : available_backends()) {
    const KernelTable& k = kernels(backend);
    for (std::size_t dim = 1; dim <= 200; ++dim) {
      const auto q = random_vec(dim, rng, 0.0F, 255.0F);
      const auto w = random_vec(dim, rng, 0.0F, 0.01F);
      std::vector<std::uint8_t> code(dim);
      for (auto& c : code) c = static_cast<std::uint8_t>(byte(rng));
      const auto ref_l2 =
          static_cast<double>(kernels_scalar::sq8_l2(q.data(), w.data(), code.data(), dim));
      const auto ref_dot = static_cast<double>(kernels_scalar::sq8_dot(q.data(), code.data(), dim));
      EXPECT_NEAR(k.sq8_l2(q.data(), w.data(), code.data(), dim), ref_l2, tolerance(ref_l2, dim));
      EXPECT_NEAR(k.sq8_dot(q.data(), code.data(), dim), ref_dot, tolerance(ref_dot, dim));
    }
  }
}

TEST(Distance, MetricSemantics) {
  const std::vector<float> a{1.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> b{0.0F, 1.0F, 0.0F, 0.0F};
  for (const SimdBackend backend : available_backends()) {
    EXPECT_FLOAT_EQ(distance_function(Metric::L2, backend)(a.data(), b.data(), 4), 2.0F);
    EXPECT_FLOAT_EQ(distance_function(Metric::InnerProduct, backend)(a.data(), b.data(), 4), 1.0F);
    EXPECT_FLOAT_EQ(distance_function(Metric::InnerProduct, backend)(a.data(), a.data(), 4), 0.0F);
    EXPECT_FLOAT_EQ(distance_function(Metric::L2, backend)(a.data(), a.data(), 4), 0.0F);
  }
}

TEST(Distance, NormalizeProducesUnitVectorsAndLeavesZeroAlone) {
  std::vector<float> v{3.0F, 4.0F};
  normalize(v);
  EXPECT_NEAR(v[0], 0.6F, 1e-6F);
  EXPECT_NEAR(v[1], 0.8F, 1e-6F);
  std::vector<float> z{0.0F, 0.0F, 0.0F};
  normalize(z);
  EXPECT_EQ(z, (std::vector<float>{0.0F, 0.0F, 0.0F}));
}

TEST(Distance, CosineOnNormalizedVectorsMatchesReference) {
  std::mt19937 rng(3);
  const auto fn = distance_function(Metric::Cosine);
  for (std::size_t dim : {3UL, 17UL, 100UL, 768UL}) {
    auto a = random_vec(dim, rng);
    auto b = random_vec(dim, rng);
    const double ref = reference_distance(Metric::Cosine, a, b);
    normalize(a);
    normalize(b);
    EXPECT_NEAR(fn(a.data(), b.data(), dim), ref, 1e-5);
  }
}

TEST(Distance, ParseMetric) {
  EXPECT_EQ(parse_metric("l2"), Metric::L2);
  EXPECT_EQ(parse_metric("angular"), Metric::Cosine);
  EXPECT_EQ(parse_metric("dot"), Metric::InnerProduct);
  EXPECT_THROW((void)parse_metric("hamming"), Error);
}

}  // namespace
}  // namespace vs
