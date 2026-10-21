// Micro-benchmarks for the distance kernels (Google Benchmark).
//
//   ./distance_bench --benchmark_filter=L2
//
// Vectors live in a 4 MiB pool cycled through per iteration so the numbers
// reflect L1/L2-resident data rather than a single hot pair of registers.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/quantization.hpp"

namespace {

using namespace vs;

constexpr std::size_t kPoolBytes = 4U << 20;

struct Pool {
  std::vector<float> data;
  std::size_t dim;
  std::size_t count;
  explicit Pool(std::size_t d) : dim(d), count(kPoolBytes / (d * sizeof(float))) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    data.resize(count * dim);
    for (float& x : data) x = u(rng);
  }
  [[nodiscard]] const float* row(std::size_t i) const { return data.data() + (i % count) * dim; }
};

template <SimdBackend B, bool L2>
void BM_Float(benchmark::State& state) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const Pool pool(dim);
  const KernelTable& k = kernels(B);
  const auto fn = L2 ? k.l2sq : k.dot;
  const float* q = pool.row(0);
  std::size_t i = 1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fn(q, pool.row(i++), dim));
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(dim * sizeof(float)));
}

template <SimdBackend B>
void BM_Sq8(benchmark::State& state) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const Pool pool(dim);
  const std::size_t train_rows = std::min<std::size_t>(1000, pool.count);
  const auto sq = ScalarQuantizer::train(std::span<const float>(pool.data.data(), train_rows * dim),
                                         dim, Metric::L2);
  std::vector<std::uint8_t> codes(pool.count * dim);
  for (std::size_t r = 0; r < pool.count; ++r) sq.encode(pool.row(r), codes.data() + r * dim);
  const auto q = sq.prepare(pool.row(0), B);
  std::size_t i = 1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(q.distance(codes.data() + (i++ % pool.count) * dim));
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_PqAdc(benchmark::State& state) {
  constexpr std::size_t dim = 128;
  const auto m = static_cast<std::size_t>(state.range(0));
  const Pool pool(dim);
  QuantizerConfig cfg{.type = QuantizerType::PQ, .pq_subspaces = m, .kmeans_iters = 4};
  const auto pq = ProductQuantizer::train(
      std::span<const float>(pool.data.data(), std::min<std::size_t>(4096, pool.count) * dim), dim,
      Metric::L2, cfg, 4);
  std::vector<std::uint8_t> codes(pool.count * m);
  for (std::size_t r = 0; r < pool.count; ++r) pq.encode(pool.row(r), codes.data() + r * m);
  const auto q = pq.prepare(pool.row(0));
  std::size_t i = 1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(q.distance(codes.data() + (i++ % pool.count) * m));
  }
  state.SetItemsProcessed(state.iterations());
}

#define VS_DIMS ->Arg(96)->Arg(100)->Arg(128)->Arg(384)->Arg(768)->Arg(1536)

BENCHMARK(BM_Float<SimdBackend::Scalar, true>)->Name("L2/scalar") VS_DIMS;
BENCHMARK(BM_Float<SimdBackend::Scalar, false>)->Name("Dot/scalar") VS_DIMS;
BENCHMARK(BM_Sq8<SimdBackend::Scalar>)->Name("SQ8-L2/scalar") VS_DIMS;
#if defined(__aarch64__)
BENCHMARK(BM_Float<SimdBackend::Neon, true>)->Name("L2/neon") VS_DIMS;
BENCHMARK(BM_Float<SimdBackend::Neon, false>)->Name("Dot/neon") VS_DIMS;
BENCHMARK(BM_Sq8<SimdBackend::Neon>)->Name("SQ8-L2/neon") VS_DIMS;
#elif defined(__AVX2__) && defined(__FMA__)
BENCHMARK(BM_Float<SimdBackend::Avx2, true>)->Name("L2/avx2") VS_DIMS;
BENCHMARK(BM_Float<SimdBackend::Avx2, false>)->Name("Dot/avx2") VS_DIMS;
BENCHMARK(BM_Sq8<SimdBackend::Avx2>)->Name("SQ8-L2/avx2") VS_DIMS;
#endif
BENCHMARK(BM_PqAdc)->Name("PQ-ADC/d128")->Arg(16)->Arg(32)->Arg(64);

}  // namespace

BENCHMARK_MAIN();
