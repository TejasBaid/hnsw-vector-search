#include <cmath>
#include <string>

#include "distance_internal.hpp"
#include "vecsearch/distance.hpp"

namespace vs {

Metric parse_metric(std::string_view name) {
  if (name == "l2" || name == "euclidean") return Metric::L2;
  if (name == "ip" || name == "dot" || name == "inner_product") return Metric::InnerProduct;
  if (name == "cosine" || name == "angular") return Metric::Cosine;
  throw Error("unknown metric '" + std::string(name) + "' (expected l2, ip or cosine)");
}

SimdBackend compiled_backend() noexcept {
#if VS_HAVE_NEON
  return SimdBackend::Neon;
#elif VS_HAVE_AVX2
  return SimdBackend::Avx2;
#else
  return SimdBackend::Scalar;
#endif
}

std::string_view to_string(SimdBackend b) noexcept {
  switch (b) {
    case SimdBackend::Scalar: return "scalar";
    case SimdBackend::Neon: return "neon";
    case SimdBackend::Avx2: return "avx2";
  }
  return "unknown";
}

SimdBackend parse_backend(std::string_view name) {
  SimdBackend b{};
  if (name == "auto") return compiled_backend();
  if (name == "scalar") {
    b = SimdBackend::Scalar;
  } else if (name == "neon") {
    b = SimdBackend::Neon;
  } else if (name == "avx2") {
    b = SimdBackend::Avx2;
  } else {
    throw Error("unknown SIMD backend '" + std::string(name) + "'");
  }
  (void)kernels(b);  // throws if not compiled in
  return b;
}

namespace {

constexpr KernelTable kScalarTable{&kernels_scalar::l2sq, &kernels_scalar::dot,
                                   &kernels_scalar::sq8_l2, &kernels_scalar::sq8_dot};
#if VS_HAVE_NEON
constexpr KernelTable kNeonTable{&kernels_neon::l2sq, &kernels_neon::dot, &kernels_neon::sq8_l2,
                                 &kernels_neon::sq8_dot};
#endif
#if VS_HAVE_AVX2
constexpr KernelTable kAvx2Table{&kernels_avx2::l2sq, &kernels_avx2::dot, &kernels_avx2::sq8_l2,
                                 &kernels_avx2::sq8_dot};
#endif

// 1 - <a, b>: the inner-product / cosine distance built on a dot kernel.
template <FloatKernel Dot>
float one_minus_dot(const float* a, const float* b, std::size_t dim) noexcept {
  return 1.0F - Dot(a, b, dim);
}

}  // namespace

const KernelTable& kernels(SimdBackend backend) {
  switch (backend) {
    case SimdBackend::Scalar: return kScalarTable;
    case SimdBackend::Neon:
#if VS_HAVE_NEON
      return kNeonTable;
#else
      break;
#endif
    case SimdBackend::Avx2:
#if VS_HAVE_AVX2
      return kAvx2Table;
#else
      break;
#endif
  }
  throw Error("SIMD backend '" + std::string(to_string(backend)) +
              "' is not compiled into this binary");
}

DistanceFn distance_function(Metric metric, SimdBackend backend) {
  switch (backend) {
    case SimdBackend::Scalar:
      return metric == Metric::L2 ? &kernels_scalar::l2sq : &one_minus_dot<&kernels_scalar::dot>;
#if VS_HAVE_NEON
    case SimdBackend::Neon:
      return metric == Metric::L2 ? &kernels_neon::l2sq : &one_minus_dot<&kernels_neon::dot>;
#endif
#if VS_HAVE_AVX2
    case SimdBackend::Avx2:
      return metric == Metric::L2 ? &kernels_avx2::l2sq : &one_minus_dot<&kernels_avx2::dot>;
#endif
    default: break;
  }
  throw Error("SIMD backend '" + std::string(to_string(backend)) +
              "' is not compiled into this binary");
}

void normalize(std::span<float> v) noexcept {
  double norm = 0.0;
  for (const float x : v) norm += static_cast<double>(x) * static_cast<double>(x);
  if (norm <= 0.0) return;
  const auto inv = static_cast<float>(1.0 / std::sqrt(norm));
  for (float& x : v) x *= inv;
}

double reference_distance(Metric metric, std::span<const float> a,
                          std::span<const float> b) noexcept {
  double acc = 0.0;
  const std::size_t n = a.size() < b.size() ? a.size() : b.size();
  if (metric == Metric::L2) {
    for (std::size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      acc += d * d;
    }
    return acc;
  }
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (metric == Metric::Cosine) {
    if (na <= 0.0 || nb <= 0.0) return 1.0;
    return 1.0 - acc / (std::sqrt(na) * std::sqrt(nb));
  }
  return 1.0 - acc;
}

}  // namespace vs
