// Distance kernels.
//
// Every kernel exists in a portable scalar version (the reference used by the
// tests) and, when the target supports it, an ARM NEON or x86 AVX2+FMA
// version. The SIMD path is chosen at compile time from the target ISA
// (__ARM_NEON / __AVX2__ && __FMA__); the scalar path can still be selected at
// runtime, which the benchmarks use to quantify the SIMD speed-up.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "vecsearch/types.hpp"

namespace vs {

enum class SimdBackend : std::uint8_t { Scalar, Neon, Avx2 };

/// The best backend compiled into this binary.
[[nodiscard]] SimdBackend compiled_backend() noexcept;
[[nodiscard]] std::string_view to_string(SimdBackend b) noexcept;
/// Parses "auto" (=compiled_backend()), "scalar", "neon", "avx2". Throws if the
/// requested backend is not compiled in.
[[nodiscard]] SimdBackend parse_backend(std::string_view name);

/// Signature shared by all float kernels: (a, b, dim) -> float.
using FloatKernel = float (*)(const float*, const float*, std::size_t) noexcept;

/// Weighted SQ8 L2 kernel: sum_i w[i] * (q[i] - code[i])^2.
using Sq8L2Kernel = float (*)(const float* q, const float* w, const std::uint8_t* code,
                              std::size_t dim) noexcept;
/// SQ8 dot kernel: sum_i q[i] * code[i].
using Sq8DotKernel = float (*)(const float* q, const std::uint8_t* code, std::size_t dim) noexcept;

/// The raw kernels for one backend.
struct KernelTable {
  FloatKernel l2sq;      ///< squared Euclidean distance
  FloatKernel dot;       ///< inner product
  Sq8L2Kernel sq8_l2;    ///< asymmetric float-vs-uint8 weighted L2
  Sq8DotKernel sq8_dot;  ///< asymmetric float-vs-uint8 dot
};

/// Kernel table for `backend` (throws if not compiled in).
[[nodiscard]] const KernelTable& kernels(SimdBackend backend = compiled_backend());

namespace kernels_scalar {
[[nodiscard]] float l2sq(const float* a, const float* b, std::size_t dim) noexcept;
[[nodiscard]] float dot(const float* a, const float* b, std::size_t dim) noexcept;
[[nodiscard]] float sq8_l2(const float* q, const float* w, const std::uint8_t* code,
                           std::size_t dim) noexcept;
[[nodiscard]] float sq8_dot(const float* q, const std::uint8_t* code, std::size_t dim) noexcept;
}  // namespace kernels_scalar

/// Index-level distance function for a metric. Vectors passed to a Cosine
/// distance function must already be normalised (the index does this).
using DistanceFn = FloatKernel;
[[nodiscard]] DistanceFn distance_function(Metric metric, SimdBackend backend = compiled_backend());

/// L2-normalises `v` in place. Zero vectors are left unchanged.
void normalize(std::span<float> v) noexcept;

/// Reference (double-precision accumulation) distance, used by tests.
[[nodiscard]] double reference_distance(Metric metric, std::span<const float> a,
                                        std::span<const float> b) noexcept;

}  // namespace vs
