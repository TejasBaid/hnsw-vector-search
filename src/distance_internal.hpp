// Compile-time ISA detection and per-backend kernel declarations.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) && defined(__aarch64__)
#define VS_HAVE_NEON 1
#else
#define VS_HAVE_NEON 0
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define VS_HAVE_AVX2 1
#else
#define VS_HAVE_AVX2 0
#endif

namespace vs {

#if VS_HAVE_NEON
namespace kernels_neon {
[[nodiscard]] float l2sq(const float* a, const float* b, std::size_t dim) noexcept;
[[nodiscard]] float dot(const float* a, const float* b, std::size_t dim) noexcept;
[[nodiscard]] float sq8_l2(const float* q, const float* w, const std::uint8_t* code,
                           std::size_t dim) noexcept;
[[nodiscard]] float sq8_dot(const float* q, const std::uint8_t* code, std::size_t dim) noexcept;
}  // namespace kernels_neon
#endif

#if VS_HAVE_AVX2
namespace kernels_avx2 {
[[nodiscard]] float l2sq(const float* a, const float* b, std::size_t dim) noexcept;
[[nodiscard]] float dot(const float* a, const float* b, std::size_t dim) noexcept;
[[nodiscard]] float sq8_l2(const float* q, const float* w, const std::uint8_t* code,
                           std::size_t dim) noexcept;
[[nodiscard]] float sq8_dot(const float* q, const std::uint8_t* code, std::size_t dim) noexcept;
}  // namespace kernels_avx2
#endif

}  // namespace vs
