// x86-64 AVX2 + FMA kernels. Enabled when the translation unit is compiled
// with AVX2 and FMA available (e.g. -march=native on Haswell or newer).
#include "distance_internal.hpp"

#if VS_HAVE_AVX2
#include <immintrin.h>

#include <cstring>

namespace vs::kernels_avx2 {

namespace {
inline float hsum256(__m256 v) noexcept {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// Widens 8 uint8 codes to a __m256 of floats.
inline __m256 widen_u8x8(const std::uint8_t* code) noexcept {
  std::int64_t bits = 0;
  std::memcpy(&bits, code, sizeof(bits));  // unaligned 8-byte load without cast-align UB
  const __m128i raw = _mm_cvtsi64_si128(bits);
  return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw));
}
}  // namespace

float l2sq(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
  }
  float acc = hsum256(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  for (; i + 8 <= dim; i += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  }
  float acc = hsum256(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) acc += a[i] * b[i];
  return acc;
}

float sq8_l2(const float* q, const float* w, const std::uint8_t* code, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + i), widen_u8x8(code + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(q + i + 8), widen_u8x8(code + i + 8));
    s0 = _mm256_fmadd_ps(_mm256_mul_ps(d0, d0), _mm256_loadu_ps(w + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_mul_ps(d1, d1), _mm256_loadu_ps(w + i + 8), s1);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + i), widen_u8x8(code + i));
    s0 = _mm256_fmadd_ps(_mm256_mul_ps(d0, d0), _mm256_loadu_ps(w + i), s0);
  }
  float acc = hsum256(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) {
    const float d = q[i] - static_cast<float>(code[i]);
    acc += w[i] * d * d;
  }
  return acc;
}

float sq8_dot(const float* q, const std::uint8_t* code, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), widen_u8x8(code + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), widen_u8x8(code + i + 8), s1);
  }
  for (; i + 8 <= dim; i += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), widen_u8x8(code + i), s0);
  }
  float acc = hsum256(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) acc += q[i] * static_cast<float>(code[i]);
  return acc;
}

}  // namespace vs::kernels_avx2

#endif  // VS_HAVE_AVX2
