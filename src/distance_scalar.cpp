// Portable reference kernels. Written as plain sequential loops on purpose:
// they are the ground truth that the SIMD kernels are tested against, and the
// fallback on targets without NEON/AVX2.
#include "vecsearch/distance.hpp"

namespace vs::kernels_scalar {

float l2sq(const float* a, const float* b, std::size_t dim) noexcept {
  float acc = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  float acc = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) acc += a[i] * b[i];
  return acc;
}

float sq8_l2(const float* q, const float* w, const std::uint8_t* code, std::size_t dim) noexcept {
  float acc = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = q[i] - static_cast<float>(code[i]);
    acc += w[i] * d * d;
  }
  return acc;
}

float sq8_dot(const float* q, const std::uint8_t* code, std::size_t dim) noexcept {
  float acc = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) acc += q[i] * static_cast<float>(code[i]);
  return acc;
}

}  // namespace vs::kernels_scalar
