// ARM NEON kernels (AArch64). Four independent accumulators hide the 4-cycle
// FMA latency of Apple/Neoverse cores; the loop body consumes 16 floats.
#include "distance_internal.hpp"

#if VS_HAVE_NEON
#include <arm_neon.h>

namespace vs::kernels_neon {

float l2sq(const float* a, const float* b, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0F);
  float32x4_t s1 = vdupq_n_f32(0.0F);
  float32x4_t s2 = vdupq_n_f32(0.0F);
  float32x4_t s3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  for (; i + 4 <= dim; i += 4) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, d0, d0);
  }
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0F);
  float32x4_t s1 = vdupq_n_f32(0.0F);
  float32x4_t s2 = vdupq_n_f32(0.0F);
  float32x4_t s3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= dim; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) acc += a[i] * b[i];
  return acc;
}

namespace {
// Widens 16 uint8 codes to four float32x4 vectors.
inline void widen_u8x16(const std::uint8_t* code, float32x4_t& c0, float32x4_t& c1, float32x4_t& c2,
                        float32x4_t& c3) noexcept {
  const uint8x16_t raw = vld1q_u8(code);
  const uint16x8_t lo = vmovl_u8(vget_low_u8(raw));
  const uint16x8_t hi = vmovl_u8(vget_high_u8(raw));
  c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
  c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
  c2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
  c3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
}
}  // namespace

float sq8_l2(const float* q, const float* w, const std::uint8_t* code, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0F);
  float32x4_t s1 = vdupq_n_f32(0.0F);
  float32x4_t s2 = vdupq_n_f32(0.0F);
  float32x4_t s3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    float32x4_t c0;
    float32x4_t c1;
    float32x4_t c2;
    float32x4_t c3;
    widen_u8x16(code + i, c0, c1, c2, c3);
    const float32x4_t d0 = vsubq_f32(vld1q_f32(q + i), c0);
    const float32x4_t d1 = vsubq_f32(vld1q_f32(q + i + 4), c1);
    const float32x4_t d2 = vsubq_f32(vld1q_f32(q + i + 8), c2);
    const float32x4_t d3 = vsubq_f32(vld1q_f32(q + i + 12), c3);
    s0 = vfmaq_f32(s0, vmulq_f32(d0, d0), vld1q_f32(w + i));
    s1 = vfmaq_f32(s1, vmulq_f32(d1, d1), vld1q_f32(w + i + 4));
    s2 = vfmaq_f32(s2, vmulq_f32(d2, d2), vld1q_f32(w + i + 8));
    s3 = vfmaq_f32(s3, vmulq_f32(d3, d3), vld1q_f32(w + i + 12));
  }
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    const float d = q[i] - static_cast<float>(code[i]);
    acc += w[i] * d * d;
  }
  return acc;
}

float sq8_dot(const float* q, const std::uint8_t* code, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0F);
  float32x4_t s1 = vdupq_n_f32(0.0F);
  float32x4_t s2 = vdupq_n_f32(0.0F);
  float32x4_t s3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    float32x4_t c0;
    float32x4_t c1;
    float32x4_t c2;
    float32x4_t c3;
    widen_u8x16(code + i, c0, c1, c2, c3);
    s0 = vfmaq_f32(s0, vld1q_f32(q + i), c0);
    s1 = vfmaq_f32(s1, vld1q_f32(q + i + 4), c1);
    s2 = vfmaq_f32(s2, vld1q_f32(q + i + 8), c2);
    s3 = vfmaq_f32(s3, vld1q_f32(q + i + 12), c3);
  }
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) acc += q[i] * static_cast<float>(code[i]);
  return acc;
}

}  // namespace vs::kernels_neon

#endif  // VS_HAVE_NEON
