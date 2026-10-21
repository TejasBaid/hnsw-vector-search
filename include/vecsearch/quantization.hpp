// Vector compression for the graph traversal working set.
//
// ScalarQuantizer (SQ8): per-dimension affine map to uint8 -> 4x smaller.
// ProductQuantizer (PQ): split into m sub-vectors, k-means codebook of 256
//   centroids per sub-space, 1 byte per sub-vector -> 4*dim/m x smaller.
//
// Both use *asymmetric* distance computation: the query stays in float and a
// per-query state (SQ8: rescaled query; PQ: m x 256 lookup table) is built
// once, so each candidate costs one cheap kernel call on its code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vecsearch/aligned_buffer.hpp"
#include "vecsearch/distance.hpp"
#include "vecsearch/types.hpp"

namespace vs {

enum class QuantizerType : std::uint32_t { None = 0, SQ8 = 1, PQ = 2 };

struct QuantizerConfig {
  QuantizerType type = QuantizerType::None;
  std::size_t pq_subspaces = 16;   ///< PQ: number of sub-vectors (must divide dim)
  std::size_t train_size = 65536;  ///< vectors sampled for training
  std::size_t kmeans_iters = 20;   ///< PQ: Lloyd iterations per sub-space
  std::uint64_t seed = 1234;
};

class ScalarQuantizer {
 public:
  ScalarQuantizer() = default;

  /// Learns per-dimension [min, max] from `n` row-major vectors.
  [[nodiscard]] static ScalarQuantizer train(std::span<const float> data, std::size_t dim,
                                             Metric metric);

  [[nodiscard]] std::size_t dim() const noexcept { return vmin_.size(); }
  [[nodiscard]] std::size_t code_size() const noexcept { return vmin_.size(); }
  [[nodiscard]] Metric metric() const noexcept { return metric_; }

  void encode(const float* x, std::uint8_t* code) const noexcept;
  void decode(const std::uint8_t* code, float* x) const noexcept;

  /// Precomputed per-query state; distance() is called once per visited node.
  class Query {
   public:
    [[nodiscard]] float distance(const std::uint8_t* code) const noexcept {
      if (l2_) return bias_ + kernels_->sq8_l2(a_.data(), b_.data(), code, dim_);
      return 1.0F - (bias_ + kernels_->sq8_dot(a_.data(), code, dim_));
    }

   private:
    friend class ScalarQuantizer;
    AlignedBuffer<float> a_;
    AlignedBuffer<float> b_;
    float bias_ = 0.0F;
    std::size_t dim_ = 0;
    bool l2_ = true;
    const KernelTable* kernels_ = nullptr;
  };

  [[nodiscard]] Query prepare(const float* query, SimdBackend backend) const;

  void serialize(std::vector<std::byte>& out) const;
  [[nodiscard]] static ScalarQuantizer deserialize(std::span<const std::byte> in);

 private:
  Metric metric_ = Metric::L2;
  std::vector<float> vmin_;
  std::vector<float> step_;
};

class ProductQuantizer {
 public:
  static constexpr std::size_t kCentroids = 256;

  ProductQuantizer() = default;

  [[nodiscard]] static ProductQuantizer train(std::span<const float> data, std::size_t dim,
                                              Metric metric, const QuantizerConfig& cfg,
                                              std::size_t num_threads);

  [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
  [[nodiscard]] std::size_t code_size() const noexcept { return m_; }
  [[nodiscard]] std::size_t subspaces() const noexcept { return m_; }
  [[nodiscard]] Metric metric() const noexcept { return metric_; }

  void encode(const float* x, std::uint8_t* code) const noexcept;
  void decode(const std::uint8_t* code, float* x) const noexcept;

  class Query {
   public:
    [[nodiscard]] float distance(const std::uint8_t* code) const noexcept {
      const float* t = table_.data();
      float s0 = 0.0F;
      float s1 = 0.0F;
      float s2 = 0.0F;
      float s3 = 0.0F;
      std::size_t j = 0;
      for (; j + 4 <= m_; j += 4) {
        s0 += t[(j + 0) * kCentroids + code[j + 0]];
        s1 += t[(j + 1) * kCentroids + code[j + 1]];
        s2 += t[(j + 2) * kCentroids + code[j + 2]];
        s3 += t[(j + 3) * kCentroids + code[j + 3]];
      }
      for (; j < m_; ++j) s0 += t[j * kCentroids + code[j]];
      return offset_ + (s0 + s1) + (s2 + s3);
    }

   private:
    friend class ProductQuantizer;
    AlignedBuffer<float> table_;
    std::size_t m_ = 0;
    float offset_ = 0.0F;
  };

  [[nodiscard]] Query prepare(const float* query) const;

  void serialize(std::vector<std::byte>& out) const;
  [[nodiscard]] static ProductQuantizer deserialize(std::span<const std::byte> in);

 private:
  [[nodiscard]] const float* centroid(std::size_t sub, std::size_t c) const noexcept {
    return centroids_.data() + (sub * kCentroids + c) * dsub_;
  }

  Metric metric_ = Metric::L2;
  std::size_t dim_ = 0;
  std::size_t m_ = 0;
  std::size_t dsub_ = 0;
  std::vector<float> centroids_;  // m * 256 * dsub
};

}  // namespace vs
