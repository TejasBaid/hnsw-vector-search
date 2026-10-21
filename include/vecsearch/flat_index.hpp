// Exact brute-force index. Used as ground truth for recall measurement and as
// the right tool for small collections or very selective filters.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vecsearch/aligned_buffer.hpp"
#include "vecsearch/distance.hpp"
#include "vecsearch/function_ref.hpp"
#include "vecsearch/types.hpp"

namespace vs {

class FlatIndex {
 public:
  FlatIndex(std::size_t dim, Metric metric, SimdBackend backend = compiled_backend());

  /// Appends `labels.size()` row-major vectors. Not thread-safe.
  void add(std::span<const float> vectors, std::span<const label_t> labels,
           std::span<const std::uint64_t> tags = {});

  /// Exact k nearest neighbours, ascending by distance.
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                 const TagFilter& filter = {},
                                                 FunctionRef<bool(label_t)> predicate = {}) const;

  /// Exact k-NN for a batch of row-major queries, parallelised over queries.
  [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(
      std::span<const float> queries, std::size_t k, std::size_t num_threads,
      const TagFilter& filter = {}) const;

  [[nodiscard]] std::size_t size() const noexcept { return labels_.size(); }
  [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
  [[nodiscard]] Metric metric() const noexcept { return metric_; }

 private:
  std::size_t dim_;
  Metric metric_;
  DistanceFn dist_;
  std::vector<float> vectors_;
  std::vector<label_t> labels_;
  std::vector<std::uint64_t> tags_;
};

}  // namespace vs
