// Shared helpers for the test suite: synthetic data and recall computation.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vecsearch/flat_index.hpp"
#include "vecsearch/types.hpp"

namespace vs::test {

/// Gaussian-mixture data: `clusters` centres in [-1,1]^dim with isotropic
/// noise. Clustered data is the hard case for graph indexes (and the case the
/// neighbour heuristic exists for), so tests use it rather than uniform noise.
inline std::vector<float> clustered_data(std::size_t n, std::size_t dim, std::uint64_t seed,
                                         std::size_t clusters = 32, float sigma = 0.15F) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
  std::normal_distribution<float> noise(0.0F, sigma);
  std::vector<float> centres(clusters * dim);
  for (float& c : centres) c = uni(rng);
  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
  std::vector<float> data(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pick(rng);
    for (std::size_t j = 0; j < dim; ++j) data[i * dim + j] = centres[c * dim + j] + noise(rng);
  }
  return data;
}

/// Base vectors and queries drawn from the *same* mixture (queries are the
/// last `nq` samples), i.e. in-distribution queries as in real workloads.
struct Split {
  std::vector<float> base;
  std::vector<float> queries;
};
inline Split clustered_split(std::size_t n, std::size_t nq, std::size_t dim, std::uint64_t seed,
                             bool unit_norm = false) {
  std::vector<float> all = clustered_data(n + nq, dim, seed);
  if (unit_norm) {
    for (std::size_t i = 0; i < n + nq; ++i) {
      double norm = 0.0;
      for (std::size_t j = 0; j < dim; ++j)
        norm += static_cast<double>(all[i * dim + j]) * static_cast<double>(all[i * dim + j]);
      const auto inv = static_cast<float>(1.0 / std::sqrt(norm));
      for (std::size_t j = 0; j < dim; ++j) all[i * dim + j] *= inv;
    }
  }
  Split s;
  s.base.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(n * dim));
  s.queries.assign(all.begin() + static_cast<std::ptrdiff_t>(n * dim), all.end());
  return s;
}

inline std::vector<label_t> iota_labels(std::size_t n, label_t start = 0) {
  std::vector<label_t> labels(n);
  for (std::size_t i = 0; i < n; ++i) labels[i] = start + i;
  return labels;
}

/// Owns a set of threads and joins them on join() or destruction (a portable
/// stand-in for std::vector<std::jthread>).
class ThreadGroup {
 public:
  ThreadGroup() = default;
  ThreadGroup(const ThreadGroup&) = delete;
  ThreadGroup& operator=(const ThreadGroup&) = delete;
  ~ThreadGroup() { join(); }
  template <class F>
  void spawn(F&& f) {
    threads_.emplace_back(std::forward<F>(f));
  }
  void join() {
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
    threads_.clear();
  }

 private:
  std::vector<std::thread> threads_;
};

/// recall@k = |approx ∩ exact| / |exact|, averaged over queries.
inline double recall_at_k(const std::vector<std::vector<SearchResult>>& approx,
                          const std::vector<std::vector<SearchResult>>& exact, std::size_t k) {
  double hits = 0.0;
  double total = 0.0;
  for (std::size_t q = 0; q < exact.size(); ++q) {
    std::unordered_set<label_t> truth;
    for (std::size_t i = 0; i < std::min(k, exact[q].size()); ++i) truth.insert(exact[q][i].label);
    for (std::size_t i = 0; i < std::min(k, approx[q].size()); ++i) {
      hits += truth.contains(approx[q][i].label) ? 1.0 : 0.0;
    }
    total += static_cast<double>(truth.size());
  }
  return total == 0.0 ? 1.0 : hits / total;
}

}  // namespace vs::test
