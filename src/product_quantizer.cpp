#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

#include "vecsearch/detail/byte_stream.hpp"
#include "vecsearch/parallel.hpp"
#include "vecsearch/quantization.hpp"

namespace vs {

namespace {

inline float sq_dist(const float* a, const float* b, std::size_t d) noexcept {
  float acc = 0.0F;
  for (std::size_t i = 0; i < d; ++i) {
    const float t = a[i] - b[i];
    acc += t * t;
  }
  return acc;
}

inline std::size_t nearest(const float* x, const float* centroids, std::size_t k,
                           std::size_t d) noexcept {
  std::size_t best = 0;
  float best_d = std::numeric_limits<float>::max();
  for (std::size_t c = 0; c < k; ++c) {
    const float dist = sq_dist(x, centroids + c * d, d);
    if (dist < best_d) {
      best_d = dist;
      best = c;
    }
  }
  return best;
}

// Lloyd's k-means on n points of dimension d (row-major). Empty clusters are
// re-seeded by splitting the largest cluster, as in FAISS.
std::vector<float> kmeans(const float* x, std::size_t n, std::size_t d, std::size_t k,
                          std::size_t iters, std::mt19937_64& rng) {
  std::vector<float> centroids(k * d);
  std::vector<std::size_t> perm(n);
  std::iota(perm.begin(), perm.end(), std::size_t{0});
  std::shuffle(perm.begin(), perm.end(), rng);
  for (std::size_t c = 0; c < k; ++c) {
    std::memcpy(centroids.data() + c * d, x + perm[c % n] * d, d * sizeof(float));
  }

  std::vector<std::uint32_t> assign(n);
  std::vector<double> sums(k * d);
  std::vector<std::size_t> counts(k);
  for (std::size_t it = 0; it < iters; ++it) {
    for (std::size_t i = 0; i < n; ++i) {
      assign[i] = static_cast<std::uint32_t>(nearest(x + i * d, centroids.data(), k, d));
    }
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), std::size_t{0});
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t c = assign[i];
      ++counts[c];
      for (std::size_t j = 0; j < d; ++j) sums[c * d + j] += static_cast<double>(x[i * d + j]);
    }
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) continue;
      for (std::size_t j = 0; j < d; ++j) {
        centroids[c * d + j] = static_cast<float>(sums[c * d + j] / static_cast<double>(counts[c]));
      }
    }
    // Re-seed empty clusters near the centroid of the largest one.
    std::uniform_real_distribution<float> jitter(-1e-3F, 1e-3F);
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] != 0) continue;
      const auto big =
          static_cast<std::size_t>(std::max_element(counts.begin(), counts.end()) - counts.begin());
      for (std::size_t j = 0; j < d; ++j) {
        const float v = centroids[big * d + j];
        centroids[c * d + j] = v + jitter(rng) * (1.0F + std::abs(v));
      }
      counts[big] /= 2;
      counts[c] = counts[big];
    }
  }
  return centroids;
}

}  // namespace

ProductQuantizer ProductQuantizer::train(std::span<const float> data, std::size_t dim,
                                         Metric metric, const QuantizerConfig& cfg,
                                         std::size_t num_threads) {
  const std::size_t m = cfg.pq_subspaces;
  if (dim == 0 || m == 0 || dim % m != 0) {
    throw Error("ProductQuantizer: pq_subspaces must divide the dimension");
  }
  if (data.size() % dim != 0 || data.size() / dim < kCentroids) {
    throw Error("ProductQuantizer: need at least 256 training vectors");
  }
  ProductQuantizer pq;
  pq.metric_ = metric;
  pq.dim_ = dim;
  pq.m_ = m;
  pq.dsub_ = dim / m;
  pq.centroids_.resize(m * kCentroids * pq.dsub_);

  const std::size_t n = data.size() / dim;
  parallel_for(
      0, m, num_threads,
      [&](std::size_t sub) {
        // Gather the sub-vectors contiguously for cache-friendly k-means.
        std::vector<float> slice(n * pq.dsub_);
        for (std::size_t i = 0; i < n; ++i) {
          std::memcpy(slice.data() + i * pq.dsub_, data.data() + i * dim + sub * pq.dsub_,
                      pq.dsub_ * sizeof(float));
        }
        std::mt19937_64 rng(cfg.seed + sub);
        const std::vector<float> c =
            kmeans(slice.data(), n, pq.dsub_, kCentroids, cfg.kmeans_iters, rng);
        std::memcpy(pq.centroids_.data() + sub * kCentroids * pq.dsub_, c.data(),
                    c.size() * sizeof(float));
      },
      1);
  return pq;
}

void ProductQuantizer::encode(const float* x, std::uint8_t* code) const noexcept {
  for (std::size_t sub = 0; sub < m_; ++sub) {
    code[sub] =
        static_cast<std::uint8_t>(nearest(x + sub * dsub_, centroid(sub, 0), kCentroids, dsub_));
  }
}

void ProductQuantizer::decode(const std::uint8_t* code, float* x) const noexcept {
  for (std::size_t sub = 0; sub < m_; ++sub) {
    std::memcpy(x + sub * dsub_, centroid(sub, code[sub]), dsub_ * sizeof(float));
  }
}

ProductQuantizer::Query ProductQuantizer::prepare(const float* query) const {
  Query st;
  st.m_ = m_;
  st.table_ = AlignedBuffer<float>(m_ * kCentroids);
  const bool l2 = metric_ == Metric::L2;
  st.offset_ = l2 ? 0.0F : 1.0F;  // IP/cosine distance = 1 - sum_j <q_j, c_j>
  for (std::size_t sub = 0; sub < m_; ++sub) {
    const float* q = query + sub * dsub_;
    for (std::size_t c = 0; c < kCentroids; ++c) {
      const float* cen = centroid(sub, c);
      float v = 0.0F;
      if (l2) {
        v = sq_dist(q, cen, dsub_);
      } else {
        for (std::size_t j = 0; j < dsub_; ++j) v -= q[j] * cen[j];
      }
      st.table_[sub * kCentroids + c] = v;
    }
  }
  return st;
}

void ProductQuantizer::serialize(std::vector<std::byte>& out) const {
  detail::ByteWriter w;
  w.put(static_cast<std::uint32_t>(metric_));
  w.put(static_cast<std::uint64_t>(dim_));
  w.put(static_cast<std::uint64_t>(m_));
  w.put_array(std::span<const float>(centroids_));
  const auto& b = w.bytes();
  out.insert(out.end(), b.begin(), b.end());
}

ProductQuantizer ProductQuantizer::deserialize(std::span<const std::byte> in) {
  detail::ByteReader r(in);
  ProductQuantizer pq;
  const auto metric = r.get<std::uint32_t>();
  if (metric > 2) throw FormatError("PQ: bad metric");
  pq.metric_ = static_cast<Metric>(metric);
  pq.dim_ = r.get<std::uint64_t>();
  pq.m_ = r.get<std::uint64_t>();
  if (pq.dim_ == 0 || pq.m_ == 0 || pq.dim_ % pq.m_ != 0 || pq.dim_ > (1U << 20)) {
    throw FormatError("PQ: bad geometry");
  }
  pq.dsub_ = pq.dim_ / pq.m_;
  pq.centroids_.resize(pq.m_ * kCentroids * pq.dsub_);
  r.get_array(std::span<float>(pq.centroids_));
  return pq;
}

}  // namespace vs
