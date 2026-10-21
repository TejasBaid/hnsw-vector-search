// Private state of HnswIndex, shared by the algorithm (hnsw_index.cpp) and
// persistence (hnsw_persistence.cpp) translation units.
//
// Memory layout
// -------------
//  vectors  : capacity x dim floats, contiguous, 64-byte aligned.
//  level0   : capacity x (1 + M0) uint32: [count, n_0 .. n_{M0-1}] per node.
//             Fixed-size blocks keep layer 0 (where >90% of the search time is
//             spent) in one flat array with no pointer chasing.
//  upper[i] : levels[i] x (1 + M) uint32, allocated only for nodes with level>0
//             (a fraction 1/M of nodes), same [count, ids...] block format.
//  Both vectors/level0 (and codes) may point into an mmap'd file.
#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "vecsearch/aligned_buffer.hpp"
#include "vecsearch/detail/atomics.hpp"
#include "vecsearch/detail/spin_lock.hpp"
#include "vecsearch/detail/visited_pool.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/mmap_file.hpp"
#include "vecsearch/quantization.hpp"

namespace vs {

inline constexpr std::uint64_t kNoEntry = ~std::uint64_t{0};
inline constexpr int kMaxLevel = 31;

[[nodiscard]] constexpr std::uint64_t pack_entry(node_id id, int level) noexcept {
  return (static_cast<std::uint64_t>(level) << 32) | id;
}
[[nodiscard]] constexpr node_id entry_id(std::uint64_t packed) noexcept {
  return static_cast<node_id>(packed & 0xffffffffU);
}
[[nodiscard]] constexpr int entry_level(std::uint64_t packed) noexcept {
  return static_cast<int>(packed >> 32);
}

struct HnswIndex::Impl {
  explicit Impl(const HnswConfig& config);

  // ---- configuration -----------------------------------------------------
  HnswConfig cfg;
  std::size_t dim = 0;
  std::size_t M = 0;
  std::size_t M0 = 0;
  double level_mult = 0.0;
  DistanceFn dist = nullptr;
  std::size_t capacity = 0;
  bool read_only = false;

  // ---- storage (owned, or pointing into `mapping`) -------------------------
  AlignedBuffer<float> vectors_owned;
  float* vectors = nullptr;
  AlignedBuffer<std::uint32_t> level0_owned;
  std::uint32_t* level0 = nullptr;
  std::vector<std::unique_ptr<std::uint32_t[]>> upper;
  AlignedBuffer<std::uint8_t> levels;
  AlignedBuffer<label_t> labels;
  AlignedBuffer<std::uint64_t> tags;
  AlignedBuffer<std::uint8_t> state;  // kStateDeleted | kStateReady, accessed atomically
  std::unique_ptr<detail::SpinLock[]> locks;
  MappedFile mapping;

  // ---- quantization --------------------------------------------------------
  QuantizerType qtype = QuantizerType::None;
  ScalarQuantizer sq;
  ProductQuantizer pq;
  std::size_t code_size = 0;
  AlignedBuffer<std::uint8_t> codes_owned;
  std::uint8_t* codes = nullptr;

  // ---- concurrency state -------------------------------------------------
  std::atomic<std::uint32_t> count{0};         // allocated node ids
  std::atomic<std::uint64_t> entry{kNoEntry};  // packed (level, id)
  std::mutex entry_mutex;                      // serialises max-level growth
  mutable std::mutex label_mutex;              // guards label_map
  std::unordered_map<label_t, node_id> label_map;
  std::atomic<std::size_t> num_deleted{0};
  std::array<std::atomic<std::int64_t>, 64> tag_counts{};
  std::unique_ptr<detail::VisitedPool> visited;

  // ---- accessors -----------------------------------------------------------
  [[nodiscard]] std::size_t level0_stride() const noexcept { return 1 + M0; }
  [[nodiscard]] std::size_t upper_stride() const noexcept { return 1 + M; }
  [[nodiscard]] std::size_t max_links(int level) const noexcept { return level == 0 ? M0 : M; }

  [[nodiscard]] std::uint32_t* links(node_id id, int level) const noexcept {
    if (level == 0) return level0 + static_cast<std::size_t>(id) * level0_stride();
    return upper[id].get() + static_cast<std::size_t>(level - 1) * upper_stride();
  }
  [[nodiscard]] const float* vec(node_id id) const noexcept {
    return vectors + static_cast<std::size_t>(id) * dim;
  }
  [[nodiscard]] float* vec(node_id id) noexcept {
    return vectors + static_cast<std::size_t>(id) * dim;
  }
  [[nodiscard]] const std::uint8_t* code(node_id id) const noexcept {
    return codes + static_cast<std::size_t>(id) * code_size;
  }
  [[nodiscard]] int level_of(node_id id) const noexcept { return levels[id]; }

  static constexpr std::uint8_t kStateDeleted = 1;  // tombstone (persisted)
  static constexpr std::uint8_t kStateReady = 2;    // fully initialised (runtime only)

  [[nodiscard]] std::uint8_t load_state(node_id id) const noexcept {
    return detail::load_acquire(state[id]);
  }
  [[nodiscard]] bool is_deleted(node_id id) const noexcept {
    return (load_state(id) & kStateDeleted) != 0;
  }
  void set_state_bits(node_id id, std::uint8_t bits) noexcept {
    (void)detail::fetch_or_acq_rel(state[id], bits);
  }
  void set_deleted(node_id id) noexcept { set_state_bits(id, kStateDeleted); }

  /// Live-node count per tag bit, for filter selectivity estimation.
  void count_tags(std::uint64_t t, std::int64_t delta) noexcept {
    for (; t != 0; t &= t - 1) {
      tag_counts[static_cast<std::size_t>(std::countr_zero(t))].fetch_add(
          delta, std::memory_order_relaxed);
    }
  }

  static std::uint32_t load_acquire(const std::uint32_t& x) noexcept {
    return detail::load_acquire(x);
  }
  static std::uint32_t load_relaxed(const std::uint32_t& x) noexcept {
    return detail::load_relaxed(x);
  }
  static void store_release(std::uint32_t& x, std::uint32_t v) noexcept {
    detail::store_release(x, v);
  }

  /// Allocates owned storage for `cap` slots (fresh index).
  void allocate(std::size_t cap);
  /// Moves mapped arrays into owned memory with capacity `cap` (>= count).
  void grow_owned(std::size_t cap);
  void allocate_upper(node_id id, int level);
  void encode_node(node_id id) noexcept;
};

}  // namespace vs
