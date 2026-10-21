// Hierarchical Navigable Small World (HNSW) index.
//
// Reference: Y. Malkov, D. Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs", TPAMI 2018.
//
// Thread-safety contract
// ----------------------
//  * add(), add_batch(), remove(), contains(), get_vector() and search() may be
//    called concurrently from any number of threads.
//  * Searches never take locks: neighbour lists are read with acquire loads
//    and written (under a per-node spin lock) with release stores, so a reader
//    always observes a list of valid, fully-initialised node ids.
//  * reserve(), train_quantizer(), save() and the move operations require that
//    no other operation runs concurrently (the HTTP server enforces this with
//    a reader/writer gate). compact() only reads, and may run alongside
//    searches.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/function_ref.hpp"
#include "vecsearch/quantization.hpp"
#include "vecsearch/types.hpp"

namespace vs {

struct HnswConfig {
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t M = 16;                 ///< max out-degree on layers >= 1 (layer 0 uses 2*M)
  std::size_t ef_construction = 200;  ///< beam width while inserting
  std::size_t capacity = 1024;        ///< initial number of slots (grow with reserve())
  std::uint64_t seed = 0x5eed;        ///< seeds the level generator
  bool keep_pruned = false;           ///< paper's keepPrunedConnections option
  SimdBackend simd = compiled_backend();
};

inline constexpr std::size_t kRerankAll = static_cast<std::size_t>(-1);

/// Optional counters filled in by search() for analysis and benchmarks.
struct SearchStats {
  std::size_t distance_computations = 0;
  std::size_t hops = 0;     ///< nodes expanded (neighbour lists scanned)
  bool exact_scan = false;  ///< the planner chose an exact filtered scan
};

struct SearchParams {
  std::size_t k = 10;
  std::size_t ef = 64;  ///< beam width on layer 0; effectively max(ef, k)
  TagFilter filter{};   ///< metadata predicate evaluated during traversal
  /// Arbitrary label predicate (non-owning; keep the callable alive).
  FunctionRef<bool(label_t)> predicate{};
  /// Traverse using the quantized codes (requires train_quantizer()).
  bool use_quantized = false;
  /// With use_quantized: re-score this many of the best approximate candidates
  /// with full-precision vectors (clamped to [k, ef]). 0 disables re-ranking
  /// and returns approximate distances.
  std::size_t rerank = kRerankAll;
  /// Let the planner answer highly selective tag filters with an exact scan
  /// when its cost model predicts that to be cheaper than graph traversal.
  bool allow_exact_scan = true;
  SearchStats* stats = nullptr;
};

/// Byte-level memory accounting of the index (excludes allocator overhead).
struct MemoryUsage {
  std::size_t vectors = 0;   ///< full-precision vectors
  std::size_t codes = 0;     ///< quantized codes + codebooks
  std::size_t graph = 0;     ///< level-0 and upper-level adjacency lists
  std::size_t metadata = 0;  ///< labels, tags, levels, flags, locks, label map
  std::size_t mapped = 0;    ///< portion of the above backed by an mmap'd file

  [[nodiscard]] std::size_t total() const noexcept { return vectors + codes + graph + metadata; }
};

/// Result of check_integrity(): structural invariants of the graph.
struct IntegrityReport {
  bool ok = true;
  std::size_t nodes = 0;
  std::size_t reachable_level0 = 0;  ///< nodes reachable from the entry point on layer 0
  std::size_t max_level = 0;
  double avg_degree_level0 = 0.0;
  std::vector<std::string> errors;
};

enum class LoadMode {
  Copy,  ///< read everything into owned memory; the index stays mutable
  Mmap,  ///< map vectors/graph/codes zero-copy; inserts are rejected
};

class HnswIndex {
 public:
  explicit HnswIndex(const HnswConfig& config);
  ~HnswIndex();
  HnswIndex(HnswIndex&&) noexcept;
  HnswIndex& operator=(HnswIndex&&) noexcept;
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  /// Inserts or replaces (upsert) the vector for `label`. Replacing tombstones
  /// the old node and inserts a fresh one. Throws CapacityError when full.
  void add(label_t label, std::span<const float> vector, std::uint64_t tags = 0);

  /// Inserts `labels.size()` row-major vectors using `num_threads` threads
  /// (0 = all cores). `tags` is empty or one mask per label.
  void add_batch(std::span<const float> vectors, std::span<const label_t> labels,
                 std::size_t num_threads = 0, std::span<const std::uint64_t> tags = {});

  /// Tombstones `label`. Returns false if it was not present. The node stays
  /// in the graph (to preserve connectivity) until compact().
  bool remove(label_t label);

  [[nodiscard]] bool contains(label_t label) const;
  /// Returns the stored vector (normalised for cosine). Throws if absent.
  [[nodiscard]] std::vector<float> get_vector(label_t label) const;

  /// k approximate nearest neighbours, ascending by distance.
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 const SearchParams& params) const;

  /// Batch search over row-major queries, parallelised across queries.
  [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(std::span<const float> queries,
                                                                    const SearchParams& params,
                                                                    std::size_t num_threads) const;

  /// Grows the slot capacity. Requires exclusive access.
  void reserve(std::size_t new_capacity);

  /// Trains a quantizer on a sample of the stored vectors and encodes all of
  /// them; later inserts are encoded automatically. Requires exclusive access.
  void train_quantizer(const QuantizerConfig& config, std::size_t num_threads = 0);
  [[nodiscard]] QuantizerType quantizer_type() const noexcept;

  /// Builds a fresh index containing only live (non-deleted) vectors.
  [[nodiscard]] HnswIndex compact(std::size_t num_threads = 0) const;

  /// Writes the index in the versioned binary format (atomic temp + rename).
  void save(const std::filesystem::path& path) const;
  /// `simd` selects the distance kernels for the loaded index (the file
  /// format is independent of the kernel backend).
  [[nodiscard]] static HnswIndex load(const std::filesystem::path& path,
                                      LoadMode mode = LoadMode::Copy,
                                      SimdBackend simd = compiled_backend());

  [[nodiscard]] std::size_t size() const noexcept;           ///< live vectors
  [[nodiscard]] std::size_t node_count() const noexcept;     ///< nodes incl. tombstones
  [[nodiscard]] std::size_t deleted_count() const noexcept;  ///< tombstoned nodes
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t dim() const noexcept;
  [[nodiscard]] Metric metric() const noexcept;
  [[nodiscard]] const HnswConfig& config() const noexcept;
  [[nodiscard]] bool read_only() const noexcept;
  [[nodiscard]] MemoryUsage memory_usage() const;

  /// Verifies graph invariants (degree bounds, id ranges, no self-loops or
  /// duplicates, level consistency, label map consistency, reachability).
  /// Requires that no writes run concurrently.
  [[nodiscard]] IntegrityReport check_integrity() const;

  struct Impl;

 private:
  explicit HnswIndex(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vs
