// HTTP/JSON serving layer.
//
// VectorService is a thread-safe facade over HnswIndex that adds what a
// long-running server needs on top of the core index: automatic capacity
// growth, online compaction that swaps in a rebuilt index, and snapshots.
// HttpServer maps REST endpoints onto it (see README "HTTP API").
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "vecsearch/hnsw_index.hpp"

namespace httplib {
class Server;
}

namespace vs::server {

struct UpsertItem {
  label_t id = 0;
  std::vector<float> values;
  std::uint64_t tags = 0;
};

struct ServiceStats {
  std::size_t size = 0;
  std::size_t deleted = 0;
  std::size_t capacity = 0;
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t M = 0;
  std::size_t ef_construction = 0;
  QuantizerType quantizer = QuantizerType::None;
  bool read_only = false;
  MemoryUsage memory;
};

class VectorService {
 public:
  /// Fresh, empty index.
  VectorService(const HnswConfig& config, std::filesystem::path snapshot_path = {});
  /// Wraps an existing (e.g. loaded) index.
  explicit VectorService(HnswIndex index, std::filesystem::path snapshot_path = {});

  /// Upserts a batch; grows capacity as needed. Returns the number upserted.
  std::size_t upsert(const std::vector<UpsertItem>& items);
  bool remove(label_t id);
  [[nodiscard]] std::optional<std::vector<float>> get(label_t id) const;
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 const SearchParams& params) const;
  [[nodiscard]] ServiceStats stats() const;

  struct CompactResult {
    std::size_t nodes_before = 0;
    std::size_t nodes_after = 0;
    double seconds = 0.0;
  };
  /// Rebuilds the index without tombstones. Searches keep running against the
  /// old index during the rebuild; writes are held until the swap.
  CompactResult compact();

  struct SnapshotResult {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    double seconds = 0.0;
  };
  /// Saves to the configured snapshot path. Throws if none is configured.
  SnapshotResult snapshot();

  [[nodiscard]] std::size_t dim() const;

 private:
  // Lock hierarchy: write_gate_ (writers shared / maintenance exclusive)
  // before index_mutex_ (everyone shared / index swap+grow exclusive).
  mutable std::shared_mutex write_gate_;
  mutable std::shared_mutex index_mutex_;
  std::unique_ptr<HnswIndex> index_;
  std::filesystem::path snapshot_path_;
};

/// REST front-end. Endpoints:
///   GET    /health
///   GET    /stats
///   POST   /vectors            {"vectors":[{"id":1,"values":[...],"tags":0}]}
///   GET    /vectors/{id}
///   DELETE /vectors/{id}
///   POST   /search             {"vector":[...],"k":10,"ef":64,"filter":{...}}
///   POST   /admin/compact
///   POST   /admin/snapshot
class HttpServer {
 public:
  explicit HttpServer(VectorService& service);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  /// Binds to host:port (port 0 picks an ephemeral port). Returns the bound
  /// port. Throws on failure.
  int bind(const std::string& host, int port);
  /// Serves until stop() is called. Must be preceded by bind().
  void serve();
  void stop();

 private:
  void register_routes();

  VectorService& service_;
  std::unique_ptr<httplib::Server> server_;
};

}  // namespace vs::server
