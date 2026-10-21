#include "server/http_server.hpp"

#include <httplib.h>

#include <charconv>
#include <nlohmann/json.hpp>

#include "vecsearch/system.hpp"

namespace vs::server {

namespace {

using json = nlohmann::json;

/// Client errors map to HTTP 400.
class BadRequest : public Error {
 public:
  using Error::Error;
};

std::vector<float> parse_vector(const json& j, std::size_t dim, const char* field) {
  if (!j.is_array()) throw BadRequest(std::string("'") + field + "' must be an array of numbers");
  if (j.size() != dim) {
    throw BadRequest(std::string("'") + field + "' has " + std::to_string(j.size()) +
                     " elements, index dimension is " + std::to_string(dim));
  }
  std::vector<float> v;
  v.reserve(dim);
  for (const auto& x : j) {
    if (!x.is_number()) throw BadRequest(std::string("'") + field + "' must contain only numbers");
    v.push_back(x.get<float>());
  }
  return v;
}

label_t parse_id(const json& j) {
  if (!j.is_number_unsigned() && !(j.is_number_integer() && j.get<std::int64_t>() >= 0)) {
    throw BadRequest("'id' must be a non-negative integer");
  }
  return j.get<label_t>();
}

std::uint64_t get_u64(const json& obj, const char* key, std::uint64_t fallback) {
  if (!obj.contains(key)) return fallback;
  const json& v = obj.at(key);
  if (!v.is_number_unsigned() && !(v.is_number_integer() && v.get<std::int64_t>() >= 0)) {
    throw BadRequest(std::string("'") + key + "' must be a non-negative integer");
  }
  return v.get<std::uint64_t>();
}

json memory_json(const MemoryUsage& m) {
  return {{"vectors_bytes", m.vectors},   {"graph_bytes", m.graph},   {"codes_bytes", m.codes},
          {"metadata_bytes", m.metadata}, {"total_bytes", m.total()}, {"mapped_bytes", m.mapped}};
}

std::string_view quantizer_name(QuantizerType t) {
  switch (t) {
    case QuantizerType::SQ8: return "sq8";
    case QuantizerType::PQ: return "pq";
    case QuantizerType::None: break;
  }
  return "none";
}

void reply(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

/// Runs a handler, translating exceptions into JSON error responses.
template <class Fn>
void guarded(httplib::Response& res, Fn&& fn) {
  try {
    fn();
  } catch (const BadRequest& e) {
    reply(res, 400, {{"error", e.what()}});
  } catch (const json::exception& e) {
    reply(res, 400, {{"error", std::string("invalid JSON: ") + e.what()}});
  } catch (const Error& e) {
    reply(res, 422, {{"error", e.what()}});
  } catch (const std::exception& e) {
    reply(res, 500, {{"error", e.what()}});
  }
}

std::optional<label_t> path_id(const httplib::Request& req) {
  const std::string& s = req.path_params.at("id");
  label_t id = 0;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), id);
  if (ec != std::errc{} || p != s.data() + s.size()) return std::nullopt;
  return id;
}

}  // namespace

// ============================================================================
// VectorService
// ============================================================================

VectorService::VectorService(const HnswConfig& config, std::filesystem::path snapshot_path)
    : index_(std::make_unique<HnswIndex>(config)), snapshot_path_(std::move(snapshot_path)) {}

VectorService::VectorService(HnswIndex index, std::filesystem::path snapshot_path)
    : index_(std::make_unique<HnswIndex>(std::move(index))),
      snapshot_path_(std::move(snapshot_path)) {}

std::size_t VectorService::dim() const {
  const std::shared_lock lock(index_mutex_);
  return index_->dim();
}

std::size_t VectorService::upsert(const std::vector<UpsertItem>& items) {
  const std::shared_lock gate(write_gate_);
  // Grow ahead of time (each upsert consumes a fresh slot). Growth needs
  // exclusive access, so the exclusive lock is only taken when required.
  auto needs_growth = [&] {
    return index_->node_count() + items.size() > index_->capacity() || index_->read_only();
  };
  bool grow = false;
  {
    const std::shared_lock lock(index_mutex_);
    grow = needs_growth();
  }
  if (grow) {
    const std::unique_lock lock(index_mutex_);
    if (needs_growth()) {
      index_->reserve(std::max(index_->node_count() + items.size(), index_->capacity() * 2));
    }
  }
  std::size_t done = 0;
  for (const UpsertItem& item : items) {
    for (;;) {
      {
        const std::shared_lock lock(index_mutex_);
        try {
          index_->add(item.id, item.values, item.tags);
          ++done;
          break;
        } catch (const CapacityError&) {
          // A concurrent batch consumed our headroom; grow and retry below.
        }
      }
      const std::unique_lock lock(index_mutex_);
      index_->reserve(index_->capacity() * 2);
    }
  }
  return done;
}

bool VectorService::remove(label_t id) {
  const std::shared_lock gate(write_gate_);
  const std::shared_lock lock(index_mutex_);
  return index_->remove(id);
}

std::optional<std::vector<float>> VectorService::get(label_t id) const {
  const std::shared_lock lock(index_mutex_);
  if (!index_->contains(id)) return std::nullopt;
  try {
    return index_->get_vector(id);
  } catch (const Error&) {
    return std::nullopt;  // deleted between the two calls
  }
}

std::vector<SearchResult> VectorService::search(std::span<const float> query,
                                                const SearchParams& params) const {
  const std::shared_lock lock(index_mutex_);
  return index_->search(query, params);
}

ServiceStats VectorService::stats() const {
  const std::shared_lock lock(index_mutex_);
  const HnswConfig& c = index_->config();
  return {index_->size(),
          index_->deleted_count(),
          index_->capacity(),
          index_->dim(),
          c.metric,
          c.M,
          c.ef_construction,
          index_->quantizer_type(),
          index_->read_only(),
          index_->memory_usage()};
}

VectorService::CompactResult VectorService::compact() {
  const Stopwatch sw;
  const std::unique_lock gate(write_gate_);  // hold writers; readers continue
  std::unique_ptr<HnswIndex> fresh;
  CompactResult r;
  {
    const std::shared_lock lock(index_mutex_);
    r.nodes_before = index_->node_count();
    fresh = std::make_unique<HnswIndex>(index_->compact());
  }
  {
    const std::unique_lock lock(index_mutex_);
    index_ = std::move(fresh);
    r.nodes_after = index_->node_count();
  }
  r.seconds = sw.seconds();
  return r;
}

VectorService::SnapshotResult VectorService::snapshot() {
  if (snapshot_path_.empty()) throw Error("no snapshot path configured (start with --index)");
  const Stopwatch sw;
  const std::unique_lock gate(write_gate_);
  const std::shared_lock lock(index_mutex_);
  index_->save(snapshot_path_);
  return {snapshot_path_, std::filesystem::file_size(snapshot_path_), sw.seconds()};
}

// ============================================================================
// HttpServer
// ============================================================================

HttpServer::HttpServer(VectorService& service)
    : service_(service), server_(std::make_unique<httplib::Server>()) {
  register_routes();
}

HttpServer::~HttpServer() { stop(); }

int HttpServer::bind(const std::string& host, int port) {
  const int bound =
      port == 0 ? server_->bind_to_any_port(host) : (server_->bind_to_port(host, port) ? port : -1);
  if (bound <= 0) throw Error("cannot bind " + host + ":" + std::to_string(port));
  return bound;
}

void HttpServer::serve() {
  if (!server_->listen_after_bind()) throw Error("HTTP server failed");
}

void HttpServer::stop() {
  if (server_) server_->stop();
}

void HttpServer::register_routes() {
  httplib::Server& s = *server_;
  s.set_payload_max_length(256U << 20);

  s.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    reply(res, 200, {{"status", "ok"}});
  });

  s.Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
    guarded(res, [&] {
      const ServiceStats st = service_.stats();
      reply(res, 200,
            {{"size", st.size},
             {"deleted", st.deleted},
             {"capacity", st.capacity},
             {"dim", st.dim},
             {"metric", std::string(to_string(st.metric))},
             {"M", st.M},
             {"ef_construction", st.ef_construction},
             {"quantizer", std::string(quantizer_name(st.quantizer))},
             {"read_only", st.read_only},
             {"simd", std::string(to_string(compiled_backend()))},
             {"memory", memory_json(st.memory)}});
    });
  });

  s.Post("/vectors", [this](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      const json body = json::parse(req.body);
      const std::size_t dim = service_.dim();
      std::vector<UpsertItem> items;
      auto parse_item = [&](const json& j) {
        if (!j.is_object() || !j.contains("id") || !j.contains("values")) {
          throw BadRequest("each vector needs 'id' and 'values'");
        }
        items.push_back({parse_id(j.at("id")), parse_vector(j.at("values"), dim, "values"),
                         get_u64(j, "tags", 0)});
      };
      if (body.contains("vectors")) {
        if (!body.at("vectors").is_array()) throw BadRequest("'vectors' must be an array");
        for (const auto& j : body.at("vectors")) parse_item(j);
      } else {
        parse_item(body);
      }
      reply(res, 200, {{"upserted", service_.upsert(items)}});
    });
  });

  s.Get("/vectors/:id", [this](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      const auto id = path_id(req);
      if (!id) throw BadRequest("id must be a non-negative integer");
      const auto v = service_.get(*id);
      if (!v) {
        reply(res, 404, {{"error", "not found"}});
        return;
      }
      reply(res, 200, {{"id", *id}, {"values", *v}});
    });
  });

  s.Delete("/vectors/:id", [this](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      const auto id = path_id(req);
      if (!id) throw BadRequest("id must be a non-negative integer");
      const bool deleted = service_.remove(*id);
      reply(res, deleted ? 200 : 404, {{"deleted", deleted}});
    });
  });

  s.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      const Stopwatch sw;
      const json body = json::parse(req.body);
      if (!body.contains("vector")) throw BadRequest("missing 'vector'");
      const std::vector<float> q = parse_vector(body.at("vector"), service_.dim(), "vector");
      SearchParams p;
      p.k = get_u64(body, "k", 10);
      p.ef = get_u64(body, "ef", 64);
      if (p.k == 0 || p.k > 10'000) throw BadRequest("'k' must be in [1, 10000]");
      if (p.ef > 100'000) throw BadRequest("'ef' must be <= 100000");
      if (body.contains("filter")) {
        const json& f = body.at("filter");
        if (!f.is_object()) throw BadRequest("'filter' must be an object");
        p.filter = {get_u64(f, "any_of", 0), get_u64(f, "all_of", 0), get_u64(f, "none_of", 0)};
      }
      if (body.contains("quantized")) p.use_quantized = body.at("quantized").get<bool>();
      if (body.contains("rerank")) p.rerank = get_u64(body, "rerank", kRerankAll);
      const auto results = service_.search(q, p);
      json arr = json::array();
      for (const auto& r : results) arr.push_back({{"id", r.label}, {"distance", r.distance}});
      reply(res, 200, {{"results", arr}, {"took_us", sw.micros()}});
    });
  });

  s.Post("/admin/compact", [this](const httplib::Request&, httplib::Response& res) {
    guarded(res, [&] {
      const auto r = service_.compact();
      reply(res, 200,
            {{"nodes_before", r.nodes_before},
             {"nodes_after", r.nodes_after},
             {"seconds", r.seconds}});
    });
  });

  s.Post("/admin/snapshot", [this](const httplib::Request&, httplib::Response& res) {
    guarded(res, [&] {
      const auto r = service_.snapshot();
      reply(res, 200, {{"path", r.path.string()}, {"bytes", r.bytes}, {"seconds", r.seconds}});
    });
  });
}

}  // namespace vs::server
