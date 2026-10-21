// vsearch-server: serve an HNSW index over HTTP/JSON.
//
//   vsearch-server --dim 128 --metric l2 --port 8080
//   vsearch-server --index sift.vsi --mmap --port 8080
#include <csignal>
#include <filesystem>
#include <iostream>

#include "args.hpp"
#include "server/http_server.hpp"

namespace {

vs::server::HttpServer* g_server = nullptr;

void on_signal(int /*sig*/) {
  if (g_server != nullptr) g_server->stop();
}

void usage() {
  std::cerr << R"(usage: vsearch-server [options]
  --index PATH        load this index at startup (if it exists) and use it as
                      the snapshot target for POST /admin/snapshot
  --mmap              load the index memory-mapped (becomes writable on first upsert)
  --dim N             dimension for a new, empty index (required without --index)
  --metric l2|ip|cosine
  --M N --efc N       graph parameters for a new index (default 16 / 200)
  --capacity N        initial capacity of a new index (default 100000)
  --host ADDR         bind address (default 127.0.0.1)
  --port N            port (default 8080; 0 = pick a free port)
)";
}

}  // namespace

int main(int argc, char** argv) try {
  const vs::apps::Args args(argc, argv, 1, {"mmap", "help"});
  if (args.has("help")) {
    usage();
    return 0;
  }
  const std::filesystem::path index_path = args.str("index");
  std::unique_ptr<vs::server::VectorService> service;
  if (!index_path.empty() && std::filesystem::exists(index_path)) {
    const auto mode = args.has("mmap") ? vs::LoadMode::Mmap : vs::LoadMode::Copy;
    auto index = vs::HnswIndex::load(index_path, mode);
    std::cerr << "loaded " << index.size() << " vectors (dim " << index.dim() << ") from "
              << index_path << (args.has("mmap") ? " [mmap]" : "") << '\n';
    service = std::make_unique<vs::server::VectorService>(std::move(index), index_path);
  } else {
    if (!args.has("dim")) {
      usage();
      return 2;
    }
    vs::HnswConfig cfg;
    cfg.dim = args.size("dim", 0);
    cfg.metric = vs::parse_metric(args.str("metric", "l2"));
    cfg.M = args.size("M", 16);
    cfg.ef_construction = args.size("efc", 200);
    cfg.capacity = args.size("capacity", 100'000);
    service = std::make_unique<vs::server::VectorService>(cfg, index_path);
  }

  vs::server::HttpServer server(*service);
  const std::string host = args.str("host", "127.0.0.1");
  const int port = server.bind(host, static_cast<int>(args.size("port", 8080)));
  g_server = &server;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::cerr << "listening on http://" << host << ":" << port << '\n';
  server.serve();
  g_server = nullptr;
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
