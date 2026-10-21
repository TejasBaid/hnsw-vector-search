// End-to-end tests of the HTTP API. The server binds to port 0 (an ephemeral
// port chosen by the OS) so tests never collide with other processes.
#include <gtest/gtest.h>
#include <httplib.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <thread>

#include "server/http_server.hpp"
#include "test_util.hpp"

namespace vs::server {
namespace {

using json = nlohmann::json;

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    snapshot_ = std::filesystem::temp_directory_path() /
                ("vs_server_test_" + std::to_string(::getpid()) + ".vsi");
    service_ = std::make_unique<VectorService>(
        HnswConfig{.dim = 4, .metric = Metric::L2, .M = 8, .ef_construction = 32, .capacity = 2},
        snapshot_);
    server_ = std::make_unique<HttpServer>(*service_);
    port_ = server_->bind("127.0.0.1", 0);
    thread_ = std::thread([this] { server_->serve(); });
    client_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
    client_->set_connection_timeout(5);
    // Wait until the listener accepts connections.
    for (int i = 0; i < 100 && !client_->Get("/health"); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  void TearDown() override {
    server_->stop();
    if (thread_.joinable()) thread_.join();
    std::filesystem::remove(snapshot_);
  }

  json post(const std::string& path, const json& body, int expect = 200) {
    auto res = client_->Post(path, body.dump(), "application/json");
    EXPECT_TRUE(res) << path;
    if (!res) return {};
    EXPECT_EQ(res->status, expect) << path << ": " << res->body;
    return json::parse(res->body);
  }
  json get(const std::string& path, int expect = 200) {
    auto res = client_->Get(path);
    EXPECT_TRUE(res) << path;
    if (!res) return {};
    EXPECT_EQ(res->status, expect) << res->body;
    return json::parse(res->body);
  }

  std::filesystem::path snapshot_;
  std::unique_ptr<VectorService> service_;
  std::unique_ptr<HttpServer> server_;
  std::unique_ptr<httplib::Client> client_;
  std::thread thread_;
  int port_ = 0;
};

TEST_F(ServerTest, UpsertSearchDeleteStats) {
  EXPECT_GT(port_, 0);
  EXPECT_EQ(get("/health")["status"], "ok");

  // Batch upsert beyond the initial capacity of 2 (server must auto-grow).
  json vectors = json::array();
  for (int i = 0; i < 50; ++i) {
    vectors.push_back({{"id", i}, {"values", {i, i, i, i}}, {"tags", i % 2 == 0 ? 1 : 2}});
  }
  EXPECT_EQ(post("/vectors", {{"vectors", vectors}})["upserted"], 50);
  // Single-object form.
  EXPECT_EQ(post("/vectors", {{"id", 1000}, {"values", {100, 100, 100, 100}}})["upserted"], 1);

  json res = post("/search", {{"vector", {10.2, 10.2, 10.2, 10.2}}, {"k", 3}, {"ef", 32}});
  ASSERT_EQ(res["results"].size(), 3U);
  EXPECT_EQ(res["results"][0]["id"], 10);
  EXPECT_GE(res["took_us"].get<double>(), 0.0);

  // Filtered: only odd ids carry tag bit 2.
  res = post("/search", {{"vector", {10, 10, 10, 10}}, {"k", 5}, {"filter", {{"any_of", 2}}}});
  for (const auto& r : res["results"]) EXPECT_EQ(r["id"].get<int>() % 2, 1);

  EXPECT_EQ(get("/vectors/7")["values"], json({7.0, 7.0, 7.0, 7.0}));
  auto del = client_->Delete("/vectors/10");
  ASSERT_TRUE(del);
  EXPECT_EQ(del->status, 200);
  EXPECT_EQ(client_->Delete("/vectors/10")->status, 404);
  get("/vectors/10", 404);
  res = post("/search", {{"vector", {10, 10, 10, 10}}, {"k", 1}});
  EXPECT_NE(res["results"][0]["id"], 10);

  json stats = get("/stats");
  EXPECT_EQ(stats["size"], 50);
  EXPECT_EQ(stats["deleted"], 1);
  EXPECT_EQ(stats["dim"], 4);
  EXPECT_EQ(stats["metric"], "l2");
  EXPECT_GE(stats["capacity"].get<int>(), 51);

  // Compaction removes the tombstone; snapshot writes the configured file.
  json c = post("/admin/compact", json::object());
  EXPECT_EQ(c["nodes_before"], 51);
  EXPECT_EQ(c["nodes_after"], 50);
  EXPECT_EQ(get("/stats")["deleted"], 0);
  json snap = post("/admin/snapshot", json::object());
  EXPECT_TRUE(std::filesystem::exists(snapshot_));
  const HnswIndex loaded = HnswIndex::load(snapshot_);
  EXPECT_EQ(loaded.size(), 50U);
}

TEST_F(ServerTest, RejectsBadRequests) {
  post("/vectors", {{"id", 1}, {"values", {1, 2}}}, 400);         // wrong dimension
  post("/vectors", {{"id", -5}, {"values", {1, 2, 3, 4}}}, 400);  // negative id
  post("/vectors", {{"values", {1, 2, 3, 4}}}, 400);              // missing id
  post("/search", {{"vector", {1, 2, 3, 4}}, {"k", 0}}, 400);     // bad k
  post("/search", {{"vector", "nope"}}, 400);                     // not an array
  auto res = client_->Post("/search", "{not json", "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 400);
  get("/vectors/abc", 400);
}

TEST_F(ServerTest, ConcurrentClients) {
  test::ThreadGroup clients;
  for (int t = 0; t < 4; ++t) {
    clients.spawn([this, t] {
      httplib::Client c("127.0.0.1", port_);
      for (int i = 0; i < 50; ++i) {
        const int id = t * 1000 + i;
        const json body = {{"id", id}, {"values", {id, 0, 0, 0}}};
        auto r = c.Post("/vectors", body.dump(), "application/json");
        EXPECT_TRUE(r && r->status == 200);
        const json q = {{"vector", {id, 0, 0, 0}}, {"k", 1}};
        auto s = c.Post("/search", q.dump(), "application/json");
        EXPECT_TRUE(s && s->status == 200);
      }
    });
  }
  clients.join();
  EXPECT_EQ(get("/stats")["size"], 200);
}

}  // namespace
}  // namespace vs::server
