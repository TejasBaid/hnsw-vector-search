#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

#include "test_util.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/io.hpp"

namespace vs {
namespace {

constexpr std::size_t kDim = 24;
constexpr std::size_t kN = 3000;

class Persistence : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           ("vecsearch_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
            "_" + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(dir_);
    data_ = test::clustered_data(kN, kDim, 77);
    queries_ = test::clustered_data(50, kDim, 78);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }

  HnswIndex build(Metric metric = Metric::L2) const {
    std::vector<std::uint64_t> tags(kN);
    for (std::size_t i = 0; i < kN; ++i) tags[i] = i % 7;
    HnswIndex index(
        {.dim = kDim, .metric = metric, .M = 12, .ef_construction = 80, .capacity = kN});
    index.add_batch(data_, test::iota_labels(kN, 1000), 4, tags);
    return index;
  }

  void expect_same_results(const HnswIndex& a, const HnswIndex& b, const SearchParams& p) const {
    for (std::size_t q = 0; q < 50; ++q) {
      const std::span<const float> query(queries_.data() + q * kDim, kDim);
      EXPECT_EQ(a.search(query, p), b.search(query, p)) << "query " << q;
    }
  }

  std::filesystem::path dir_;
  std::vector<float> data_;
  std::vector<float> queries_;
};

TEST_F(Persistence, RoundTripCopyAndMmapGiveIdenticalResults) {
  for (const Metric metric : {Metric::L2, Metric::Cosine}) {
    const HnswIndex index = build(metric);
    const auto path = dir_ / "index.vsi";
    index.save(path);
    EXPECT_FALSE(std::filesystem::exists(dir_ / "index.vsi.tmp"));

    const HnswIndex copy = HnswIndex::load(path, LoadMode::Copy);
    const HnswIndex mapped = HnswIndex::load(path, LoadMode::Mmap);
    EXPECT_FALSE(copy.read_only());
    EXPECT_TRUE(mapped.read_only());
    EXPECT_EQ(copy.size(), index.size());
    EXPECT_EQ(mapped.metric(), metric);
    EXPECT_EQ(copy.config().M, 12U);
    EXPECT_GT(mapped.memory_usage().mapped, 0U);
    EXPECT_TRUE(copy.check_integrity().ok);
    EXPECT_TRUE(mapped.check_integrity().ok);

    const SearchParams p{.k = 10, .ef = 64};
    expect_same_results(index, copy, p);
    expect_same_results(index, mapped, p);
    const SearchParams filtered{.k = 5, .ef = 64, .filter = {.any_of = 0b100}};
    expect_same_results(index, mapped, filtered);
  }
}

TEST_F(Persistence, TombstonesAndQuantizerSurviveRoundTrip) {
  HnswIndex index = build();
  for (label_t l = 1000; l < 1000 + kN; l += 5) index.remove(l);
  index.train_quantizer({.type = QuantizerType::SQ8});
  const auto path = dir_ / "q.vsi";
  index.save(path);

  for (const LoadMode mode : {LoadMode::Copy, LoadMode::Mmap}) {
    const HnswIndex loaded = HnswIndex::load(path, mode);
    EXPECT_EQ(loaded.deleted_count(), index.deleted_count());
    EXPECT_EQ(loaded.quantizer_type(), QuantizerType::SQ8);
    EXPECT_FALSE(loaded.contains(1000));
    EXPECT_TRUE(loaded.contains(1001));
    expect_same_results(index, loaded, {.k = 10, .ef = 64, .use_quantized = true});
    expect_same_results(index, loaded, {.k = 10, .ef = 64});
  }
}

TEST_F(Persistence, MmapIndexBecomesWritableAfterReserve) {
  const auto path = dir_ / "w.vsi";
  build().save(path);
  HnswIndex mapped = HnswIndex::load(path, LoadMode::Mmap);
  EXPECT_THROW(mapped.add(1, std::vector<float>(kDim, 0.5F)), Error);
  EXPECT_TRUE(mapped.remove(1000));  // tombstones are allowed on a mapped index
  mapped.reserve(kN + 100);
  EXPECT_FALSE(mapped.read_only());
  mapped.add(1, std::vector<float>(kDim, 0.5F));
  EXPECT_TRUE(mapped.contains(1));
  EXPECT_FALSE(mapped.contains(1000));
  EXPECT_EQ(mapped.memory_usage().mapped, 0U);
  EXPECT_TRUE(mapped.check_integrity().ok);
}

TEST_F(Persistence, EmptyIndexRoundTrips) {
  const auto path = dir_ / "empty.vsi";
  HnswIndex({.dim = 4}).save(path);
  const HnswIndex loaded = HnswIndex::load(path);
  EXPECT_EQ(loaded.size(), 0U);
  EXPECT_TRUE(loaded.search(std::vector<float>{1, 2, 3, 4}, {}).empty());
}

TEST_F(Persistence, CorruptFilesAreRejected) {
  const auto path = dir_ / "c.vsi";
  build().save(path);
  const auto size = std::filesystem::file_size(path);

  auto with_bytes = [&](auto mutate) {
    const auto bad = dir_ / "bad.vsi";
    std::filesystem::copy_file(path, bad, std::filesystem::copy_options::overwrite_existing);
    mutate(bad);
    return bad;
  };
  auto poke = [](const std::filesystem::path& p, std::streamoff off, char value) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(off);
    f.write(&value, 1);
  };

  // Bad magic.
  EXPECT_THROW((void)HnswIndex::load(with_bytes([&](auto& p) { poke(p, 0, 'X'); })), FormatError);
  // Future format version.
  EXPECT_THROW((void)HnswIndex::load(with_bytes([&](auto& p) { poke(p, 8, 9); })), FormatError);
  // Header bit flip -> checksum mismatch.
  EXPECT_THROW((void)HnswIndex::load(with_bytes([&](auto& p) { poke(p, 40, 0x55); })), FormatError);
  // Truncated file.
  EXPECT_THROW((void)HnswIndex::load(
                   with_bytes([&](auto& p) { std::filesystem::resize_file(p, size / 2); })),
               FormatError);
  // Not an index at all.
  const auto junk = dir_ / "junk.vsi";
  std::ofstream(junk) << "hello world, definitely not an index";
  EXPECT_THROW((void)HnswIndex::load(junk), FormatError);
  // Missing file.
  EXPECT_THROW((void)HnswIndex::load(dir_ / "nope.vsi"), Error);
}

TEST_F(Persistence, CorruptAdjacencyIsDetected) {
  // Flip bytes inside the level-0 graph section: the loader validates every
  // neighbour id, so this must fail cleanly instead of reading out of bounds.
  const auto path = dir_ / "g.vsi";
  build().save(path);
  const auto vec_bytes = kN * kDim * sizeof(float);
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    // Header is < 1 KiB and sections are 64-byte aligned; the level-0
    // section starts right after the vectors.
    const auto off = static_cast<std::streamoff>(((1024 + vec_bytes) / 64 + 1) * 64);
    f.seekp(off);
    const std::uint32_t garbage = 0xFFFFFFF0U;
    for (int i = 0; i < 64; ++i) f.write(reinterpret_cast<const char*>(&garbage), sizeof(garbage));
  }
  EXPECT_THROW((void)HnswIndex::load(path), FormatError);
}

TEST(DatasetIo, FvecsIvecsRoundTrip) {
  const auto dir = std::filesystem::temp_directory_path();
  Matrix<float> m(5, 3);
  for (std::size_t i = 0; i < 15; ++i) m.data[i] = static_cast<float>(i) * 0.5F;
  Matrix<std::int32_t> g(2, 4);
  for (std::size_t i = 0; i < 8; ++i) g.data[i] = static_cast<std::int32_t>(i);
  const auto fp = dir / "vs_test.fvecs";
  const auto ip = dir / "vs_test.ivecs";
  write_fvecs(fp, m);
  write_ivecs(ip, g);
  const auto m2 = read_fvecs(fp);
  const auto g2 = read_ivecs(ip);
  ASSERT_EQ(m2.rows, 5U);
  ASSERT_EQ(m2.cols, 3U);
  for (std::size_t i = 0; i < 15; ++i) EXPECT_EQ(m2.data[i], m.data[i]);
  ASSERT_EQ(g2.rows, 2U);
  for (std::size_t i = 0; i < 8; ++i) EXPECT_EQ(g2.data[i], g.data[i]);
  EXPECT_EQ(read_fvecs(fp, 2).rows, 2U);
  std::filesystem::remove(fp);
  std::filesystem::remove(ip);
}

}  // namespace
}  // namespace vs
