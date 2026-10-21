// Versioned binary on-disk format (see README "On-disk format").
//
//   [header]  magic "VSHNSW\0\0", version, endianness tag, config, counts,
//             entry point, section table, FNV-1a checksum of the header
//   [section] VECTORS | LEVEL0 | LEVELS | UPPER | LABELS | TAGS | DELETED |
//             QUANTIZER | CODES, each starting on a 64-byte boundary
//
// Large arrays are stored exactly as they are laid out in memory, so LoadMode
// ::Mmap can point the index straight at the mapped pages.
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <fstream>
#include <string>
#include <system_error>
#include <type_traits>

#include "hnsw_impl.hpp"
#include "vecsearch/detail/byte_stream.hpp"

namespace vs {

namespace {

constexpr std::array<char, 8> kMagic{'V', 'S', 'H', 'N', 'S', 'W', '\0', '\0'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kEndianTag = 0x01020304U;
constexpr std::size_t kSectionAlign = 64;
constexpr std::uint32_t kFlagKeepPruned = 1U;

enum class Section : std::uint32_t {
  Vectors = 1,
  Level0 = 2,
  Levels = 3,
  Upper = 4,
  Labels = 5,
  Tags = 6,
  Deleted = 7,
  Quantizer = 8,
  Codes = 9,
};

struct SectionEntry {
  std::uint32_t kind = 0;
  std::uint32_t reserved = 0;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
};
static_assert(sizeof(SectionEntry) == 24);

std::uint64_t fnv1a(std::span<const std::byte> data) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const std::byte b : data) {
    h ^= static_cast<std::uint64_t>(b);
    h *= 0x100000001b3ULL;
  }
  return h;
}

constexpr std::uint64_t align_up(std::uint64_t x) noexcept {
  return (x + kSectionAlign - 1) / kSectionAlign * kSectionAlign;
}

struct Payload {
  Section kind;
  const void* data;
  std::uint64_t size;
};

void fsync_path(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd >= 0) {
    (void)::fsync(fd);
    ::close(fd);
  }
}

}  // namespace

void HnswIndex::save(const std::filesystem::path& path) const {
  const Impl& ix = *impl_;
  const std::uint64_t n = ix.count.load(std::memory_order_acquire);

  // Flatten the upper layers: node i contributes levels[i] * (1 + M) uint32s,
  // so offsets can be recomputed from LEVELS on load.
  std::vector<std::uint32_t> upper_blob;
  for (node_id i = 0; i < n; ++i) {
    const std::size_t words = static_cast<std::size_t>(ix.levels[i]) * ix.upper_stride();
    if (words != 0)
      upper_blob.insert(upper_blob.end(), ix.upper[i].get(), ix.upper[i].get() + words);
  }
  // Only the tombstone bit is persistent state.
  std::vector<std::uint8_t> deleted_bytes(n);
  for (node_id i = 0; i < n; ++i) deleted_bytes[i] = ix.is_deleted(i) ? 1 : 0;
  std::vector<std::byte> qblob;
  if (ix.qtype == QuantizerType::SQ8) ix.sq.serialize(qblob);
  if (ix.qtype == QuantizerType::PQ) ix.pq.serialize(qblob);

  std::vector<Payload> payloads{
      {Section::Vectors, ix.vectors, n * ix.dim * sizeof(float)},
      {Section::Level0, ix.level0, n * ix.level0_stride() * sizeof(std::uint32_t)},
      {Section::Levels, ix.levels.data(), n},
      {Section::Upper, upper_blob.data(), upper_blob.size() * sizeof(std::uint32_t)},
      {Section::Labels, ix.labels.data(), n * sizeof(label_t)},
      {Section::Tags, ix.tags.data(), n * sizeof(std::uint64_t)},
      {Section::Deleted, deleted_bytes.data(), n},
  };
  if (ix.qtype != QuantizerType::None) {
    payloads.push_back({Section::Quantizer, qblob.data(), qblob.size()});
    payloads.push_back({Section::Codes, ix.codes, n * ix.code_size});
  }

  // Header (everything up to and including the checksum).
  auto build_header = [&](std::span<const SectionEntry> table) {
    detail::ByteWriter w;
    w.put(kMagic);
    w.put(kFormatVersion);
    w.put(kEndianTag);
    w.put(static_cast<std::uint32_t>(ix.cfg.metric));
    w.put(static_cast<std::uint32_t>(ix.qtype));
    w.put(static_cast<std::uint64_t>(ix.dim));
    w.put(static_cast<std::uint64_t>(ix.M));
    w.put(static_cast<std::uint64_t>(ix.cfg.ef_construction));
    w.put(static_cast<std::uint64_t>(ix.cfg.seed));
    w.put(n);
    w.put(ix.entry.load(std::memory_order_acquire));
    w.put(static_cast<std::uint64_t>(ix.num_deleted.load(std::memory_order_acquire)));
    w.put(ix.cfg.keep_pruned ? kFlagKeepPruned : 0U);
    w.put(static_cast<std::uint32_t>(table.size()));
    w.put_array(table);
    const std::uint64_t checksum = fnv1a(w.bytes());
    w.put(checksum);
    return w.take();
  };

  std::vector<SectionEntry> table(payloads.size());
  const std::size_t header_size = build_header(table).size();
  std::uint64_t offset = align_up(header_size);
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    table[i] = {static_cast<std::uint32_t>(payloads[i].kind), 0, offset, payloads[i].size};
    offset = align_up(offset + payloads[i].size);
  }
  const std::vector<std::byte> header = build_header(table);

  // Write to a temporary file and rename, so a crash never leaves a torn index.
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw Error("save: cannot open '" + tmp.string() + "' for writing");
    const std::array<char, kSectionAlign> zeros{};
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));
    std::uint64_t pos = header.size();
    for (std::size_t i = 0; i < payloads.size(); ++i) {
      out.write(zeros.data(), static_cast<std::streamsize>(table[i].offset - pos));
      if (payloads[i].size != 0) {
        out.write(static_cast<const char*>(payloads[i].data),
                  static_cast<std::streamsize>(payloads[i].size));
      }
      pos = table[i].offset + payloads[i].size;
    }
    out.flush();
    if (!out) throw Error("save: write to '" + tmp.string() + "' failed");
  }
  fsync_path(tmp);
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) throw Error("save: cannot rename '" + tmp.string() + "': " + ec.message());
}

HnswIndex HnswIndex::load(const std::filesystem::path& path, LoadMode mode, SimdBackend simd) {
  MappedFile file(path);
  const std::span<const std::byte> bytes = file.bytes();
  detail::ByteReader r(bytes);

  const auto magic = r.get<std::array<char, 8>>();
  if (magic != kMagic) throw FormatError("'" + path.string() + "' is not a vecsearch index");
  const auto version = r.get<std::uint32_t>();
  if (version == 0 || version > kFormatVersion) {
    throw FormatError("unsupported index format version " + std::to_string(version) +
                      " (this build reads up to " + std::to_string(kFormatVersion) + ")");
  }
  if (r.get<std::uint32_t>() != kEndianTag) throw FormatError("index has foreign endianness");

  HnswConfig cfg;
  cfg.simd = simd;
  const auto metric = r.get<std::uint32_t>();
  const auto qtype = r.get<std::uint32_t>();
  if (metric > 2 || qtype > 2) throw FormatError("bad metric or quantizer type");
  cfg.metric = static_cast<Metric>(metric);
  cfg.dim = r.get<std::uint64_t>();
  cfg.M = r.get<std::uint64_t>();
  cfg.ef_construction = r.get<std::uint64_t>();
  cfg.seed = r.get<std::uint64_t>();
  const auto n = r.get<std::uint64_t>();
  const auto entry = r.get<std::uint64_t>();
  const auto num_deleted = r.get<std::uint64_t>();
  const auto flags = r.get<std::uint32_t>();
  cfg.keep_pruned = (flags & kFlagKeepPruned) != 0;
  const auto num_sections = r.get<std::uint32_t>();
  if (num_sections > 64) throw FormatError("too many sections");
  std::vector<SectionEntry> table(num_sections);
  r.get_array(std::span<SectionEntry>(table));
  const std::size_t checked_bytes = bytes.size() - r.remaining();
  if (r.get<std::uint64_t>() != fnv1a(bytes.first(checked_bytes))) {
    throw FormatError("header checksum mismatch (file corrupt)");
  }
  if (n >= kInvalidNode) throw FormatError("node count out of range");
  if (cfg.dim == 0 || cfg.dim > (1U << 20)) throw FormatError("bad dimension");
  cfg.capacity = std::max<std::uint64_t>(n, 1);

  auto impl = std::make_unique<Impl>(cfg);  // validates M etc.
  Impl& ix = *impl;

  auto section = [&](Section kind, std::uint64_t expected, bool required) -> std::byte* {
    for (const SectionEntry& s : table) {
      if (s.kind != static_cast<std::uint32_t>(kind)) continue;
      if (s.offset > bytes.size() || s.size > bytes.size() - s.offset) {
        throw FormatError("section " + std::to_string(s.kind) + " extends past end of file");
      }
      if (s.offset % kSectionAlign != 0) throw FormatError("misaligned section");
      if (expected != ~std::uint64_t{0} && s.size != expected) {
        throw FormatError("section " + std::to_string(s.kind) + " has size " +
                          std::to_string(s.size) + ", expected " + std::to_string(expected));
      }
      return file.data() + s.offset;
    }
    if (required) throw FormatError("missing section " + std::to_string(static_cast<int>(kind)));
    return nullptr;
  };
  constexpr std::uint64_t kAnySize = ~std::uint64_t{0};

  // Small per-node arrays are always copied (they are mutable metadata).
  auto copy_into = [&](auto& buf, std::size_t cap, const std::byte* src, std::size_t bytes_n) {
    using T = std::remove_reference_t<decltype(buf[0])>;
    buf = AlignedBuffer<T>(cap);
    if (bytes_n != 0) std::memcpy(buf.data(), src, bytes_n);
  };
  const std::size_t cap = cfg.capacity;
  copy_into(ix.levels, cap, section(Section::Levels, n, true), n);
  copy_into(ix.labels, cap, section(Section::Labels, n * sizeof(label_t), true),
            n * sizeof(label_t));
  copy_into(ix.tags, cap, section(Section::Tags, n * sizeof(std::uint64_t), true),
            n * sizeof(std::uint64_t));
  copy_into(ix.state, cap, section(Section::Deleted, n, true), n);

  std::uint64_t upper_words = 0;
  for (std::uint64_t i = 0; i < n; ++i) {
    if (ix.levels[i] > kMaxLevel) throw FormatError("node level out of range");
    upper_words += static_cast<std::uint64_t>(ix.levels[i]) * ix.upper_stride();
  }
  const std::byte* upper_src = section(Section::Upper, upper_words * sizeof(std::uint32_t), true);
  std::byte* vec_src = section(Section::Vectors, n * ix.dim * sizeof(float), true);
  std::byte* l0_src =
      section(Section::Level0, n * ix.level0_stride() * sizeof(std::uint32_t), true);

  ix.upper.resize(cap);
  {
    std::uint64_t off = 0;
    for (node_id i = 0; i < n; ++i) {
      const int L = ix.levels[i];
      if (L == 0) continue;
      ix.allocate_upper(i, L);
      const std::size_t words = static_cast<std::size_t>(L) * ix.upper_stride();
      std::memcpy(ix.upper[i].get(), upper_src + off * sizeof(std::uint32_t),
                  words * sizeof(std::uint32_t));
      off += words;
    }
  }

  ix.qtype = static_cast<QuantizerType>(qtype);
  std::byte* codes_src = nullptr;
  if (ix.qtype != QuantizerType::None) {
    const SectionEntry* qs = nullptr;
    for (const SectionEntry& s : table) {
      if (s.kind == static_cast<std::uint32_t>(Section::Quantizer)) qs = &s;
    }
    const std::byte* qsrc = section(Section::Quantizer, kAnySize, true);
    const std::span<const std::byte> qbytes(qsrc, qs->size);
    if (ix.qtype == QuantizerType::SQ8) {
      ix.sq = ScalarQuantizer::deserialize(qbytes);
      ix.code_size = ix.sq.code_size();
    } else {
      ix.pq = ProductQuantizer::deserialize(qbytes);
      ix.code_size = ix.pq.code_size();
    }
    codes_src = section(Section::Codes, n * ix.code_size, true);
  }

  if (mode == LoadMode::Mmap) {
    ix.vectors = reinterpret_cast<float*>(vec_src);
    ix.level0 = reinterpret_cast<std::uint32_t*>(l0_src);
    ix.codes = reinterpret_cast<std::uint8_t*>(codes_src);
    ix.read_only = true;
  } else {
    ix.vectors_owned = AlignedBuffer<float>(cap * ix.dim);
    std::memcpy(ix.vectors_owned.data(), vec_src, n * ix.dim * sizeof(float));
    ix.vectors = ix.vectors_owned.data();
    ix.level0_owned = AlignedBuffer<std::uint32_t>(cap * ix.level0_stride());
    std::memcpy(ix.level0_owned.data(), l0_src, n * ix.level0_stride() * sizeof(std::uint32_t));
    ix.level0 = ix.level0_owned.data();
    if (codes_src != nullptr) {
      ix.codes_owned = AlignedBuffer<std::uint8_t>(cap * ix.code_size);
      std::memcpy(ix.codes_owned.data(), codes_src, n * ix.code_size);
      ix.codes = ix.codes_owned.data();
    }
  }
  ix.locks = std::make_unique<detail::SpinLock[]>(cap);
  ix.visited = std::make_unique<detail::VisitedPool>(cap);
  ix.capacity = cap;

  // Validate every adjacency list so that a corrupt file can never make the
  // search read out of bounds.
  for (node_id i = 0; i < n; ++i) {
    for (int l = 0; l <= ix.levels[i]; ++l) {
      const std::uint32_t* ll = ix.links(i, l);
      if (ll[0] > ix.max_links(l)) throw FormatError("adjacency list exceeds max degree");
      for (std::uint32_t j = 0; j < ll[0]; ++j) {
        const node_id x = ll[1 + j];
        if (x >= n || ix.levels[x] < l) throw FormatError("adjacency list has invalid node id");
      }
    }
  }
  if (n != 0) {
    if (entry == kNoEntry || entry_id(entry) >= n ||
        entry_level(entry) != ix.levels[entry_id(entry)]) {
      throw FormatError("invalid entry point");
    }
  } else if (entry != kNoEntry) {
    throw FormatError("empty index with an entry point");
  }

  std::size_t deleted = 0;
  ix.label_map.reserve(n);
  for (node_id i = 0; i < n; ++i) {
    const bool is_deleted = (ix.state[i] & Impl::kStateDeleted) != 0;
    ix.state[i] =
        static_cast<std::uint8_t>((is_deleted ? Impl::kStateDeleted : 0) | Impl::kStateReady);
    if (is_deleted) {
      ++deleted;
      continue;
    }
    if (!ix.label_map.try_emplace(ix.labels[i], i).second) {
      throw FormatError("duplicate live label " + std::to_string(ix.labels[i]));
    }
    ix.count_tags(ix.tags[i], +1);
  }
  if (deleted != num_deleted) throw FormatError("tombstone count mismatch");

  ix.count.store(static_cast<std::uint32_t>(n), std::memory_order_release);
  ix.entry.store(entry, std::memory_order_release);
  ix.num_deleted.store(deleted, std::memory_order_release);
  if (mode == LoadMode::Mmap) ix.mapping = std::move(file);
  return HnswIndex(std::move(impl));
}

}  // namespace vs
