// HNSW construction and search.
//
// Concurrency design (see also hnsw_index.hpp):
//  * Node ids are allocated with a CAS on `count` and never reused, so every
//    slot is written exactly once before it becomes reachable.
//  * A node's vector, label, tags and level are written before it is linked
//    into any neighbour list. Links are published with release stores and read
//    with acquire loads, which gives readers a happens-before edge to all of
//    that data without taking any lock.
//  * Writers modify a neighbour list only while holding that node's 4-byte
//    spin lock, and never hold two node locks at once (no lock ordering, no
//    deadlock). The only other lock on the insert path is `entry_mutex`, held
//    by the rare insert whose level exceeds the current maximum.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>

#include "hnsw_impl.hpp"
#include "vecsearch/parallel.hpp"

namespace vs {

namespace {

constexpr std::size_t kMaxM = 128;
constexpr std::size_t kMaxDegree = 2 * kMaxM;

struct Candidate {
  float dist;
  node_id id;
};
struct CloserFirst {  // priority_queue comparator -> min-heap on distance
  bool operator()(const Candidate& a, const Candidate& b) const noexcept { return a.dist > b.dist; }
};
struct FurtherFirst {  // priority_queue comparator -> max-heap on distance
  bool operator()(const Candidate& a, const Candidate& b) const noexcept { return a.dist < b.dist; }
};
using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst>;
using MaxHeap = std::priority_queue<Candidate, std::vector<Candidate>, FurtherFirst>;

constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline void prefetch_read(const void* p) noexcept { __builtin_prefetch(p, 0, 3); }

using Impl = HnswIndex::Impl;

// ---- distance-to-query functors -------------------------------------------
// The traversal is templated on these so the hot loop is specialised per
// representation (float / SQ8 / PQ) with no virtual dispatch.

struct FloatDistance {
  const float* query;
  const Impl* ix;
  float operator()(node_id id) const noexcept { return ix->dist(query, ix->vec(id), ix->dim); }
  void prefetch(node_id id) const noexcept {
    const auto* p = reinterpret_cast<const char*>(ix->vec(id));
    prefetch_read(p);
    if (ix->dim * sizeof(float) > kCacheLine) prefetch_read(p + kCacheLine);
  }
};

struct Sq8Distance {
  const ScalarQuantizer::Query* query;
  const Impl* ix;
  float operator()(node_id id) const noexcept { return query->distance(ix->code(id)); }
  void prefetch(node_id id) const noexcept { prefetch_read(ix->code(id)); }
};

struct PqDistance {
  const ProductQuantizer::Query* query;
  const Impl* ix;
  float operator()(node_id id) const noexcept { return query->distance(ix->code(id)); }
  void prefetch(node_id id) const noexcept { prefetch_read(ix->code(id)); }
};

// ---- result acceptance predicates -------------------------------------------
// Rejected nodes are still expanded (they keep the graph navigable) but never
// enter the result set.

struct AcceptAll {
  constexpr bool operator()(node_id /*id*/) const noexcept { return true; }
};

struct AcceptLive {
  const Impl* ix;
  const TagFilter* filter;
  FunctionRef<bool(label_t)> predicate;
  bool operator()(node_id id) const {
    if (ix->is_deleted(id)) return false;
    if (filter->active() && !filter->matches(ix->tags[id])) return false;
    return !predicate || predicate(ix->labels[id]);
  }
};

// ---- core graph algorithms -------------------------------------------------

/// Greedy best-neighbour walk on layers (from_level, to_level].
template <class Dist>
Candidate greedy_descend(const Impl& ix, const Dist& dist, Candidate cur, int from_level,
                         int to_level, std::size_t& ndist) {
  for (int level = from_level; level > to_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const std::uint32_t* ll = ix.links(cur.id, level);
      const std::uint32_t n = Impl::load_acquire(ll[0]);
      for (std::uint32_t j = 0; j < n; ++j) {
        const node_id nb = Impl::load_acquire(ll[1 + j]);
        const float d = dist(nb);
        ++ndist;
        if (d < cur.dist) {
          cur = {d, nb};
          changed = true;
        }
      }
    }
  }
  return cur;
}

/// Beam search on one layer (Algorithm 2 of the paper). Returns up to `ef`
/// accepted candidates sorted by ascending distance.
template <class Dist, class Accept>
std::vector<Candidate> search_layer(const Impl& ix, const Dist& dist,
                                    std::span<const Candidate> entry_points, std::size_t ef,
                                    int level, const Accept& accept, detail::VisitedList& visited,
                                    SearchStats& stats) {
  std::vector<Candidate> cbuf;
  cbuf.reserve(ef * 2);
  MinHeap candidates(CloserFirst{}, std::move(cbuf));
  std::vector<Candidate> tbuf;
  tbuf.reserve(ef + 1);
  MaxHeap top(FurtherFirst{}, std::move(tbuf));

  for (const Candidate& ep : entry_points) {
    if (visited.test_and_mark(ep.id)) continue;
    candidates.push(ep);
    if (accept(ep.id)) {
      top.push(ep);
      if (top.size() > ef) top.pop();
    }
  }
  float lower = top.empty() ? std::numeric_limits<float>::infinity() : top.top().dist;

  std::array<node_id, kMaxDegree> fresh{};
  std::size_t ndist = 0;
  std::size_t hops = 0;
  while (!candidates.empty()) {
    const Candidate cur = candidates.top();
    if (top.size() >= ef && cur.dist > lower) break;
    candidates.pop();
    ++hops;

    // Pass 1: collect unvisited neighbours and issue prefetches for all of
    // them, so their cache misses overlap instead of serialising.
    const std::uint32_t* ll = ix.links(cur.id, level);
    const std::uint32_t n = Impl::load_acquire(ll[0]);
    std::size_t nfresh = 0;
    for (std::uint32_t j = 0; j < n; ++j) {
      const node_id nb = Impl::load_acquire(ll[1 + j]);
      if (visited.test_and_mark(nb)) continue;
      dist.prefetch(nb);
      fresh[nfresh++] = nb;
    }
    // Pass 2: score them.
    for (std::size_t j = 0; j < nfresh; ++j) {
      const node_id nb = fresh[j];
      const float d = dist(nb);
      ++ndist;
      if (top.size() < ef || d < lower) {
        candidates.push({d, nb});
        if (accept(nb)) {
          top.push({d, nb});
          if (top.size() > ef) top.pop();
        }
        if (!top.empty()) lower = top.top().dist;
      }
    }
  }
  stats.distance_computations += ndist;
  stats.hops += hops;

  std::vector<Candidate> out(top.size());
  for (std::size_t i = out.size(); i-- > 0;) {
    out[i] = top.top();
    top.pop();
  }
  return out;
}

/// Neighbour-selection heuristic (Algorithm 4). `sorted` must be ascending by
/// distance to the base element. A candidate is kept only if it is closer to
/// the base than to every already-selected neighbour, which favours links in
/// diverse directions and keeps clustered data navigable.
std::vector<Candidate> select_neighbors(const Impl& ix, std::span<const Candidate> sorted,
                                        std::size_t m) {
  if (sorted.size() <= m) return {sorted.begin(), sorted.end()};
  std::vector<Candidate> result;
  result.reserve(m);
  std::vector<Candidate> pruned;
  for (const Candidate& c : sorted) {
    if (result.size() >= m) break;
    bool keep = true;
    for (const Candidate& r : result) {
      if (ix.dist(ix.vec(c.id), ix.vec(r.id), ix.dim) < c.dist) {
        keep = false;
        break;
      }
    }
    if (keep) {
      result.push_back(c);
    } else if (ix.cfg.keep_pruned) {
      pruned.push_back(c);
    }
  }
  for (std::size_t i = 0; i < pruned.size() && result.size() < m; ++i) result.push_back(pruned[i]);
  return result;
}

void write_links(std::uint32_t* ll, std::span<const Candidate> sel) noexcept {
  for (std::size_t i = 0; i < sel.size(); ++i) {
    Impl::store_release(ll[1 + i], sel[i].id);
  }
  Impl::store_release(ll[0], static_cast<std::uint32_t>(sel.size()));
}

/// Sets the new node's own neighbour list on `level`. Normally the list is
/// still empty, but a concurrent inserter that started its descent at this
/// node may already have linked itself in; such links are merged, not lost.
void set_own_links(Impl& ix, node_id id, int level, std::vector<Candidate> selected) {
  const std::scoped_lock lock(ix.locks[id]);
  std::uint32_t* ll = ix.links(id, level);
  const std::uint32_t existing = Impl::load_relaxed(ll[0]);
  if (existing != 0) {
    for (std::uint32_t j = 0; j < existing; ++j) {
      const node_id x = Impl::load_relaxed(ll[1 + j]);
      const bool dup = std::any_of(selected.begin(), selected.end(),
                                   [x](const Candidate& c) { return c.id == x; });
      if (!dup && x != id) selected.push_back({ix.dist(ix.vec(id), ix.vec(x), ix.dim), x});
    }
    std::sort(selected.begin(), selected.end(),
              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
    selected = select_neighbors(ix, selected, ix.max_links(level));
  }
  write_links(ll, selected);
}

/// Adds the reverse edge target -> newcomer, shrinking target's list with the
/// heuristic when it is full.
void add_back_link(Impl& ix, node_id target, Candidate newcomer, int level) {
  const std::scoped_lock lock(ix.locks[target]);
  std::uint32_t* ll = ix.links(target, level);
  const std::uint32_t n = Impl::load_relaxed(ll[0]);
  for (std::uint32_t j = 0; j < n; ++j) {
    if (Impl::load_relaxed(ll[1 + j]) == newcomer.id) return;
  }
  const std::size_t max_links = ix.max_links(level);
  if (n < max_links) {
    Impl::store_release(ll[1 + n], newcomer.id);
    Impl::store_release(ll[0], n + 1);
    return;
  }
  std::vector<Candidate> cands;
  cands.reserve(n + 1);
  for (std::uint32_t j = 0; j < n; ++j) {
    const node_id x = Impl::load_relaxed(ll[1 + j]);
    cands.push_back({ix.dist(ix.vec(target), ix.vec(x), ix.dim), x});
  }
  cands.push_back(newcomer);
  std::sort(cands.begin(), cands.end(),
            [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
  write_links(ll, select_neighbors(ix, cands, max_links));
}

int sample_level(const Impl& ix, label_t label) noexcept {
  // Deterministic in (seed, label): the graph's layer structure does not
  // depend on thread scheduling, which makes parallel builds reproducible in
  // shape and keeps upserts of a label on the same layers.
  const std::uint64_t h = splitmix64(ix.cfg.seed ^ splitmix64(label));
  const double u = static_cast<double>((h >> 11) + 1) * 0x1.0p-53;  // (0, 1]
  const auto level = static_cast<int>(-std::log(u) * ix.level_mult);
  return std::min(level, kMaxLevel);
}

/// Exact scan over all live nodes that pass the filter. Used by the planner
/// when a filter is so selective that graph traversal would have to expand
/// most of the graph to fill its beam.
std::vector<Candidate> exact_scan(const Impl& ix, const float* q, const SearchParams& p,
                                  SearchStats& stats) {
  const std::size_t n = ix.count.load(std::memory_order_acquire);
  MaxHeap top;
  for (node_id id = 0; id < n; ++id) {
    const std::uint8_t st = ix.load_state(id);
    if ((st & Impl::kStateReady) == 0 || (st & Impl::kStateDeleted) != 0) continue;
    if (!p.filter.matches(ix.tags[id])) continue;
    if (p.predicate && !p.predicate(ix.labels[id])) continue;
    const float d = ix.dist(q, ix.vec(id), ix.dim);
    ++stats.distance_computations;
    if (top.size() < p.k) {
      top.push({d, id});
    } else if (d < top.top().dist) {
      top.pop();
      top.push({d, id});
    }
  }
  std::vector<Candidate> out(top.size());
  for (std::size_t i = out.size(); i-- > 0;) {
    out[i] = top.top();
    top.pop();
  }
  return out;
}

/// Upper-bound estimate of the number of live nodes matching a tag filter,
/// from the per-bit counters (exact for single-bit filters).
std::size_t estimate_matches(const Impl& ix, const TagFilter& f, std::size_t live) noexcept {
  auto bit_count = [&](int b) {
    return static_cast<std::size_t>(std::max<std::int64_t>(
        0, ix.tag_counts[static_cast<std::size_t>(b)].load(std::memory_order_relaxed)));
  };
  std::size_t est = live;
  for (std::uint64_t t = f.all_of; t != 0; t &= t - 1)
    est = std::min(est, bit_count(std::countr_zero(t)));
  if (f.any_of != 0) {
    std::size_t sum = 0;
    for (std::uint64_t t = f.any_of; t != 0; t &= t - 1) sum += bit_count(std::countr_zero(t));
    est = std::min(est, sum);
  }
  return est;
}

/// Cost model, in units of one sequential-scan distance evaluation.
/// Calibrated on SIFT-1M (M=16) from the distance counts and timings that
/// bench/ann_bench reports for filtered sweeps:
///  * an unfiltered beam search evaluates D(ef) ~= 67 * ef^0.73 distances;
///  * under a filter of selectivity s that grows to D(ef) * s^-0.75 (the beam
///    must expand more nodes to collect ef matches);
///  * a graph-side evaluation costs ~2.5x a scan-side one (random access,
///    heap and visited-set work), and a scan's per-node tag check ~1/64.
/// The exact scan wins when s * N + N / 64 < 2.5 * D(ef) * s^-0.75.
bool prefer_exact_scan(const Impl& ix, const SearchParams& p, std::size_t ef) noexcept {
  if (!p.allow_exact_scan || !p.filter.active()) return false;
  const std::size_t live =
      ix.count.load(std::memory_order_relaxed) - ix.num_deleted.load(std::memory_order_relaxed);
  if (live == 0) return false;
  const double n = static_cast<double>(live);
  const double matches = static_cast<double>(estimate_matches(ix, p.filter, live));
  const double sel = std::max(matches, 1.0) / n;
  const double unfiltered = 67.0 * std::pow(static_cast<double>(ef), 0.73);
  const double graph_cost = 2.5 * unfiltered * std::pow(sel, -0.75);
  const double scan_cost = matches + n / 64.0;
  return scan_cost < graph_cost;
}

/// Links an already-initialised node into the graph (Algorithm 1).
void insert_node(Impl& ix, node_id id, int level) {
  std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  std::unique_lock<std::mutex> top_lock(ix.entry_mutex, std::defer_lock);
  if (packed == kNoEntry || entry_level(packed) < level) {
    top_lock.lock();
    packed = ix.entry.load(std::memory_order_acquire);
    if (packed == kNoEntry) {
      ix.entry.store(pack_entry(id, level), std::memory_order_release);
      return;
    }
    if (entry_level(packed) >= level) top_lock.unlock();
  }

  const int ep_level = entry_level(packed);
  const FloatDistance dist{ix.vec(id), &ix};
  SearchStats stats;
  Candidate cur{dist(entry_id(packed)), entry_id(packed)};
  cur = greedy_descend(ix, dist, cur, ep_level, level, stats.distance_computations);

  auto visited = ix.visited->acquire();
  std::vector<Candidate> eps{cur};
  for (int l = std::min(level, ep_level); l >= 0; --l) {
    visited->reset();
    std::vector<Candidate> found =
        search_layer(ix, dist, eps, ix.cfg.ef_construction, l, AcceptAll{}, *visited, stats);
    std::erase_if(found, [id](const Candidate& c) { return c.id == id; });
    const std::vector<Candidate> selected = select_neighbors(ix, found, ix.M);
    // Own list first, so a thread that reaches this node through one of the
    // reverse edges below can continue its traversal from here.
    set_own_links(ix, id, l, selected);
    for (const Candidate& c : selected) add_back_link(ix, c.id, {c.dist, id}, l);
    if (!found.empty()) eps = std::move(found);
  }

  if (top_lock.owns_lock()) ix.entry.store(pack_entry(id, level), std::memory_order_release);
}

}  // namespace

// ============================================================================
// Impl
// ============================================================================

HnswIndex::Impl::Impl(const HnswConfig& config) : cfg(config) {
  if (cfg.dim == 0) throw Error("HnswConfig.dim must be positive");
  if (cfg.M < 2 || cfg.M > kMaxM) throw Error("HnswConfig.M must be in [2, 128]");
  if (cfg.ef_construction == 0) throw Error("HnswConfig.ef_construction must be positive");
  dim = cfg.dim;
  M = cfg.M;
  M0 = 2 * cfg.M;
  level_mult = 1.0 / std::log(static_cast<double>(M));
  dist = distance_function(cfg.metric, cfg.simd);
}

void HnswIndex::Impl::allocate(std::size_t cap) {
  cap = std::max<std::size_t>(cap, 1);
  vectors_owned = AlignedBuffer<float>(cap * dim);
  vectors = vectors_owned.data();
  level0_owned = AlignedBuffer<std::uint32_t>(cap * level0_stride());
  level0 = level0_owned.data();
  upper.clear();
  upper.resize(cap);
  levels = AlignedBuffer<std::uint8_t>(cap);
  labels = AlignedBuffer<label_t>(cap);
  tags = AlignedBuffer<std::uint64_t>(cap);
  state = AlignedBuffer<std::uint8_t>(cap);
  locks = std::make_unique<detail::SpinLock[]>(cap);
  visited = std::make_unique<detail::VisitedPool>(cap);
  capacity = cap;
}

void HnswIndex::Impl::grow_owned(std::size_t cap) {
  const std::size_t n = count.load(std::memory_order_acquire);
  cap = std::max({cap, n, std::size_t{1}});
  AlignedBuffer<float> nv(cap * dim);
  if (n != 0) std::memcpy(nv.data(), vectors, n * dim * sizeof(float));
  AlignedBuffer<std::uint32_t> nl(cap * level0_stride());
  if (n != 0) std::memcpy(nl.data(), level0, n * level0_stride() * sizeof(std::uint32_t));
  if (code_size != 0) {
    AlignedBuffer<std::uint8_t> nc(cap * code_size);
    if (n != 0) std::memcpy(nc.data(), codes, n * code_size);
    codes_owned = std::move(nc);
    codes = codes_owned.data();
  }
  vectors_owned = std::move(nv);
  vectors = vectors_owned.data();
  level0_owned = std::move(nl);
  level0 = level0_owned.data();
  upper.resize(cap);
  levels = levels.resized_copy(cap);
  labels = labels.resized_copy(cap);
  tags = tags.resized_copy(cap);
  state = state.resized_copy(cap);
  locks = std::make_unique<detail::SpinLock[]>(cap);
  visited = std::make_unique<detail::VisitedPool>(cap);
  mapping = MappedFile{};
  capacity = cap;
  read_only = false;
}

void HnswIndex::Impl::allocate_upper(node_id id, int level) {
  if (level > 0) {
    upper[id] = std::make_unique<std::uint32_t[]>(static_cast<std::size_t>(level) * upper_stride());
  }
}

void HnswIndex::Impl::encode_node(node_id id) noexcept {
  std::uint8_t* out = codes + static_cast<std::size_t>(id) * code_size;
  switch (qtype) {
    case QuantizerType::SQ8: sq.encode(vec(id), out); break;
    case QuantizerType::PQ: pq.encode(vec(id), out); break;
    case QuantizerType::None: break;
  }
}

// ============================================================================
// HnswIndex
// ============================================================================

HnswIndex::HnswIndex(const HnswConfig& config) : impl_(std::make_unique<Impl>(config)) {
  impl_->allocate(config.capacity);
}
HnswIndex::HnswIndex(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
HnswIndex::~HnswIndex() = default;
HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
HnswIndex& HnswIndex::operator=(HnswIndex&&) noexcept = default;

void HnswIndex::add(label_t label, std::span<const float> vector, std::uint64_t tags) {
  Impl& ix = *impl_;
  if (ix.read_only) {
    throw Error("index is memory-mapped read-only; call reserve() to make it writable");
  }
  if (vector.size() != ix.dim) {
    throw Error("add: vector has dimension " + std::to_string(vector.size()) + ", index expects " +
                std::to_string(ix.dim));
  }

  // 1. Claim a fresh slot.
  std::uint32_t id = ix.count.load(std::memory_order_relaxed);
  do {
    if (id >= ix.capacity) {
      throw CapacityError("index is full (capacity " + std::to_string(ix.capacity) +
                          "); call reserve()");
    }
  } while (!ix.count.compare_exchange_weak(id, id + 1, std::memory_order_relaxed));

  // 2. Initialise the node privately (nobody can reach it yet).
  std::memcpy(ix.vec(id), vector.data(), ix.dim * sizeof(float));
  if (ix.cfg.metric == Metric::Cosine) normalize(std::span<float>(ix.vec(id), ix.dim));
  const int level = sample_level(ix, label);
  ix.labels[id] = label;
  ix.tags[id] = tags;
  ix.levels[id] = static_cast<std::uint8_t>(level);
  ix.allocate_upper(id, level);
  if (ix.qtype != QuantizerType::None) ix.encode_node(id);
  ix.set_state_bits(id, Impl::kStateReady);  // visible to exact scans from here on

  // 3. Publish the label (linearisation point of the upsert).
  {
    const std::scoped_lock lock(ix.label_mutex);
    auto [it, inserted] = ix.label_map.try_emplace(label, id);
    if (!inserted) {
      ix.set_deleted(it->second);
      ix.num_deleted.fetch_add(1, std::memory_order_relaxed);
      ix.count_tags(ix.tags[it->second], -1);
      it->second = id;
    }
    ix.count_tags(tags, +1);
  }

  // 4. Link into the graph.
  insert_node(ix, id, level);
}

void HnswIndex::add_batch(std::span<const float> vectors, std::span<const label_t> labels,
                          std::size_t num_threads, std::span<const std::uint64_t> tags) {
  const std::size_t d = impl_->dim;
  if (vectors.size() != labels.size() * d) {
    throw Error("add_batch: vectors.size() must equal labels.size() * dim");
  }
  if (!tags.empty() && tags.size() != labels.size()) {
    throw Error("add_batch: tags must be empty or have one entry per label");
  }
  parallel_for(
      0, labels.size(), num_threads,
      [&](std::size_t i) { add(labels[i], vectors.subspan(i * d, d), tags.empty() ? 0 : tags[i]); },
      16);
}

bool HnswIndex::remove(label_t label) {
  Impl& ix = *impl_;
  const std::scoped_lock lock(ix.label_mutex);
  const auto it = ix.label_map.find(label);
  if (it == ix.label_map.end()) return false;
  ix.set_deleted(it->second);
  ix.num_deleted.fetch_add(1, std::memory_order_relaxed);
  ix.count_tags(ix.tags[it->second], -1);
  ix.label_map.erase(it);
  return true;
}

bool HnswIndex::contains(label_t label) const {
  const std::scoped_lock lock(impl_->label_mutex);
  return impl_->label_map.contains(label);
}

std::vector<float> HnswIndex::get_vector(label_t label) const {
  const Impl& ix = *impl_;
  node_id id = kInvalidNode;
  {
    const std::scoped_lock lock(ix.label_mutex);
    const auto it = ix.label_map.find(label);
    if (it == ix.label_map.end()) throw Error("get_vector: unknown label " + std::to_string(label));
    id = it->second;
  }
  return {ix.vec(id), ix.vec(id) + ix.dim};
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query,
                                            const SearchParams& params) const {
  const Impl& ix = *impl_;
  if (query.size() != ix.dim) {
    throw Error("search: query has dimension " + std::to_string(query.size()) + ", index expects " +
                std::to_string(ix.dim));
  }
  if (params.k == 0) return {};
  const std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (packed == kNoEntry) return {};

  std::vector<float> normalized;
  const float* q = query.data();
  if (ix.cfg.metric == Metric::Cosine) {
    normalized.assign(query.begin(), query.end());
    normalize(normalized);
    q = normalized.data();
  }

  const std::size_t ef = std::max(params.ef, params.k);
  const bool filtered = params.filter.active() || static_cast<bool>(params.predicate) ||
                        ix.num_deleted.load(std::memory_order_relaxed) != 0;
  SearchStats stats;

  auto run = [&](const auto& dist) {
    Candidate cur{dist(entry_id(packed)), entry_id(packed)};
    ++stats.distance_computations;
    cur = greedy_descend(ix, dist, cur, entry_level(packed), 0, stats.distance_computations);
    auto visited = ix.visited->acquire();
    const std::array<Candidate, 1> eps{cur};
    if (filtered) {
      return search_layer(ix, dist, eps, ef, 0, AcceptLive{&ix, &params.filter, params.predicate},
                          *visited, stats);
    }
    return search_layer(ix, dist, eps, ef, 0, AcceptAll{}, *visited, stats);
  };

  std::vector<Candidate> found;
  if (prefer_exact_scan(ix, params, ef)) {
    found = exact_scan(ix, q, params, stats);
    stats.exact_scan = true;
  } else if (!params.use_quantized) {
    found = run(FloatDistance{q, &ix});
  } else {
    switch (ix.qtype) {
      case QuantizerType::None:
        throw Error("search: use_quantized requested but no quantizer is trained");
      case QuantizerType::SQ8: {
        const auto qs = ix.sq.prepare(q, ix.cfg.simd);
        found = run(Sq8Distance{&qs, &ix});
        break;
      }
      case QuantizerType::PQ: {
        const auto qs = ix.pq.prepare(q);
        found = run(PqDistance{&qs, &ix});
        break;
      }
    }
    // Re-rank the best approximate candidates with exact distances.
    if (params.rerank != 0) {
      const std::size_t r = std::min(found.size(), std::max(params.rerank, params.k));
      found.resize(r);
      for (Candidate& c : found) c.dist = ix.dist(q, ix.vec(c.id), ix.dim);
      stats.distance_computations += r;
      std::sort(found.begin(), found.end(),
                [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
    }
  }

  if (found.size() > params.k) found.resize(params.k);
  std::vector<SearchResult> out;
  out.reserve(found.size());
  for (const Candidate& c : found) out.push_back({ix.labels[c.id], c.dist});
  if (params.stats != nullptr) *params.stats = stats;
  return out;
}

std::vector<std::vector<SearchResult>> HnswIndex::search_batch(std::span<const float> queries,
                                                               const SearchParams& params,
                                                               std::size_t num_threads) const {
  const std::size_t d = impl_->dim;
  if (queries.size() % d != 0) throw Error("search_batch: query matrix size not a multiple of dim");
  const std::size_t nq = queries.size() / d;
  std::vector<std::vector<SearchResult>> out(nq);
  SearchParams p = params;
  p.stats = nullptr;
  parallel_for(
      0, nq, num_threads, [&](std::size_t i) { out[i] = search(queries.subspan(i * d, d), p); }, 4);
  return out;
}

void HnswIndex::reserve(std::size_t new_capacity) {
  Impl& ix = *impl_;
  if (new_capacity <= ix.capacity && !ix.read_only) return;
  ix.grow_owned(new_capacity);
}

void HnswIndex::train_quantizer(const QuantizerConfig& qc, std::size_t num_threads) {
  Impl& ix = *impl_;
  const std::size_t n = ix.count.load(std::memory_order_acquire);
  if (qc.type == QuantizerType::None) {
    ix.qtype = QuantizerType::None;
    ix.code_size = 0;
    ix.codes_owned = {};
    ix.codes = nullptr;
    return;
  }
  if (n == 0) throw Error("train_quantizer: index is empty");

  // Uniform sample without replacement (partial Fisher-Yates).
  std::vector<node_id> ids(n);
  std::iota(ids.begin(), ids.end(), node_id{0});
  const std::size_t ns = std::min(n, std::max<std::size_t>(qc.train_size, 1));
  std::mt19937_64 rng(qc.seed);
  for (std::size_t i = 0; i < ns; ++i) {
    std::uniform_int_distribution<std::size_t> pick(i, n - 1);
    std::swap(ids[i], ids[pick(rng)]);
  }
  std::vector<float> sample(ns * ix.dim);
  for (std::size_t i = 0; i < ns; ++i) {
    std::memcpy(sample.data() + i * ix.dim, ix.vec(ids[i]), ix.dim * sizeof(float));
  }

  if (qc.type == QuantizerType::SQ8) {
    ix.sq = ScalarQuantizer::train(sample, ix.dim, ix.cfg.metric);
    ix.code_size = ix.sq.code_size();
  } else {
    ix.pq = ProductQuantizer::train(sample, ix.dim, ix.cfg.metric, qc, num_threads);
    ix.code_size = ix.pq.code_size();
  }
  ix.codes_owned = AlignedBuffer<std::uint8_t>(ix.capacity * ix.code_size);
  ix.codes = ix.codes_owned.data();
  ix.qtype = qc.type;
  parallel_for(
      0, n, num_threads, [&](std::size_t i) { ix.encode_node(static_cast<node_id>(i)); }, 256);
}

QuantizerType HnswIndex::quantizer_type() const noexcept { return impl_->qtype; }

HnswIndex HnswIndex::compact(std::size_t num_threads) const {
  const Impl& ix = *impl_;
  std::vector<std::pair<label_t, node_id>> live;
  {
    const std::scoped_lock lock(ix.label_mutex);
    live.assign(ix.label_map.begin(), ix.label_map.end());
  }
  // Insert in id order: approximates the original insertion order.
  std::sort(live.begin(), live.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  HnswConfig cfg = ix.cfg;
  cfg.capacity = std::max<std::size_t>(live.size() + live.size() / 4, 1024);
  HnswIndex out(cfg);
  Impl& ox = *out.impl_;
  if (ix.qtype != QuantizerType::None) {  // reuse the trained codebooks
    ox.qtype = ix.qtype;
    ox.sq = ix.sq;
    ox.pq = ix.pq;
    ox.code_size = ix.code_size;
    ox.codes_owned = AlignedBuffer<std::uint8_t>(ox.capacity * ox.code_size);
    ox.codes = ox.codes_owned.data();
  }
  parallel_for(
      0, live.size(), num_threads,
      [&](std::size_t i) {
        const auto [label, id] = live[i];
        out.add(label, std::span<const float>(ix.vec(id), ix.dim), ix.tags[id]);
      },
      16);
  return out;
}

std::size_t HnswIndex::size() const noexcept {
  return impl_->count.load(std::memory_order_relaxed) -
         impl_->num_deleted.load(std::memory_order_relaxed);
}
std::size_t HnswIndex::node_count() const noexcept {
  return impl_->count.load(std::memory_order_relaxed);
}
std::size_t HnswIndex::deleted_count() const noexcept {
  return impl_->num_deleted.load(std::memory_order_relaxed);
}
std::size_t HnswIndex::capacity() const noexcept { return impl_->capacity; }
std::size_t HnswIndex::dim() const noexcept { return impl_->dim; }
Metric HnswIndex::metric() const noexcept { return impl_->cfg.metric; }
const HnswConfig& HnswIndex::config() const noexcept { return impl_->cfg; }
bool HnswIndex::read_only() const noexcept { return impl_->read_only; }

MemoryUsage HnswIndex::memory_usage() const {
  const Impl& ix = *impl_;
  const std::size_t n = ix.count.load(std::memory_order_acquire);
  MemoryUsage m;
  const bool mapped = ix.mapping.valid();
  const std::size_t vec_bytes = mapped ? n * ix.dim * sizeof(float) : ix.vectors_owned.bytes();
  const std::size_t l0_bytes =
      mapped ? n * ix.level0_stride() * sizeof(std::uint32_t) : ix.level0_owned.bytes();
  std::size_t upper_bytes = ix.upper.capacity() * sizeof(std::unique_ptr<std::uint32_t[]>);
  for (std::size_t i = 0; i < n; ++i) {
    upper_bytes +=
        static_cast<std::size_t>(ix.levels[i]) * ix.upper_stride() * sizeof(std::uint32_t);
  }
  m.vectors = vec_bytes;
  m.graph = l0_bytes + upper_bytes;
  if (ix.qtype != QuantizerType::None) {
    m.codes = mapped && ix.codes_owned.empty() ? n * ix.code_size : ix.codes_owned.bytes();
    m.codes += ix.qtype == QuantizerType::SQ8 ? 2 * ix.dim * sizeof(float)
                                              : ix.pq.subspaces() * ProductQuantizer::kCentroids *
                                                    (ix.dim / ix.pq.subspaces()) * sizeof(float);
  }
  m.metadata = ix.levels.bytes() + ix.labels.bytes() + ix.tags.bytes() + ix.state.bytes() +
               ix.capacity * sizeof(detail::SpinLock);
  {
    const std::scoped_lock lock(ix.label_mutex);
    // libc++/libstdc++ node-based map: bucket array + one heap node per entry.
    m.metadata += ix.label_map.bucket_count() * sizeof(void*) +
                  ix.label_map.size() * (sizeof(std::pair<label_t, node_id>) + 2 * sizeof(void*));
  }
  if (mapped) {
    m.mapped = vec_bytes + l0_bytes + (ix.codes_owned.empty() ? n * ix.code_size : 0);
  }
  return m;
}

IntegrityReport HnswIndex::check_integrity() const {
  const Impl& ix = *impl_;
  IntegrityReport r;
  const std::size_t n = ix.count.load(std::memory_order_acquire);
  r.nodes = n;
  auto fail = [&r](std::string msg) {
    r.ok = false;
    if (r.errors.size() < 32) r.errors.push_back(std::move(msg));
  };
  if (n == 0) return r;

  const std::uint64_t packed = ix.entry.load(std::memory_order_acquire);
  if (packed == kNoEntry) {
    fail("non-empty index has no entry point");
    return r;
  }
  const node_id ep = entry_id(packed);
  if (ep >= n) {
    fail("entry point out of range");
    return r;
  }

  int max_level = 0;
  std::size_t deleted = 0;
  std::size_t degree_sum = 0;
  std::vector<node_id> scratch;
  for (node_id i = 0; i < n; ++i) {
    const int L = ix.level_of(i);
    max_level = std::max(max_level, L);
    if (ix.is_deleted(i)) ++deleted;
    if (L > kMaxLevel) {
      fail("node " + std::to_string(i) + " has invalid level");
      continue;
    }
    for (int l = 0; l <= L; ++l) {
      const std::uint32_t* ll = ix.links(i, l);
      const std::uint32_t c = ll[0];
      if (l == 0) degree_sum += c;
      if (c > ix.max_links(l)) {
        fail("node " + std::to_string(i) + " level " + std::to_string(l) + " exceeds max degree");
        continue;
      }
      scratch.assign(ll + 1, ll + 1 + c);
      for (const node_id x : scratch) {
        if (x >= n) {
          fail("node " + std::to_string(i) + " links to out-of-range id");
        } else if (x == i) {
          fail("node " + std::to_string(i) + " has a self-loop");
        } else if (ix.level_of(x) < l) {
          fail("node " + std::to_string(i) + " links on level " + std::to_string(l) +
               " to a node that does not exist on that level");
        }
      }
      std::sort(scratch.begin(), scratch.end());
      if (std::adjacent_find(scratch.begin(), scratch.end()) != scratch.end()) {
        fail("node " + std::to_string(i) + " has duplicate links");
      }
    }
  }
  r.max_level = static_cast<std::size_t>(max_level);
  r.avg_degree_level0 = static_cast<double>(degree_sum) / static_cast<double>(n);
  if (entry_level(packed) != max_level || ix.level_of(ep) != max_level) {
    fail("entry point is not on the top level");
  }
  if (deleted != ix.num_deleted.load()) fail("tombstone counter mismatch");

  {
    const std::scoped_lock lock(ix.label_mutex);
    if (ix.label_map.size() + deleted != n) fail("label map size does not match live node count");
    for (const auto& [label, id] : ix.label_map) {
      if (id >= n || ix.labels[id] != label || ix.is_deleted(id)) {
        fail("label map entry " + std::to_string(label) + " is inconsistent");
      }
    }
  }

  // Layer-0 reachability from the entry point (tombstones included, since
  // traversal passes through them).
  std::vector<std::uint8_t> seen(n, 0);
  std::vector<node_id> stack{ep};
  seen[ep] = 1;
  std::size_t reached = 0;
  while (!stack.empty()) {
    const node_id u = stack.back();
    stack.pop_back();
    ++reached;
    const std::uint32_t* ll = ix.links(u, 0);
    const std::uint32_t c = std::min<std::uint32_t>(ll[0], static_cast<std::uint32_t>(ix.M0));
    for (std::uint32_t j = 0; j < c; ++j) {
      const node_id v = ll[1 + j];
      if (v < n && seen[v] == 0) {
        seen[v] = 1;
        stack.push_back(v);
      }
    }
  }
  r.reachable_level0 = reached;
  return r;
}

}  // namespace vs
