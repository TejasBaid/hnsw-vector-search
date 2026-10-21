#include "vecsearch/flat_index.hpp"

#include <algorithm>
#include <queue>

#include "vecsearch/parallel.hpp"

namespace vs {

FlatIndex::FlatIndex(std::size_t dim, Metric metric, SimdBackend backend)
    : dim_(dim), metric_(metric), dist_(distance_function(metric, backend)) {
  if (dim == 0) throw Error("FlatIndex: dimension must be positive");
}

void FlatIndex::add(std::span<const float> vectors, std::span<const label_t> labels,
                    std::span<const std::uint64_t> tags) {
  if (vectors.size() != labels.size() * dim_) {
    throw Error("FlatIndex::add: vectors.size() must equal labels.size() * dim");
  }
  if (!tags.empty() && tags.size() != labels.size()) {
    throw Error("FlatIndex::add: tags must be empty or match labels");
  }
  const std::size_t first = vectors_.size();
  vectors_.insert(vectors_.end(), vectors.begin(), vectors.end());
  if (metric_ == Metric::Cosine) {
    for (std::size_t off = first; off < vectors_.size(); off += dim_) {
      normalize(std::span<float>(vectors_.data() + off, dim_));
    }
  }
  labels_.insert(labels_.end(), labels.begin(), labels.end());
  if (tags.empty()) {
    tags_.resize(labels_.size(), 0);
  } else {
    tags_.insert(tags_.end(), tags.begin(), tags.end());
  }
}

std::vector<SearchResult> FlatIndex::search(std::span<const float> query, std::size_t k,
                                            const TagFilter& filter,
                                            FunctionRef<bool(label_t)> predicate) const {
  if (query.size() != dim_) throw Error("FlatIndex::search: query dimension mismatch");
  std::vector<float> q(query.begin(), query.end());
  if (metric_ == Metric::Cosine) normalize(q);

  // Max-heap of the best k so far; top() is the current k-th distance.
  auto cmp = [](const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> heap(cmp);
  const bool use_filter = filter.active();
  for (std::size_t i = 0; i < labels_.size(); ++i) {
    if (use_filter && !filter.matches(tags_[i])) continue;
    if (predicate && !predicate(labels_[i])) continue;
    const float d = dist_(q.data(), vectors_.data() + i * dim_, dim_);
    if (heap.size() < k) {
      heap.push({labels_[i], d});
    } else if (d < heap.top().distance) {
      heap.pop();
      heap.push({labels_[i], d});
    }
  }
  std::vector<SearchResult> out(heap.size());
  for (std::size_t i = out.size(); i-- > 0;) {
    out[i] = heap.top();
    heap.pop();
  }
  return out;
}

std::vector<std::vector<SearchResult>> FlatIndex::search_batch(std::span<const float> queries,
                                                               std::size_t k,
                                                               std::size_t num_threads,
                                                               const TagFilter& filter) const {
  if (queries.size() % dim_ != 0) throw Error("FlatIndex::search_batch: bad query matrix");
  const std::size_t nq = queries.size() / dim_;
  std::vector<std::vector<SearchResult>> out(nq);
  parallel_for(
      0, nq, num_threads,
      [&](std::size_t i) { out[i] = search(queries.subspan(i * dim_, dim_), k, filter); }, 1);
  return out;
}

}  // namespace vs
