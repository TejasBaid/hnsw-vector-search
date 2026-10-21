// Epoch-stamped visited sets, pooled across queries.
//
// Clearing an O(N) bitmap per query would dominate search cost for large N.
// Instead each slot stores the epoch in which it was last visited; starting a
// new query is just ++epoch. The array is only memset when the 16-bit epoch
// wraps around (once every 65535 queries per list).
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "vecsearch/aligned_buffer.hpp"

namespace vs::detail {

class VisitedList {
 public:
  explicit VisitedList(std::size_t capacity) : marks_(capacity) {}

  /// Starts a new traversal; all slots become unvisited in O(1) amortised.
  void reset() noexcept {
    if (++epoch_ == 0) {
      for (std::size_t i = 0; i < marks_.size(); ++i) marks_[i] = 0;
      epoch_ = 1;
    }
  }

  [[nodiscard]] bool visited(std::uint32_t id) const noexcept { return marks_[id] == epoch_; }
  void mark(std::uint32_t id) noexcept { marks_[id] = epoch_; }
  /// Marks `id` and returns whether it had already been visited.
  [[nodiscard]] bool test_and_mark(std::uint32_t id) noexcept {
    const bool seen = marks_[id] == epoch_;
    marks_[id] = epoch_;
    return seen;
  }
  [[nodiscard]] const std::uint16_t* address(std::uint32_t id) const noexcept {
    return marks_.data() + id;
  }
  [[nodiscard]] std::size_t capacity() const noexcept { return marks_.size(); }

 private:
  AlignedBuffer<std::uint16_t> marks_;
  std::uint16_t epoch_ = 0;
};

class VisitedPool {
 public:
  explicit VisitedPool(std::size_t capacity) : capacity_(capacity) {}

  /// RAII lease of a VisitedList; returns it to the pool on destruction.
  class Lease {
   public:
    Lease(VisitedPool& pool, std::unique_ptr<VisitedList> list) noexcept
        : pool_(&pool), list_(std::move(list)) {}
    Lease(Lease&&) noexcept = default;
    Lease& operator=(Lease&&) noexcept = delete;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() {
      if (list_) pool_->release(std::move(list_));
    }
    [[nodiscard]] VisitedList& operator*() const noexcept { return *list_; }
    [[nodiscard]] VisitedList* operator->() const noexcept { return list_.get(); }

   private:
    VisitedPool* pool_;
    std::unique_ptr<VisitedList> list_;
  };

  [[nodiscard]] Lease acquire() {
    std::unique_ptr<VisitedList> list;
    {
      const std::scoped_lock lock(mutex_);
      if (!free_.empty()) {
        list = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (!list) list = std::make_unique<VisitedList>(capacity_);
    list->reset();
    return Lease(*this, std::move(list));
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  void release(std::unique_ptr<VisitedList> list) {
    const std::scoped_lock lock(mutex_);
    free_.push_back(std::move(list));
  }

  std::size_t capacity_;
  std::mutex mutex_;
  std::vector<std::unique_ptr<VisitedList>> free_;
};

}  // namespace vs::detail
