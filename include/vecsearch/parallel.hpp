// Minimal fork-join helper: dynamic work distribution over a fixed set of
// threads with exception propagation.
#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vs {

/// Number of hardware threads (at least 1).
[[nodiscard]] inline std::size_t hardware_threads() noexcept {
  return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

/// Calls fn(i) for every i in [begin, end) using up to `num_threads` threads
/// (0 = all hardware threads). Work is handed out in chunks of `grain`
/// indices from a shared atomic cursor, so uneven per-item cost balances
/// itself. The first exception thrown by any task is rethrown on the caller.
template <std::invocable<std::size_t> Fn>
void parallel_for(std::size_t begin, std::size_t end, std::size_t num_threads, Fn&& fn,
                  std::size_t grain = 64) {
  if (begin >= end) return;
  if (num_threads == 0) num_threads = hardware_threads();
  grain = std::max<std::size_t>(1, grain);
  const std::size_t chunks = (end - begin + grain - 1) / grain;
  num_threads = std::min(num_threads, chunks);

  if (num_threads <= 1) {
    for (std::size_t i = begin; i < end; ++i) fn(i);
    return;
  }

  std::atomic<std::size_t> cursor{begin};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;

  auto worker = [&] {
    while (!failed.load(std::memory_order_relaxed)) {
      const std::size_t start = cursor.fetch_add(grain, std::memory_order_relaxed);
      if (start >= end) return;
      const std::size_t stop = std::min(end, start + grain);
      try {
        for (std::size_t i = start; i < stop; ++i) fn(i);
      } catch (...) {
        const std::scoped_lock lock(error_mutex);
        if (!error) error = std::current_exception();
        failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  };

  {
    // Joins on scope exit, including when thread creation itself throws.
    struct Joiner {
      std::vector<std::thread> threads;
      ~Joiner() {
        for (auto& t : threads) t.join();
      }
    } pool;
    pool.threads.reserve(num_threads - 1);
    for (std::size_t t = 0; t + 1 < num_threads; ++t) pool.threads.emplace_back(worker);
    worker();  // the calling thread participates
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace vs
