// A 4-byte test-and-test-and-set lock used to guard individual graph nodes.
//
// One std::mutex per node costs 40-64 bytes (64 on Darwin), i.e. as much as a
// 16-d float vector; with millions of nodes that is real memory. Critical
// sections here are tiny (copy / rewrite one neighbour list), so a spin lock
// that yields the CPU under sustained contention is the better trade.
#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace vs::detail {

class SpinLock {
 public:
  SpinLock() noexcept = default;
  SpinLock(const SpinLock&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;

  void lock() noexcept {
    for (;;) {
      if (state_.exchange(kLocked, std::memory_order_acquire) == kUnlocked) return;
      // Spin on a plain load (no cache-line ping-pong) for a short while,
      // then start yielding so an oversubscribed machine still makes progress.
      for (int spins = 0; state_.load(std::memory_order_relaxed) == kLocked; ++spins) {
        if (spins > 64) std::this_thread::yield();
      }
    }
  }

  [[nodiscard]] bool try_lock() noexcept {
    return state_.load(std::memory_order_relaxed) == kUnlocked &&
           state_.exchange(kLocked, std::memory_order_acquire) == kUnlocked;
  }

  void unlock() noexcept { state_.store(kUnlocked, std::memory_order_release); }

 private:
  static constexpr std::uint32_t kUnlocked = 0;
  static constexpr std::uint32_t kLocked = 1;
  std::atomic<std::uint32_t> state_{kUnlocked};
};

static_assert(sizeof(SpinLock) == 4);

}  // namespace vs::detail
