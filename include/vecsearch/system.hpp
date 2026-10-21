// Small OS helpers used by the CLI and benchmarks: timing and memory usage.
#pragma once

#include <chrono>
#include <cstddef>

namespace vs {

/// Current resident set size of this process in bytes (0 if unavailable).
[[nodiscard]] std::size_t current_rss_bytes() noexcept;
/// Physical memory footprint in bytes: on macOS the kernel's phys_footprint
/// (resident + compressed, what Activity Monitor reports), elsewhere RSS.
[[nodiscard]] std::size_t memory_footprint_bytes() noexcept;
/// Peak resident set size of this process in bytes (0 if unavailable).
[[nodiscard]] std::size_t peak_rss_bytes() noexcept;

class Stopwatch {
 public:
  using clock = std::chrono::steady_clock;
  Stopwatch() noexcept : start_(clock::now()) {}
  void reset() noexcept { start_ = clock::now(); }
  [[nodiscard]] double seconds() const noexcept {
    return std::chrono::duration<double>(clock::now() - start_).count();
  }
  [[nodiscard]] double micros() const noexcept {
    return std::chrono::duration<double, std::micro>(clock::now() - start_).count();
  }

 private:
  clock::time_point start_;
};

}  // namespace vs
