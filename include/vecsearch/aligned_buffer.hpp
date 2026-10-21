// Cache-line aligned, fixed-size, owning array for trivially-copyable types.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <type_traits>

namespace vs {

inline constexpr std::size_t kCacheLine = 64;

namespace detail {
struct AlignedDeleter {
  void operator()(void* p) const noexcept { ::operator delete(p, std::align_val_t{kCacheLine}); }
};
}  // namespace detail

template <class T>
  requires std::is_trivially_copyable_v<T>
class AlignedBuffer {
 public:
  AlignedBuffer() noexcept = default;

  /// Allocates `n` zero-initialised elements.
  explicit AlignedBuffer(std::size_t n) : size_(n) {
    if (n == 0) return;
    const std::size_t bytes = round_up(n * sizeof(T));
    data_.reset(static_cast<T*>(::operator new(bytes, std::align_val_t{kCacheLine})));
    std::memset(static_cast<void*>(data_.get()), 0, bytes);
  }

  AlignedBuffer(AlignedBuffer&&) noexcept = default;
  AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  ~AlignedBuffer() = default;

  /// Returns a new buffer of `n` elements holding a copy of the first
  /// min(n, size()) elements; the remainder is zeroed.
  [[nodiscard]] AlignedBuffer resized_copy(std::size_t n) const {
    AlignedBuffer out(n);
    if (size_ != 0 && n != 0) {
      std::memcpy(static_cast<void*>(out.data()), data(), std::min(n, size_) * sizeof(T));
    }
    return out;
  }

  [[nodiscard]] T* data() noexcept { return data_.get(); }
  [[nodiscard]] const T* data() const noexcept { return data_.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
  [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_.get()[i]; }
  [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_.get()[i]; }
  [[nodiscard]] std::span<T> span() noexcept { return {data(), size_}; }
  [[nodiscard]] std::span<const T> span() const noexcept { return {data(), size_}; }

 private:
  static constexpr std::size_t round_up(std::size_t bytes) noexcept {
    return (bytes + kCacheLine - 1) / kCacheLine * kCacheLine;
  }

  std::unique_ptr<T[], detail::AlignedDeleter> data_;
  std::size_t size_ = 0;
};

}  // namespace vs
