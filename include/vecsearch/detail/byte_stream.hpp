// Bounds-checked little-endian (native on all supported targets) binary
// writer/reader used by the persistence layer.
#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vecsearch/types.hpp"

namespace vs::detail {

class ByteWriter {
 public:
  template <class T>
    requires std::is_trivially_copyable_v<T>
  void put(const T& value) {
    const auto* p = reinterpret_cast<const std::byte*>(&value);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }

  template <class T>
    requires std::is_trivially_copyable_v<T>
  void put_array(std::span<const T> values) {
    const auto* p = reinterpret_cast<const std::byte*>(values.data());
    buf_.insert(buf_.end(), p, p + values.size_bytes());
  }

  [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return buf_; }
  [[nodiscard]] std::vector<std::byte> take() noexcept { return std::move(buf_); }

 private:
  std::vector<std::byte> buf_;
};

class ByteReader {
 public:
  explicit ByteReader(std::span<const std::byte> data) noexcept : data_(data) {}

  template <class T>
    requires std::is_trivially_copyable_v<T>
  [[nodiscard]] T get() {
    require(sizeof(T));
    T value;
    std::memcpy(&value, data_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }

  template <class T>
    requires std::is_trivially_copyable_v<T>
  void get_array(std::span<T> out) {
    require(out.size_bytes());
    if (!out.empty()) std::memcpy(out.data(), data_.data() + pos_, out.size_bytes());
    pos_ += out.size_bytes();
  }

  [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - pos_; }

 private:
  void require(std::size_t n) const {
    if (n > data_.size() - pos_) {
      throw FormatError("unexpected end of data (need " + std::to_string(n) + " bytes, have " +
                        std::to_string(data_.size() - pos_) + ")");
    }
  }

  std::span<const std::byte> data_;
  std::size_t pos_ = 0;
};

}  // namespace vs::detail
