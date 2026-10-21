// Dataset I/O for the TEXMEX .fvecs / .ivecs formats used by SIFT/GIST and
// the ann-benchmarks conversions (each row: int32 dim, then dim values).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include "vecsearch/aligned_buffer.hpp"

namespace vs {

/// Dense row-major matrix.
template <class T>
struct Matrix {
  AlignedBuffer<T> data;
  std::size_t rows = 0;
  std::size_t cols = 0;

  Matrix() = default;
  Matrix(std::size_t r, std::size_t c) : data(r * c), rows(r), cols(c) {}

  [[nodiscard]] std::span<const T> row(std::size_t i) const noexcept {
    return {data.data() + i * cols, cols};
  }
  [[nodiscard]] std::span<T> row(std::size_t i) noexcept { return {data.data() + i * cols, cols}; }
};

/// Reads up to `max_rows` rows (0 = all). Throws vs::Error on malformed input.
[[nodiscard]] Matrix<float> read_fvecs(const std::filesystem::path& path, std::size_t max_rows = 0);
[[nodiscard]] Matrix<std::int32_t> read_ivecs(const std::filesystem::path& path,
                                              std::size_t max_rows = 0);
void write_fvecs(const std::filesystem::path& path, const Matrix<float>& m);
void write_ivecs(const std::filesystem::path& path, const Matrix<std::int32_t>& m);

}  // namespace vs
