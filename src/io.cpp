#include "vecsearch/io.hpp"

#include <cstring>
#include <fstream>
#include <string>

#include "vecsearch/mmap_file.hpp"
#include "vecsearch/types.hpp"

namespace vs {

namespace {

template <class T>
Matrix<T> read_vecs(const std::filesystem::path& path, std::size_t max_rows) {
  static_assert(sizeof(T) == 4);
  const MappedFile file(path);
  const std::byte* p = file.data();
  const std::size_t size = file.size();
  if (size < 4) throw Error("vecs file too small: " + path.string());

  std::int32_t dim = 0;
  std::memcpy(&dim, p, 4);
  if (dim <= 0 || dim > (1 << 20)) throw Error("bad dimension in " + path.string());
  const auto d = static_cast<std::size_t>(dim);
  const std::size_t row_bytes = 4 + d * 4;
  if (size % row_bytes != 0) throw Error("truncated vecs file: " + path.string());
  std::size_t rows = size / row_bytes;
  if (max_rows != 0 && max_rows < rows) rows = max_rows;

  Matrix<T> m(rows, d);
  for (std::size_t i = 0; i < rows; ++i) {
    const std::byte* row = p + i * row_bytes;
    std::int32_t row_dim = 0;
    std::memcpy(&row_dim, row, 4);
    if (row_dim != dim) throw Error("inconsistent row dimension in " + path.string());
    std::memcpy(m.data.data() + i * d, row + 4, d * 4);
  }
  return m;
}

template <class T>
void write_vecs(const std::filesystem::path& path, const Matrix<T>& m) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw Error("cannot open for writing: " + path.string());
  const auto dim = static_cast<std::int32_t>(m.cols);
  for (std::size_t i = 0; i < m.rows; ++i) {
    out.write(reinterpret_cast<const char*>(&dim), 4);
    out.write(reinterpret_cast<const char*>(m.data.data() + i * m.cols),
              static_cast<std::streamsize>(m.cols * sizeof(T)));
  }
  if (!out) throw Error("write failed: " + path.string());
}

}  // namespace

Matrix<float> read_fvecs(const std::filesystem::path& path, std::size_t max_rows) {
  return read_vecs<float>(path, max_rows);
}
Matrix<std::int32_t> read_ivecs(const std::filesystem::path& path, std::size_t max_rows) {
  return read_vecs<std::int32_t>(path, max_rows);
}
void write_fvecs(const std::filesystem::path& path, const Matrix<float>& m) { write_vecs(path, m); }
void write_ivecs(const std::filesystem::path& path, const Matrix<std::int32_t>& m) {
  write_vecs(path, m);
}

}  // namespace vs
