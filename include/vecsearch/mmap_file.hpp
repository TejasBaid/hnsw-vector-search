// RAII wrapper over a read-only, private memory mapping of a whole file.
#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace vs {

class MappedFile {
 public:
  MappedFile() noexcept = default;
  /// Maps `path` copy-on-write (PROT_READ|PROT_WRITE, MAP_PRIVATE): the index
  /// never writes to mapped pages, but non-const pointers keep the in-memory
  /// and mapped code paths identical. Throws vs::Error on failure.
  explicit MappedFile(const std::filesystem::path& path);
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  [[nodiscard]] std::byte* data() noexcept { return data_; }
  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }

  /// Hint the kernel to fault the whole mapping in ahead of use.
  void advise_willneed() const noexcept;

 private:
  void reset() noexcept;

  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace vs
