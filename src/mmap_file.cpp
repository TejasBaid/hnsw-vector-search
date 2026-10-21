#include "vecsearch/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include "vecsearch/types.hpp"

namespace vs {

namespace {
class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) noexcept : fd_(fd) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  ~FileDescriptor() {
    if (fd_ >= 0) ::close(fd_);
  }
  [[nodiscard]] int get() const noexcept { return fd_; }

 private:
  int fd_;
};

[[noreturn]] void throw_errno(const std::string& what, const std::filesystem::path& path) {
  throw Error(what + " '" + path.string() + "': " + std::strerror(errno));
}
}  // namespace

MappedFile::MappedFile(const std::filesystem::path& path) {
  const FileDescriptor fd(::open(path.c_str(), O_RDONLY));
  if (fd.get() < 0) throw_errno("cannot open", path);
  struct stat st{};
  if (::fstat(fd.get(), &st) != 0) throw_errno("cannot stat", path);
  if (st.st_size <= 0) throw Error("cannot map empty file '" + path.string() + "'");
  size_ = static_cast<std::size_t>(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd.get(), 0);
  if (p == MAP_FAILED) {
    size_ = 0;
    throw_errno("cannot mmap", path);
  }
  data_ = static_cast<std::byte*>(p);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

MappedFile::~MappedFile() { reset(); }

void MappedFile::reset() noexcept {
  if (data_ != nullptr) ::munmap(data_, size_);
  data_ = nullptr;
  size_ = 0;
}

void MappedFile::advise_willneed() const noexcept {
  if (data_ != nullptr) ::madvise(data_, size_, MADV_WILLNEED);
}

}  // namespace vs
