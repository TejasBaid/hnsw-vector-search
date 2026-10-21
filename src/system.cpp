#include "vecsearch/system.hpp"

#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

namespace vs {

std::size_t current_rss_bytes() noexcept {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): Mach API signature
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::size_t>(info.resident_size);
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  std::size_t pages_total = 0;
  std::size_t pages_resident = 0;
  if (!(statm >> pages_total >> pages_resident)) return 0;
  return pages_resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

std::size_t memory_footprint_bytes() noexcept {
#if defined(__APPLE__)
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): Mach API signature
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
      KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::size_t>(info.phys_footprint);
#else
  return current_rss_bytes();
#endif
}

std::size_t peak_rss_bytes() noexcept {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);  // bytes on macOS
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;  // KiB on Linux
#endif
}

}  // namespace vs
