// Atomic access to plain (non-std::atomic) memory.
//
// Graph storage is kept as plain uint32_t/uint8_t arrays so it can be written
// to disk and mmap'd back verbatim; concurrent access goes through these
// helpers. They use std::atomic_ref where the standard library provides it
// (libc++ >= 19, libstdc++ >= 10) and the equivalent GCC/Clang __atomic
// builtins otherwise, so older Apple toolchains still build.
#pragma once

#include <atomic>
#include <type_traits>

namespace vs::detail {

#ifndef VS_USE_ATOMIC_REF
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
#define VS_USE_ATOMIC_REF 1
#else
#define VS_USE_ATOMIC_REF 0
#endif
#endif

template <class T>
  requires std::is_integral_v<T>
[[nodiscard]] inline T load_acquire(const T& x) noexcept {
#if VS_USE_ATOMIC_REF
  return std::atomic_ref<T>(const_cast<T&>(x)).load(std::memory_order_acquire);
#else
  return __atomic_load_n(&x, __ATOMIC_ACQUIRE);
#endif
}

template <class T>
  requires std::is_integral_v<T>
[[nodiscard]] inline T load_relaxed(const T& x) noexcept {
#if VS_USE_ATOMIC_REF
  return std::atomic_ref<T>(const_cast<T&>(x)).load(std::memory_order_relaxed);
#else
  return __atomic_load_n(&x, __ATOMIC_RELAXED);
#endif
}

template <class T>
  requires std::is_integral_v<T>
inline void store_release(T& x, T v) noexcept {
#if VS_USE_ATOMIC_REF
  std::atomic_ref<T>(x).store(v, std::memory_order_release);
#else
  __atomic_store_n(&x, v, __ATOMIC_RELEASE);
#endif
}

template <class T>
  requires std::is_integral_v<T>
inline T fetch_or_acq_rel(T& x, T v) noexcept {
#if VS_USE_ATOMIC_REF
  return std::atomic_ref<T>(x).fetch_or(v, std::memory_order_acq_rel);
#else
  return __atomic_fetch_or(&x, v, __ATOMIC_ACQ_REL);
#endif
}

}  // namespace vs::detail
