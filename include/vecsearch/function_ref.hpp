// A minimal non-owning type-erased callable reference (C++26 std::function_ref
// in spirit). Costs one indirect call and never allocates.
//
// Like std::string_view it does not own the callable: bind it to a named
// object that outlives every call, never to a temporary lambda.
#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace vs {

template <class Signature>
class FunctionRef;

template <class R, class... Args>
class FunctionRef<R(Args...)> {
 public:
  constexpr FunctionRef() noexcept = default;

  template <class F>
    requires(!std::same_as<std::remove_cvref_t<F>, FunctionRef> &&
             std::is_invocable_r_v<R, F&, Args...>)
  // NOLINTNEXTLINE(google-explicit-constructor): implicit by design, like std::function_ref
  constexpr FunctionRef(F&& f) noexcept
      : obj_(const_cast<void*>(static_cast<const void*>(std::addressof(f)))),
        call_([](void* obj, Args... args) -> R {
          return std::invoke(*static_cast<std::add_pointer_t<F>>(obj), std::forward<Args>(args)...);
        }) {}

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return call_ != nullptr; }

  R operator()(Args... args) const { return call_(obj_, std::forward<Args>(args)...); }

 private:
  void* obj_ = nullptr;
  R (*call_)(void*, Args...) = nullptr;
};

}  // namespace vs
