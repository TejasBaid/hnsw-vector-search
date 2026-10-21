// Core vocabulary types shared by every index implementation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vs {

/// External, user-supplied identifier of a vector.
using label_t = std::uint64_t;
/// Dense internal node id (index into the graph / vector arrays).
using node_id = std::uint32_t;

inline constexpr node_id kInvalidNode = std::numeric_limits<node_id>::max();

/// Distance metric. All distances follow "smaller is closer":
///   L2           -> squared Euclidean distance
///   InnerProduct -> 1 - <a, b>
///   Cosine       -> 1 - <a, b> / (|a| |b|)   (vectors are normalised on insert)
enum class Metric : std::uint32_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

[[nodiscard]] constexpr std::string_view to_string(Metric m) noexcept {
  switch (m) {
    case Metric::L2: return "l2";
    case Metric::InnerProduct: return "ip";
    case Metric::Cosine: return "cosine";
  }
  return "unknown";
}

/// Parses "l2" | "euclidean" | "ip" | "dot" | "cosine" | "angular".
[[nodiscard]] Metric parse_metric(std::string_view name);

struct SearchResult {
  label_t label = 0;
  float distance = 0.0F;

  friend constexpr bool operator==(const SearchResult&, const SearchResult&) = default;
};

/// Bitmask metadata filter evaluated during graph traversal. Every node carries
/// a 64-bit tag mask; a node passes when
///   (any_of == 0 || tags & any_of) && (tags & all_of) == all_of && !(tags & none_of)
struct TagFilter {
  std::uint64_t any_of = 0;
  std::uint64_t all_of = 0;
  std::uint64_t none_of = 0;

  [[nodiscard]] constexpr bool active() const noexcept { return (any_of | all_of | none_of) != 0; }
  [[nodiscard]] constexpr bool matches(std::uint64_t tags) const noexcept {
    return (any_of == 0 || (tags & any_of) != 0) && (tags & all_of) == all_of &&
           (tags & none_of) == 0;
  }
};

/// Base class for all errors thrown by the library.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Thrown when an insert would exceed the index capacity (call reserve()).
class CapacityError : public Error {
 public:
  using Error::Error;
};

/// Thrown when an on-disk index file is malformed, truncated or from an
/// unsupported format version.
class FormatError : public Error {
 public:
  using Error::Error;
};

}  // namespace vs
