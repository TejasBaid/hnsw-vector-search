// Tiny "--key value" / "--flag" command-line parser shared by the apps.
#pragma once

#include <charconv>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace vs::apps {

class Args {
 public:
  /// `flags` lists the options that take no value.
  Args(int argc, char** argv, int first, std::set<std::string> flags) : flags_(std::move(flags)) {
    for (int i = first; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + a);
      a = a.substr(2);
      if (flags_.contains(a)) {
        values_[a] = "1";
      } else {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for --" + a);
        values_[a] = argv[++i];
      }
    }
  }

  [[nodiscard]] bool has(const std::string& key) const { return values_.contains(key); }

  [[nodiscard]] std::string str(const std::string& key, const std::string& fallback = "") const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  [[nodiscard]] std::string required(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) throw std::invalid_argument("missing required option --" + key);
    return it->second;
  }

  [[nodiscard]] std::size_t size(const std::string& key, std::size_t fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    std::size_t v = 0;
    const auto& s = it->second;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size()) {
      throw std::invalid_argument("--" + key + " expects a non-negative integer");
    }
    return v;
  }

 private:
  std::set<std::string> flags_;
  std::map<std::string, std::string> values_;
};

}  // namespace vs::apps
