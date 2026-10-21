#include <algorithm>
#include <cmath>
#include <limits>

#include "vecsearch/detail/byte_stream.hpp"
#include "vecsearch/quantization.hpp"

namespace vs {

ScalarQuantizer ScalarQuantizer::train(std::span<const float> data, std::size_t dim,
                                       Metric metric) {
  if (dim == 0 || data.size() < dim || data.size() % dim != 0) {
    throw Error("ScalarQuantizer::train: data size must be a positive multiple of dim");
  }
  ScalarQuantizer q;
  q.metric_ = metric;
  q.vmin_.assign(dim, std::numeric_limits<float>::max());
  std::vector<float> vmax(dim, std::numeric_limits<float>::lowest());
  const std::size_t n = data.size() / dim;
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = data.data() + i * dim;
    for (std::size_t j = 0; j < dim; ++j) {
      q.vmin_[j] = std::min(q.vmin_[j], x[j]);
      vmax[j] = std::max(vmax[j], x[j]);
    }
  }
  q.step_.resize(dim);
  for (std::size_t j = 0; j < dim; ++j) q.step_[j] = (vmax[j] - q.vmin_[j]) / 255.0F;
  return q;
}

void ScalarQuantizer::encode(const float* x, std::uint8_t* code) const noexcept {
  for (std::size_t j = 0; j < vmin_.size(); ++j) {
    if (step_[j] <= 0.0F) {
      code[j] = 0;
      continue;
    }
    const float v = std::nearbyint((x[j] - vmin_[j]) / step_[j]);
    code[j] = static_cast<std::uint8_t>(std::clamp(v, 0.0F, 255.0F));
  }
}

void ScalarQuantizer::decode(const std::uint8_t* code, float* x) const noexcept {
  for (std::size_t j = 0; j < vmin_.size(); ++j) {
    x[j] = vmin_[j] + static_cast<float>(code[j]) * step_[j];
  }
}

ScalarQuantizer::Query ScalarQuantizer::prepare(const float* query, SimdBackend backend) const {
  const std::size_t d = vmin_.size();
  Query st;
  st.dim_ = d;
  st.kernels_ = &kernels(backend);
  st.l2_ = metric_ == Metric::L2;
  st.a_ = AlignedBuffer<float>(d);
  double bias = 0.0;
  if (st.l2_) {
    // |q - (vmin + c*step)|^2 = sum step^2 * ((q - vmin)/step - c)^2
    st.b_ = AlignedBuffer<float>(d);
    for (std::size_t j = 0; j < d; ++j) {
      const float centered = query[j] - vmin_[j];
      if (step_[j] > 0.0F) {
        st.a_[j] = centered / step_[j];
        st.b_[j] = step_[j] * step_[j];
      } else {  // constant dimension: contributes a fixed offset
        bias += static_cast<double>(centered) * static_cast<double>(centered);
      }
    }
  } else {
    // <q, vmin + c*step> = <q, vmin> + sum (q*step) * c
    for (std::size_t j = 0; j < d; ++j) {
      st.a_[j] = query[j] * step_[j];
      bias += static_cast<double>(query[j]) * static_cast<double>(vmin_[j]);
    }
  }
  st.bias_ = static_cast<float>(bias);
  return st;
}

void ScalarQuantizer::serialize(std::vector<std::byte>& out) const {
  detail::ByteWriter w;
  w.put(static_cast<std::uint32_t>(metric_));
  w.put(static_cast<std::uint64_t>(vmin_.size()));
  w.put_array(std::span<const float>(vmin_));
  w.put_array(std::span<const float>(step_));
  const auto& b = w.bytes();
  out.insert(out.end(), b.begin(), b.end());
}

ScalarQuantizer ScalarQuantizer::deserialize(std::span<const std::byte> in) {
  detail::ByteReader r(in);
  ScalarQuantizer q;
  const auto metric = r.get<std::uint32_t>();
  if (metric > 2) throw FormatError("SQ8: bad metric");
  q.metric_ = static_cast<Metric>(metric);
  const auto dim = r.get<std::uint64_t>();
  if (dim == 0 || dim > (1U << 20)) throw FormatError("SQ8: bad dimension");
  q.vmin_.resize(dim);
  q.step_.resize(dim);
  r.get_array(std::span<float>(q.vmin_));
  r.get_array(std::span<float>(q.step_));
  return q;
}

}  // namespace vs
