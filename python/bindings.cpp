// pybind11 bindings: a small, NumPy-friendly surface over HnswIndex.
//
//   import numpy as np, vecsearch
//   index = vecsearch.HnswIndex(dim=128, metric="l2", M=16, ef_construction=200,
//                               capacity=1_000_000)
//   index.add(vectors, labels)                      # parallel build, GIL released
//   labels, distances = index.search(queries, k=10, ef=64)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <string>

#include "vecsearch/hnsw_index.hpp"

namespace py = pybind11;
using vs::HnswIndex;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>;

std::span<const float> rows(const FloatArray& a, std::size_t dim, std::size_t& n) {
  if (a.ndim() != 2 || static_cast<std::size_t>(a.shape(1)) != dim) {
    throw std::invalid_argument("expected a 2-D float array with " + std::to_string(dim) +
                                " columns");
  }
  n = static_cast<std::size_t>(a.shape(0));
  return {a.data(), n * dim};
}

}  // namespace

PYBIND11_MODULE(vecsearch, m) {
  m.doc() = "HNSW approximate nearest-neighbour search (C++20, SIMD, mmap persistence)";
  py::register_exception<vs::Error>(m, "Error");

  py::class_<HnswIndex>(m, "HnswIndex")
      .def(py::init([](std::size_t dim, const std::string& metric, std::size_t M,
                       std::size_t ef_construction, std::size_t capacity, std::uint64_t seed) {
             vs::HnswConfig cfg;
             cfg.dim = dim;
             cfg.metric = vs::parse_metric(metric);
             cfg.M = M;
             cfg.ef_construction = ef_construction;
             cfg.capacity = capacity;
             cfg.seed = seed;
             return HnswIndex(cfg);
           }),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("capacity") = 1024, py::arg("seed") = 0x5eed)
      .def(
          "add",
          [](HnswIndex& self, const FloatArray& vectors, const LabelArray& labels,
             std::size_t num_threads) {
            std::size_t n = 0;
            const auto data = rows(vectors, self.dim(), n);
            if (labels.ndim() != 1 || static_cast<std::size_t>(labels.shape(0)) != n) {
              throw std::invalid_argument("labels must be a 1-D array with one entry per vector");
            }
            if (self.node_count() + n > self.capacity()) self.reserve(self.node_count() + n);
            const std::span<const vs::label_t> l(labels.data(), n);
            const py::gil_scoped_release release;
            self.add_batch(data, l, num_threads);
          },
          py::arg("vectors"), py::arg("labels"), py::arg("num_threads") = 0)
      .def(
          "search",
          [](const HnswIndex& self, const FloatArray& queries, std::size_t k, std::size_t ef,
             std::size_t num_threads) {
            std::size_t nq = 0;
            const auto q = rows(queries, self.dim(), nq);
            std::vector<std::vector<vs::SearchResult>> res;
            {
              const py::gil_scoped_release release;
              res = self.search_batch(q, {.k = k, .ef = ef}, num_threads);
            }
            py::array_t<std::int64_t> labels({nq, k});
            py::array_t<float> dists({nq, k});
            auto L = labels.mutable_unchecked<2>();
            auto D = dists.mutable_unchecked<2>();
            for (std::size_t i = 0; i < nq; ++i) {
              for (std::size_t j = 0; j < k; ++j) {
                const bool have = j < res[i].size();
                L(i, j) = have ? static_cast<std::int64_t>(res[i][j].label) : -1;
                D(i, j) = have ? res[i][j].distance : std::numeric_limits<float>::infinity();
              }
            }
            return py::make_tuple(labels, dists);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("num_threads") = 0)
      .def("remove", &HnswIndex::remove, py::arg("label"))
      .def("contains", &HnswIndex::contains, py::arg("label"))
      .def("save", [](const HnswIndex& self, const std::string& path) { self.save(path); })
      .def_static(
          "load",
          [](const std::string& path, bool mmap) {
            return HnswIndex::load(path, mmap ? vs::LoadMode::Mmap : vs::LoadMode::Copy);
          },
          py::arg("path"), py::arg("mmap") = false)
      .def("compact", [](const HnswIndex& self) { return self.compact(); })
      .def("__len__", &HnswIndex::size)
      .def_property_readonly("dim", &HnswIndex::dim)
      .def_property_readonly("deleted_count", &HnswIndex::deleted_count)
      .def_property_readonly("capacity", &HnswIndex::capacity);
}
