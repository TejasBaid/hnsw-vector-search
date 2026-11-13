// swift-tools-version: 6.0
// SwiftPM manifest exposing only the core library (no server, tests or
// benchmarks), so Swift/C++ apps can depend on it with `.package(path:)`.
// NEON is always present on arm64, so compile-time kernel dispatch needs no
// -mcpu=native here.
import PackageDescription

let package = Package(
  name: "vecsearch",
  platforms: [.macOS(.v14)],
  products: [
    .library(name: "VecSearch", targets: ["VecSearch"])
  ],
  targets: [
    .target(
      name: "VecSearch",
      path: ".",
      sources: [
        "src/distance.cpp",
        "src/distance_scalar.cpp",
        "src/distance_neon.cpp",
        "src/distance_avx2.cpp",
        "src/flat_index.cpp",
        "src/hnsw_index.cpp",
        "src/hnsw_persistence.cpp",
        "src/scalar_quantizer.cpp",
        "src/product_quantizer.cpp",
        "src/mmap_file.cpp",
        "src/io.cpp",
        "src/system.cpp",
      ],
      publicHeadersPath: "include",
      cxxSettings: [.unsafeFlags(["-O3"])]
    )
  ],
  cxxLanguageStandard: .cxx20
)
