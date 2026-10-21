cmake_minimum_required(VERSION 3.25)

project(vecsearch
  VERSION 0.1.0
  DESCRIPTION "HNSW vector search engine with SIMD kernels, quantization, mmap persistence and an HTTP API"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VS_BUILD_TESTS      "Build GoogleTest unit tests"                  ON)
option(VS_BUILD_BENCHMARKS "Build benchmark drivers"                      ON)
option(VS_BUILD_SERVER     "Build the HTTP server and CLI"               ON)
option(VS_BUILD_PYTHON     "Build pybind11 Python bindings"               OFF)
option(VS_NATIVE           "Tune for the host CPU (-march/-mcpu=native)"  ON)
option(VS_WERROR           "Treat warnings as errors"                     ON)
set(VS_SANITIZE "" CACHE STRING "Sanitizer set: '', 'address,undefined', or 'thread'")

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(vecsearch
  src/distance.cpp
  src/distance_scalar.cpp
  src/distance_neon.cpp
  src/distance_avx2.cpp
  src/flat_index.cpp
  src/hnsw_index.cpp
  src/hnsw_persistence.cpp
  src/scalar_quantizer.cpp
  src/product_quantizer.cpp
  src/mmap_file.cpp
  src/io.cpp
  src/system.cpp)
add_library(vecsearch::vecsearch ALIAS vecsearch)
target_include_directories(vecsearch PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(vecsearch
  PUBLIC  vs_sanitizers
  PRIVATE vs_warnings vs_optimize)
find_package(Threads REQUIRED)
target_link_libraries(vecsearch PUBLIC Threads::Threads)
set_target_properties(vecsearch PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Server library, server binary and CLI
# ---------------------------------------------------------------------------
if(VS_BUILD_SERVER)
  add_library(vecsearch_server src/server/http_server.cpp)
  target_include_directories(vecsearch_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(vecsearch_server
    PUBLIC  vecsearch httplib::httplib nlohmann_json::nlohmann_json
    PRIVATE vs_warnings vs_optimize)

  add_executable(vsearch apps/vsearch_cli.cpp)
  target_link_libraries(vsearch PRIVATE vecsearch vs_warnings vs_optimize)

  add_executable(vsearch-server apps/vsearch_server.cpp)
  target_link_libraries(vsearch-server PRIVATE vecsearch_server vs_warnings vs_optimize)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(VS_BUILD_BENCHMARKS)
  add_executable(ann_bench bench/ann_bench.cpp)
  target_link_libraries(ann_bench PRIVATE vecsearch nlohmann_json::nlohmann_json vs_warnings vs_optimize)

  add_executable(distance_bench bench/distance_bench.cpp)
  target_link_libraries(distance_bench PRIVATE vecsearch benchmark::benchmark vs_warnings vs_optimize)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(VS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(VS_BUILD_PYTHON)
  add_subdirectory(python)
endif()
