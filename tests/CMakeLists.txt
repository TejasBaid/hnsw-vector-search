include(GoogleTest)

add_executable(vecsearch_tests
  test_distance.cpp
  test_flat.cpp
  test_hnsw.cpp
  test_quantization.cpp
  test_persistence.cpp
  test_concurrency.cpp)
target_link_libraries(vecsearch_tests PRIVATE GTest::gtest_main vs_warnings)
if(NOT VS_BUILD_SERVER)
  target_link_libraries(vecsearch_tests PRIVATE vecsearch)
endif()

if(VS_BUILD_SERVER)
  target_sources(vecsearch_tests PRIVATE test_server.cpp)
  target_link_libraries(vecsearch_tests PRIVATE vecsearch_server)  # brings in vecsearch
  target_compile_definitions(vecsearch_tests PRIVATE VS_TEST_SERVER=1)
endif()

gtest_discover_tests(vecsearch_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 600)
