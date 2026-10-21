# Python bindings (optional): cmake -DVS_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
FetchContent_Declare(pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(vecsearch_python bindings.cpp)
set_target_properties(vecsearch_python PROPERTIES OUTPUT_NAME vecsearch)
target_link_libraries(vecsearch_python PRIVATE vecsearch vs_warnings vs_optimize)
