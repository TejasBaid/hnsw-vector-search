# Warning, optimisation and sanitizer settings, exposed as interface targets so
# they apply to first-party code only (third-party deps are built without them).

add_library(vs_warnings INTERFACE)
target_compile_options(vs_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual
  -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough)
if(VS_WERROR)
  target_compile_options(vs_warnings INTERFACE -Werror)
endif()

add_library(vs_optimize INTERFACE)
if(VS_NATIVE)
  include(CheckCXXCompilerFlag)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    check_cxx_compiler_flag("-mcpu=native" VS_HAS_MCPU_NATIVE)
    if(VS_HAS_MCPU_NATIVE)
      target_compile_options(vs_optimize INTERFACE -mcpu=native)
    endif()
  else()
    check_cxx_compiler_flag("-march=native" VS_HAS_MARCH_NATIVE)
    if(VS_HAS_MARCH_NATIVE)
      target_compile_options(vs_optimize INTERFACE -march=native)
    endif()
  endif()
endif()

add_library(vs_sanitizers INTERFACE)
if(VS_SANITIZE)
  message(STATUS "vecsearch: building with -fsanitize=${VS_SANITIZE}")
  target_compile_options(vs_sanitizers INTERFACE
    -fsanitize=${VS_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(vs_sanitizers INTERFACE -fsanitize=${VS_SANITIZE})
endif()
