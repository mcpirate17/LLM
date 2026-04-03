cmake_minimum_required(VERSION 3.16)
project(aria_runtime_tests C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ── aria_core source tree (sibling project) ──────────────────────────
set(ARIA_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../aria_core"
    CACHE PATH "Path to aria_core project root")

if(NOT EXISTS "${ARIA_CORE_ROOT}/include/kernels.h")
    message(FATAL_ERROR
        "aria_core not found at ${ARIA_CORE_ROOT}\n"
        "Set -DARIA_CORE_ROOT=/path/to/aria_core if it lives elsewhere.")
endif()

set(ARIA_INCLUDE_DIR "${ARIA_CORE_ROOT}/include")
set(ARIA_CPU_SRC_DIR "${ARIA_CORE_ROOT}/src/cpu")

# ── Compiler flags ───────────────────────────────────────────────────
add_compile_options(-O2 -march=native -Wall -Wextra -Wno-unused-parameter)

# ── Kernel library (static, built from aria_core CPU sources) ────────
# kernels.cpp is a unity build that #includes all other .cpp/.c files.
add_library(aria_kernels STATIC
    "${ARIA_CPU_SRC_DIR}/kernels.cpp"
    "${ARIA_CPU_SRC_DIR}/graph_validator.cpp"
    "${ARIA_CPU_SRC_DIR}/shape_inference.cpp"
    "${ARIA_CPU_SRC_DIR}/clifford.cpp"
    "${ARIA_CPU_SRC_DIR}/hyperbolic.cpp"
)
target_include_directories(aria_kernels PUBLIC
    "${ARIA_INCLUDE_DIR}"
    "${ARIA_CPU_SRC_DIR}"
)
# Disable OpenMP for test builds to keep dependencies minimal.
# (The kernels fall back to serial paths without -DARIA_HAS_OPENMP.)
target_compile_definitions(aria_kernels PRIVATE)

# ── test_kernels ─────────────────────────────────────────────────────
add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE aria_kernels m)
target_include_directories(test_kernels PRIVATE "${ARIA_INCLUDE_DIR}")

# ── test_runtime ─────────────────────────────────────────────────────
add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE aria_kernels m)
target_include_directories(test_runtime PRIVATE "${ARIA_INCLUDE_DIR}")

# ── CTest integration ───────────────────────────────────────────────
enable_testing()
add_test(NAME kernels COMMAND test_kernels)
add_test(NAME runtime COMMAND test_runtime)
