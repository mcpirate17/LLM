cmake_minimum_required(VERSION 3.16)
project(aria_native_kernels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall -Wextra -fPIC")

# Use DT_RPATH (transitive) instead of DT_RUNPATH so bundled OpenBLAS
# can find its own deps (e.g. libgfortran) in the same directory
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")

# Include paths — both local and shared aria-designer kernels
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../aria-designer/runtime/src)

# Local native kernel sources
file(GLOB NATIVE_SRCS src/*.c)

# Shared aria-designer kernel sources (reuse, don't duplicate)
set(DESIGNER_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../aria-designer/runtime/src)
set(DESIGNER_SRCS
    ${DESIGNER_KERNELS_DIR}/kernels.c
)

# Build shared library linking both local + designer kernels
add_library(aria_native_runtime SHARED ${NATIVE_SRCS} ${DESIGNER_SRCS})
target_link_libraries(aria_native_runtime m)

# Build static library too for Rust FFI
add_library(aria_native_runtime_static STATIC ${NATIVE_SRCS} ${DESIGNER_SRCS})
target_link_libraries(aria_native_runtime_static m)

# Find OpenBLAS — check system first (standard symbol names)
find_library(OPENBLAS_LIB NAMES openblas
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    NO_DEFAULT_PATH
)

# Preferred fallback: scipy-bundled OpenBLAS 0.3.27 DYNAMIC_ARCH (scipy_cblas_sgemm prefix)
if(NOT OPENBLAS_LIB)
    file(GLOB OPENBLAS_LIB "/home/tim/venvs/llm/lib/python3.12/site-packages/scipy.libs/libscipy_openblas*.so")
    if(OPENBLAS_LIB)
        list(GET OPENBLAS_LIB 0 OPENBLAS_LIB)
        set(OPENBLAS_SCIPY_PREFIX ON)
    endif()
endif()

# Last fallback: ollama-bundled OpenBLAS 0.3.15 (standard cblas_sgemm symbols)
if(NOT OPENBLAS_LIB)
    file(GLOB OPENBLAS_LIB "/usr/local/lib/ollama/mlx_cuda_v13/libopenblas*.so*")
    if(OPENBLAS_LIB)
        # Prefer the versioned symlink (libopenblas.so.0)
        list(SORT OPENBLAS_LIB)
        list(GET OPENBLAS_LIB 0 OPENBLAS_LIB)
    endif()
endif()

# OpenMP for parallel tensor operations
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_C_FLAGS}")
    target_link_libraries(aria_native_runtime OpenMP::OpenMP_C)
    target_link_libraries(aria_native_runtime_static OpenMP::OpenMP_C)
    add_definitions(-DARIA_HAS_OPENMP=1)
else()
    message(STATUS "OpenMP not found, kernels will be single-threaded")
endif()

if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    add_definitions(-DARIA_HAS_BLAS=1)
    if(OPENBLAS_SCIPY_PREFIX)
        message(STATUS "Using scipy-prefixed BLAS symbols")
        add_definitions(-DARIA_BLAS_SCIPY_PREFIX=1)
    endif()
    target_link_libraries(aria_native_runtime ${OPENBLAS_LIB})
    target_link_libraries(aria_native_runtime_static ${OPENBLAS_LIB})
    # Extract directory for RPATH so runtime linker can find the .so and its deps
    get_filename_component(OPENBLAS_DIR ${OPENBLAS_LIB} DIRECTORY)
    set_target_properties(aria_native_runtime PROPERTIES
        BUILD_RPATH "${OPENBLAS_DIR}"
        INSTALL_RPATH "${OPENBLAS_DIR}")
    set_target_properties(aria_native_runtime_static PROPERTIES
        BUILD_RPATH "${OPENBLAS_DIR}"
        INSTALL_RPATH "${OPENBLAS_DIR}")
else()
    message(STATUS "OpenBLAS not found, using naive matmul/linear")
endif()

# Tests
enable_testing()
file(GLOB TEST_SRCS tests/*.c)
foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} aria_native_runtime m)
    if(OPENBLAS_LIB)
        target_link_libraries(${test_name} ${OPENBLAS_LIB})
        set_target_properties(${test_name} PROPERTIES
            BUILD_RPATH "${OPENBLAS_DIR}"
            INSTALL_RPATH "${OPENBLAS_DIR}")
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Install
install(TARGETS aria_native_runtime LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/aria)
