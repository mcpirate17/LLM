/**
 * bindings.cpp — pybind11 module entry point for aria_core.
 *
 * Delegates to bind_kernels.cpp, bind_ops.cpp, bind_graph.cpp.
 */
#include "bind_common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "aria_core: Unified high-performance kernel library for Aria";
    bind_kernels(m);
    bind_ops(m);
    bind_graph(m);
}
