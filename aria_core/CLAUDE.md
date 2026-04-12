# aria_core/ — High-Performance Kernel Library

C++/CUDA extension library for Aria. Provides native ops consumed by both `research/` and `aria_designer/`.

## Build
```bash
cd /home/tim/Projects/LLM/aria_core
python setup.py build_ext --inplace
# Restart dashboard/API to pick up new .so
```

## Structure
- `src/cpu/` — CPU kernels (kernels.cpp, graph_validator.cpp, shape_inference.cpp, clifford.cpp, hyperbolic.cpp)
- `src/gpu/` — CUDA kernels (tropical.cu, clifford.cu)
- `bindings/` — pybind11 bindings (bindings.cpp, bind_kernels.cpp, bind_ops.cpp, bind_graph.cpp)
- `tests/` — kernel correctness tests
- `_bootstrap.py` — native extension loader

## Rules
- All new compute ops go here as C++/CUDA first, Python fallback second
- Test every kernel with finite-value checks (`torch.isfinite`)
- Match shapes exactly — shape inference bugs cascade through the compiler
- Rebuild after any change: `python setup.py build_ext --inplace`
