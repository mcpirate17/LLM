# High-Performance & DRY Architecture TODO List

## Overview
Currently, the workspace suffers from fragmented implementations of mathematical primitives and runtime logic across multiple projects:
- `aria-designer/runtime/src/`: Contains C-based kernels and graph validation.
- `aria-designer/components/math_space/`: Contains scattered C stubs (`kernel.c`).
- `research/mathspaces/`: Contains pure PyTorch implementations (e.g., `clifford.py`, `hyperbolic.py`).
- `LA3/kernels/`: Contains Triton/CUDA kernels for attention.
- `HYDRA/`: Contains training loops that likely rely on the Python implementations.

To achieve a **DRY (Don't Repeat Yourself)** architecture with **maximum performance**, we need to consolidate these into a single, unified high-performance core library (e.g., `libaria-core`) written in C++/CUDA/Rust, which exposes bindings to all Python and frontend consumers.

---

## Phase 1: Unified Core Library Setup (`libaria-core`)
- [x] [C:gemini-cli 2026-02-26] **Create a Centralized Native Project**: Established `/home/tim/Projects/LLM/aria-core/`.
- [x] [C:gemini-cli 2026-02-26] **Select the Primary Systems Language**: Chose **C++ / PyTorch Extensions** (`torch.utils.cpp_extension`) for direct ATen integration.
- [x] [C:gemini-cli 2026-02-26] **Migrate Existing C Code**: Moved `aria-designer/runtime/src/*.c` to `aria-core/src/cpu/*.cpp`.
- [x] [C:gemini-cli 2026-02-26] **Migrate Triton/CUDA Kernels**: Moved `LA3/kernels/` and Triton ops into `aria_core/gpu/`.
- [x] [C:gemini-cli 2026-02-26] **Setup Build System**: Configured `setup.py` with `CppExtension` and tested build.
- [x] [C:claude-opus 2026-02-26] **Expand pybind11 Bindings**: 10 → 92 bindings covering all kernel categories.

## Phase 2: Mathematical Primitives Consolidation (The DRY Principle)
- [x] [C:gemini-cli 2026-02-26] **Audit Current Implementations**: Mapped PyTorch functions from `research/mathspaces/` to `aria-core`.
- [x] [C:gemini-cli 2026-02-26] **Implement Native Math Kernels**: Clifford Cl(3,0) and Hyperbolic distance implemented in `aria-core/src/cpu/`.
- [x] [C:gemini-cli 2026-02-26] **Delete Redundant Stubs**: Removed all the empty `kernel.c` stubs in `aria-designer/components/math_space/*/`.
- [x] [C:gemini-cli 2026-02-26] **Create Thin Python Wrappers**: Refactored `research/mathspaces/*.py` (`clifford.py`, `hyperbolic.py`, `padic.py`, `tropical.py`, `spiking.py`) to call `aria_core`.

## Phase 3: High-Performance Operations & Memory Management
- [x] [C:claude-opus 2026-02-26] **Zero-Copy Tensor Passing**: Achieved via pybind11 `torch::Tensor` — all bindings use ATen tensors directly, no copies between Python↔C++. The `dispatch.py` np↔torch boundary is at the API edge only.
- [x] [C:gemini-cli 2026-02-26] **SIMD Vectorization**: Expanded `simd_math.h` with AVX-512 support. Vectorized relu, binary ops, rmsnorm, layernorm, and non-Euclidean ops (Clifford GP, Hyperbolic).
- [x] [C:gemini-cli 2026-02-26] **Kernel Fusion**: Migrated HYDRA fused Triton kernels to `aria_core/gpu`. Fused Clifford rotor and Hyperbolic distance in C++.
- [x] [C:gemini-cli 2026-02-26] **Cython/C++ Graph Execution**: Implemented `GraphExecutor` class in C++ to run topological sequences of kernels with zero Python loop overhead.

## Phase 4: Integration & CI/CD
- [x] [C:claude-opus 2026-02-26] **Update `aria-designer` Backend**: Refactored `runtime/dispatch.py` to use `aria_core` pybind11 bindings (graph validation, shape inference, all kernels) with CFFI fallback. Added `validate_graph()` and `propagate_shapes()` pybind11 wrappers.
- [x] [C:gemini-cli 2026-02-26] **Update `HYDRA` and `LA3`**: Refactored `RMSNorm`, `SwiGLUMLP`, and `SimpleRMSNormTorch` to use `aria_core` native primitives.
- [x] [C:gemini-cli 2026-02-26] **Unified Testing**: Created `aria-core/tests/verify_install.py` for numerical equivalence testing.
- [x] [C:claude-opus 2026-02-26] **Automated Build Pipeline**: Created root `Makefile` with targets: `all`, `aria-core`, `test`, `test-aria-core`, `test-designer`, `clean`, `help`.

## Summary of Target Architecture
```text
/home/tim/Projects/LLM/
├── aria-core/                 <-- NEW: Unified C++/Rust/CUDA backend
│   ├── src/cpu/               <-- SIMD math, graph validation
│   ├── src/gpu/               <-- Triton/CUDA kernels (from LA3 & Mathspaces)
│   ├── bindings/              <-- pybind11 / PyO3 Python wrappers
│   └── setup.py               <-- Build system
├── aria-designer/             <-- Frontend & API (Consumes aria-core Python bindings)
│   ├── api/
│   └── ui/
├── research/                  <-- Research scripts (Consumes aria-core Python bindings)
│   └── mathspaces/            <-- Thin Python wrappers around aria-core
├── HYDRA/                     <-- Training loops (Consumes aria-core Python bindings)
└── LA3/                       <-- Attention variants (Consumes aria-core Python bindings)
```