# Native Runtime Reuse Contract

## Ownership Model

### aria-designer/runtime/ (Primary Owner)
- **C kernels** (`runtime/src/kernels.c`, `kernels.h`): All shared tensor ops live here.
- **Bridge** (`runtime/bridge.py`): Workflow-to-graph conversion and evaluation.
- **Dispatch** (`runtime/dispatch.py`): C kernel vs Python fallback selection.
- **Compiler** (`runtime/compiler.py`): Workflow → torch.nn.Module.

### research/runtime/native/ (Extension, Consumer)
- **ABI headers** (`include/`): Define C-level contracts for init/compile/execute/teardown.
- **Rust scheduler** (`rust/aria-scheduler/`): Graph execution planning, memory arena, topological executor.
- **Cython bridge** (`cython/`): Zero-copy Python boundary for kernel dispatch and scheduler invocation.
- **Runner-specific kernels** (`src/`): Only ops not present in Designer runtime. Must be explicitly approved.

## Rules

1. **No kernel duplication**: If aria-designer already has a C kernel for an op, research MUST use it via linking, not copy it.
2. **New kernels land in Designer first**: If a new kernel is needed, implement it in `aria-designer/runtime/src/` and expose via `kernels.h`. Research links against it.
3. **Temporary exceptions**: If a runner-specific kernel is urgently needed and cannot land in Designer immediately, it may live in `research/runtime/native/src/` with:
   - A tracking issue or TODO in this file
   - A retirement deadline (max 2 sprints)
   - A migration plan to move it into Designer
4. **ABI stability**: Headers in `include/` are the contract between Python/Rust and C. Changes require version bump in `native_ir.v1` schema.
5. **Build integration**: `CMakeLists.txt` links against aria-designer kernel sources directly. No vendored copies.

## Current State

| Component | Location | Owner | Status |
|-----------|----------|-------|--------|
| Elementwise ops (relu, gelu, silu, etc.) | aria-designer/runtime/src/kernels.c | Designer | Active, 8+ ops |
| Binary ops (add, mul, sub) | aria-designer/runtime/src/kernels.c | Designer | Active |
| Matmul (tiled) | aria-designer/runtime/src/kernels.c | Designer | Active |
| Linear projection | aria-designer/runtime/src/kernels.c | Designer | Active |
| RMSNorm | aria-designer/runtime/src/kernels.c | Designer | Active |
| Reductions (sum, mean) | aria-designer/runtime/src/kernels.c | Designer | Active |
| Graph scheduler | research/runtime/native/rust/ | Research | New (scaffold) |
| Memory arena | research/runtime/native/rust/ | Research | New (scaffold) |
| Kernel registry + dispatch | research/runtime/native/src/ | Research | New (scaffold) |
| Softmax | research/runtime/native/src/ | Research | Temporary — migrate to Designer |
| Concat/Split | research/runtime/native/src/ | Research | Temporary — migrate to Designer |

## Temporary Exceptions

| Kernel | Location | Reason | Retirement Target |
|--------|----------|--------|-------------------|
| softmax_f32 | research/runtime/native/src/kernels_ext.c | Not yet in Designer | Sprint +1 |
| concat_f32 | research/runtime/native/src/kernels_ext.c | Not yet in Designer | Sprint +1 |
| split_f32 | research/runtime/native/src/kernels_ext.c | Not yet in Designer | Sprint +1 |
