# aria_core — GLOBAL_DEV_PROMPT Compliance Plan

Audit date: 2026-03-14
Audited against: `/home/tim/Projects/LLM/GLOBAL_DEV_PROMPT.md`

---

## Completed Work

### Phase 1: Dead Code Removal
- [x] **F1.1** Deleted 11 orphaned root scripts (bench_cumsum.py, get_novelty*.py, test_cka*.py)
- [x] **F1.2** Investigated 34 binding_stubs functions — **NOT dead code** (used in bindings.cpp, Cython bridge, native registry)
- [x] **F1.3** Kept inner `__init__.py` minimal (circular import prevents delegation to parent)

### Phase 2: Memory Safety
- [x] **F2.1** Added null checks to malloc sites in: fp16.cpp (4 sites), math_space.cpp (3 sites), hyperbolic.cpp (3 sites), fingerprint_metrics.cpp (1 site)
- [x] **F2.2** Fixed memory leak: math_space.cpp tropical_attention/tropical_gate now free on alloc failure
- [x] **F2.3** Replaced unbounded `alloca()` in kernels.cpp with bounded malloc+free
- [x] **F2.4** Added scalar fallback functions to simd_math.h for non-AVX2/AVX512 platforms

### Phase 4: Polish (partial)
- [x] **F4.1** Added named constants to kernels_common.h: `ARIA_STACK_ALLOC_THRESHOLD`, `ARIA_OMP_COMPUTE_THRESHOLD`, `ARIA_EPSILON_DEFAULT`
- [x] **F4.2** Added `logging.warning()` to tuning_cache.py silent except blocks

### Build & Test Verification
- [x] C extension builds cleanly (`python setup.py build_ext --inplace`)
- [x] 53/54 tests pass (1 pre-existing failure: `embedding_lookup_f32` is intentional empty stub)

---

## Remaining Work

### Phase 3: God File Splits (structural, higher risk)

#### F3.1 Split `fused_ops.py` (1740 lines) into package
```
aria_core/gpu/fused_ops/
  __init__.py       — public API re-exports
  _config.py        — feature flags, triton detection
  _rope.py          — fused_rope + triton/pytorch impls
  _qk_norm.py       — fused_qk_norm + backward
  _swiglu.py        — fused_swiglu + backward
  _rms_norm.py      — fused_rms_norm + backward
  _cross_entropy.py — chunked CE + autograd.Function
  _benchmark.py     — benchmark_kernels()
```

#### F3.2 Split `bindings/bindings.cpp` (1712 lines) by tier
```
bindings/
  bindings.cpp        — module init + includes
  bind_unary.cpp      — relu, gelu, silu, exp, log, etc.
  bind_binary.cpp     — add, mul, sub, max, min
  bind_linalg.cpp     — matmul, linear, cka, tropical
  bind_norm.cpp       — rmsnorm, layernorm, softmax
  bind_routing.cpp    — routing + gating ops
  bind_math_space.cpp — hyperbolic, clifford, tropical, p-adic
  bind_fused.cpp      — fused ops + compound kernels
  bind_graph.cpp      — graph executor, validator, shape inference
```

### Phase 4: Polish (remaining)
- [ ] **F4.3** Replace magic number `4096` with `ARIA_STACK_ALLOC_THRESHOLD` in hyperbolic.cpp, math_space.cpp
- [ ] **F4.4** Add type hints to test reference functions in test_adaptive_routing_kernels.py
- [ ] **F4.5** Add null checks to remaining binding_stubs.cpp malloc sites (~10 unchecked)

### Pre-existing Issues (not introduced by this work)
- `embedding_lookup_f32` is an intentional empty stub — test always fails
- `test_cka_parity.py` and `test_proactive_gating.py` depend on `research` module (cross-module)
