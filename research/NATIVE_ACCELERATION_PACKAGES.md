# Phase 5: Native Acceleration — Work Packages

## Profiling Summary (2026-03-11)

**Source:** `cProfile` of `--mode=synthesize --n 20 --seed 42 --n_layers 6` (74.8s total)

Non-LLM compute breakdown (24s):
- Training forward/backward: 14.8s
- Mathspace ops (Python): 1.65s
- IR executor Python loop overhead: 0.21s
- CompiledLayer dict dispatch: 0.06s
- RWKV time mixing sequential scan: 0.27s
- `gromov_delta` (eval stage, not in this run): 18.7ms/call O(n^4)

---

## Package A: Mathspace C Kernels + gromov_delta (claude-opus)

**Destination:** `aria_core/src/cpu/`

### A.1: Fused Hyperbolic Ops — `hyperbolic.cpp` (extend existing)

**Problem:** `_clamp_norm` (0.19s, 12.9K calls), `exp_map` (0.14s, 6.5K calls), `log_map` (0.17s, 6.5K calls) total **0.50s**. Each function calls `_clamp_norm` redundantly. Python overhead per call dominates at this call frequency.

**Implementation:**
- `aria_hyp_exp_map_f32(x, y, batch, seq, dim, c)` — fused clamp_norm + tanh + normalize in one pass
- `aria_hyp_log_map_f32(x, y, batch, seq, dim, c)` — fused clamp_norm + atanh + normalize in one pass
- `aria_hyp_linear_fused_f32(x, W, y, batch, seq, dim, c)` — fused log_map → matmul → exp_map (eliminates 2 intermediate allocations)
- All with AVX2 SIMD for norm computation and element-wise ops
- Wire through pybind11 bindings and add Python fallback guards in `hyperbolic.py`

**Current state:** `math_space.cpp` already has `_aria_hyp_clamp_norm` (static inline, SIMD). `hyperbolic.cpp` has `mobius_add` and `distance`. Need to add exp_map, log_map, and fused linear.

### A.2: RWKV Parallel Scan C Kernel

**Problem:** `_op_rwkv_time_mixing` (0.27s, 126 calls). The inner loop at `compiler.py:1009-1017` iterates `for t in range(S)` doing exp, mul, add per step — sequential by nature but the per-step math is SIMD-able.

**Implementation:**
- `aria_rwkv_wkv_scan_f32(k, v, r, w_decay, u_bonus, out, batch, seq, dim)` — single C function replacing the Python loop
- AVX2 for the per-step `exp(kt) * vt`, `wkv * exp_w` vectorized across D dimension
- The existing `aria_core.rwkv_time_mixing_f32` handles the case when all params are present but doesn't handle the WKV scan part — it's the projection-only fast path. This adds the actual scan.

### A.3: gromov_delta C + AVX2

**Problem:** O(n^4) pure Python, 18.7ms at n=30 (27,405 4-tuple iterations). Called during investigation/validation eval stages.

**Implementation:**
- `aria_gromov_delta_f32(distance_matrix, n, result)` — C function with 4-nested loop
- AVX2 to vectorize the innermost comparison `max(d(a,c)+d(b,d), d(a,d)+d(b,c)) - d(a,b) - d(c,d)`
- Expected speedup: 10-20x (eliminate Python interpreter overhead + SIMD)

### A.4: BehaviorArchive._update_cache Fix (Python-only)

**Problem:** `_update_cache` (0.71ms/call) rebuilds entire NxN distance matrix on every archive add. With 500 entries, that's 250K distance computations when only 500 are new.

**Fix:** Append single row/column to existing matrix instead of full rebuild. No C needed.

---

## Package B: Tropical Kernel Wiring (gemini)

**Destination:** `aria_core/bindings.cpp` + `research/mathspaces/tropical.py`

### Critical Discovery: C kernels already exist but aren't exposed

The following C functions exist in `aria_core/src/cpu/` but have **no pybind11 bindings** — they're compiled but unreachable from Python:

| C Function | File | Status |
|---|---|---|
| `aria_tropical_add_f32` | `binary.cpp:80` | Exists, NOT bound |
| `aria_tropical_matmul_f32` | `linalg.cpp:88` | Exists, NOT bound |
| `aria_tropical_center_f32` | `math_space.cpp:14` | Exists, NOT bound |
| `aria_tropical_attention_f32` | `math_space.cpp:32` | Exists, SIMD, NOT bound |
| `aria_tropical_gate_f32` | `math_space.cpp:106` | Exists, SIMD, NOT bound |

The Python code in `tropical.py` already has `_HAS_ARIA_CORE` guards checking for `tropical_add_f32`, `tropical_matmul_f32`, `tropical_center_f32` — but `_HAS_ARIA_CORE` is always False because the bindings don't expose them.

### B.1: Add pybind11 Bindings

Add to `aria_core/bindings.cpp`:

```cpp
// Tropical ops
m.def("tropical_add_f32", [](py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request(), buf_b = b.request();
    auto result = py::array_t<float>(buf_a.shape);
    auto buf_r = result.request();
    aria_tropical_add_f32((float*)buf_a.ptr, (float*)buf_b.ptr,
                          (float*)buf_r.ptr, buf_a.size);
    return result;
});

m.def("tropical_matmul_f32", [](py::array_t<float> a, py::array_t<float> b) {
    // a: (S, D), b: (D, S2) -> (S, S2)
    auto buf_a = a.request(), buf_b = b.request();
    int64_t S = buf_a.shape[0], D = buf_a.shape[1], S2 = buf_b.shape[1];
    auto result = py::array_t<float>({S, S2});
    auto buf_r = result.request();
    aria_tropical_matmul_f32((float*)buf_a.ptr, (float*)buf_b.ptr,
                             (float*)buf_r.ptr, S, D, S2);
    return result;
});

m.def("tropical_center_f32", [](py::array_t<float> x) {
    auto buf = x.request();
    int64_t B = buf.shape[0], S = buf.shape[1], D = buf.shape[2];
    auto result = py::array_t<float>(buf.shape);
    auto buf_r = result.request();
    aria_tropical_center_f32((float*)buf.ptr, (float*)buf_r.ptr, B, S, D);
    return result;
});

m.def("tropical_attention_f32", [](py::array_t<float> x, float temperature) {
    auto buf = x.request();
    int64_t B = buf.shape[0], S = buf.shape[1], D = buf.shape[2];
    auto result = py::array_t<float>(buf.shape);
    auto buf_r = result.request();
    aria_tropical_attention_f32((float*)buf.ptr, (float*)buf_r.ptr,
                                B, S, D, temperature);
    return result;
});

m.def("tropical_gate_f32", [](py::array_t<float> x, float temperature) {
    auto buf = x.request();
    int64_t B = buf.shape[0], S = buf.shape[1], D = buf.shape[2];
    auto result = py::array_t<float>(buf.shape);
    auto buf_r = result.request();
    aria_tropical_gate_f32((float*)buf.ptr, (float*)buf_r.ptr,
                           B, S, D, temperature);
    return result;
});
```

### B.2: Add Batched tropical_matmul C Kernel

The current C `aria_tropical_matmul_f32` handles single matrices. Python loops over batch:
```python
torch.stack([aria_core.tropical_matmul_f32(a[i], b[i]) for i in range(B)])
```

Add `aria_tropical_matmul_batched_f32(a, b, out, batch, S, D, S2)` with OpenMP parallelism over batch dimension.

### B.3: Wire Up Python Fast Paths

In `tropical.py`, the guards already exist for `tropical_add`, `tropical_matmul`, `tropical_center`. Add guards for:

- `execute_tropical_attention` → call `aria_core.tropical_attention_f32(x, temperature=0.1)` when contiguous+CPU
- `execute_tropical_gate` → call `aria_core.tropical_gate_f32(x, temperature=0.1)` when contiguous+CPU

In `tropical_routing.py`:
- `TropicalRouter.forward` → The core computation `torch.min(expanded_x + expanded_c, dim=-1).values` is a tropical matmul variant. Add a C fast path `aria_tropical_router_f32(x, centroids, out, B, S, D, n_experts)`.

### B.4: Tests

Add tests in `research/tests/test_tropical_native.py`:
- Parity: C kernel output matches Python output within 1e-5
- Edge cases: empty batch, S=1, large D
- Gradient flow: ensure autograd still works (C kernels are forward-only, backward uses Python)

### Build & Verify

```bash
cd aria_core && python setup.py build_ext --inplace
python -m pytest research/tests/test_tropical_native.py -x -q
# Verify existing tropical tests still pass:
python -m pytest research/tests/ -k tropical -x -q
```

---

## Package C: Rust IR Executor Acceleration (codex)

**Destination:** `research/runtime/native/rust/aria-scheduler/`

### Problem

`IRExecutor.forward()` at `synthesis/ir_executor.py:128` runs a Python for-loop over all nodes in the graph IR. Per forward pass:
- 14,329 calls to `forward()` (nested — layers call ops)
- 0.21s own time in the Python loop overhead (opcode dispatch, dict lookup, consumer count tracking)
- Each iteration does: numpy int conversion, dict lookup (`idx_to_op_idx.get`), consumer count decrement, None assignment for memory reclamation

The `CompiledLayer.forward()` at `compiler.py:2199` has the same pattern: Python for-loop over topological order, dict-based node output tracking, consumer count copy+decrement.

### Current Rust State

The `aria-scheduler` crate already has:
- `GraphIR` struct with `Node`, `Edge`, `NodeId` types (`graph.rs`)
- `topological_order()` (Kahn's algorithm) (`graph.rs`)
- `KernelDispatch` trait with `dispatch()` and `dispatch_into()` (`executor.rs`)
- `NativeKernelDispatch` that calls C kernels via FFI (`executor.rs`)
- `Arena` allocator for intermediate buffers (`arena.rs`)
- `ExecutionContext` with `HashMap<NodeId, NodeBuffer>` (`executor.rs`)
- `execute_graph()` function that runs the full graph (`executor.rs`)
- `python_bridge.rs` with PyO3 bindings

### What Needs to Change

The Rust executor currently works with **raw float buffers** (`&[f32]`). The Python `IRExecutor` works with **PyTorch tensors**. The gap is:

1. The Rust executor can handle ops that are pure C kernels (unary, binary, linear, etc.)
2. It cannot handle ops that need PyTorch autograd (for training backward pass)
3. The IR executor is used during training — backward pass is critical

### C.1: Hybrid Dispatch Strategy

Extend `python_bridge.rs` to accept PyTorch tensor pointers:

```rust
/// Try to execute the entire IR graph through Rust dispatch.
/// Returns None if any op requires Python fallback.
#[pyfunction]
fn try_execute_ir(
    op_codes: Vec<i32>,
    input_indices: Vec<[i32; 2]>,
    output_node_idx: i32,
    consumer_counts: Vec<i32>,
    input_data_ptr: usize,  // raw pointer to input tensor data
    input_numel: usize,
    native_supported_ops: Vec<bool>,  // pre-computed: which ops have C kernels
) -> Option<Vec<f32>> {
    // If ALL ops are native-supported, run entirely in Rust
    // Otherwise return None to fall back to Python
}
```

The Python side checks upfront which ops have native kernels:
```python
def forward(self, x):
    if self._all_native and x.is_contiguous() and not x.requires_grad:
        result = aria_scheduler.try_execute_ir(
            self.op_codes, self.input_indices, ...)
        if result is not None:
            return torch.from_numpy(np.array(result)).reshape(x.shape)
    # Fall back to Python loop
    return self._python_forward(x)
```

### C.2: Optimized Python Loop Fallback

Even when we can't use full Rust dispatch (training with autograd), optimize the Python loop:

1. **Replace `list(self.consumer_counts)` with pre-allocated numpy array** — avoid Python list allocation every forward pass
2. **Replace `self.idx_to_op_idx.get(i)` dict lookup with pre-built numpy array** — `self._op_idx_array[i]` is O(1) without Python dict overhead
3. **Replace `int(self.input_indices[i, 0])` with pre-converted Python list** — avoid numpy→Python int conversion 14K times per forward

### C.3: Integer Opcode Dispatch in Rust

Currently the Rust `KernelDispatch` trait uses string-based dispatch (`op_name: &str`). For the IR path, add integer opcode dispatch:

```rust
/// Fast dispatch by integer opcode (avoids string comparison).
fn dispatch_by_opcode(
    &self,
    opcode: i32,
    inputs: &[&[f32]],
    config: &serde_json::Value,
    output_buf: &mut [f32],
) -> Result<usize, AriaError> {
    match opcode {
        1 => self.dispatch_into("relu", inputs, config, output_buf),
        2 => self.dispatch_into("gelu", inputs, config, output_buf),
        3 => self.dispatch_into("add", inputs, config, output_buf),
        // ... map all opcodes
        _ => Err(AriaError::UnsupportedOp(format!("opcode {}", opcode))),
    }
}
```

### C.4: Consumer Count + Memory Reclamation in Rust

The current Python loop tracks liveness:
```python
counts[in1_idx] -= 1
if counts[in1_idx] <= 0 and in1_idx != output_idx:
    node_outputs[in1_idx] = None
```

Move this to Rust with arena allocation:
- Pre-compute consumer counts once at IR load time (already done)
- Decrement in Rust loop, mark arena slots as reclaimable
- Arena reset at end of forward pass

### C.5: Tests

Extend `research/tests/test_native_runner_adapter.py` or create `research/tests/test_rust_ir_executor.py`:
- Parity: Rust executor output matches Python executor for 20 random graphs
- Performance: benchmark Python vs Rust forward pass (no grad)
- Edge cases: single-node graph, disconnected branches, dead nodes

### Build & Verify

```bash
cd research/runtime/native/rust/aria-scheduler
maturin develop --release
cd /home/tim/Projects/LLM
python -m pytest research/tests/test_native_runner_adapter.py -x -q
python -m pytest research/tests/test_e2e_benchmark.py -x -q
```

---

## Execution Order

1. **Package B first** (gemini) — pure wiring, no new C code needed, immediate speedup
2. **Package A in parallel** (claude-opus) — new C kernels + bindings
3. **Package C after A+B** (codex) — Rust executor depends on C kernels being stable

## Verification After All Packages

```bash
# Rebuild everything
cd aria_core && python setup.py build_ext --inplace
cd research/runtime/native/rust/aria-scheduler && maturin develop --release

# Run all native tests
python -m pytest research/tests/test_native_*.py research/tests/test_e2e_*.py research/tests/test_tropical_native.py -x -q

# Performance regression: run synthesis and compare times
python -m cProfile -o /tmp/synth_after.prof -m research --mode=synthesize --n 20 --seed=42 --n_layers=6 --device=cpu --db=/tmp/profile_after.db
```
