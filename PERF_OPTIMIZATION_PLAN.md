# Performance Optimization Plan: Removing "Polyfill" Fallbacks

This document outlines the strategy for eliminating performance debt in the Aria research pipeline, specifically focusing on the slow "investigation training" phase and behavioral fingerprinting.

## Current Bottlenecks & "Polyfill" Debt

### 1. Sequential Jacobian Sensitivity (O(N) Backward Passes)
*   **Problem:** `_collect_position_sensitivities_fallback` in `research/eval/fingerprint.py` uses a Python loop to call `backward(retain_graph=True)` for each probed position. This results in **128 sequential backprops** per full fingerprint (32 probes × 4 positions).
*   **Debt:** This is a "solvable" problem that currently relies on a naive autograd loop.
*   **Fix:** Replace the loop with `torch.func.vmap` and `torch.func.grad` (or `jacrev`) to compute all position sensitivities in a single vectorized backward pass.

### 2. Python-Only Interaction/Sensitivity Metrics
*   **Problem:** `interaction_metrics_f32` and `sensitivity_metrics_f32` are currently "polyfilled" in `research/eval/fingerprint.py` because the native `aria_core` C++ implementations are missing.
*   **Debt:** Heavy mathematical reductions (locality, sparsity, effective rank) are being performed in Torch/Python instead of optimized C++ SIMD.
*   **Fix:** 
    *   Implement `aria_interaction_metrics_f32` and `aria_sensitivity_metrics_f32` in `aria_core/src/cpu/kernels.cpp` using AVX2/AVX-512.
    *   Bind these in `aria_core/bindings/bindings.cpp` to remove the fallback logic entirely.

### 3. Tropical GPU Fallback (O(S²D) Bottleneck)
*   **Problem:** `tropical_matmul` and `tropical_gate` (used in the current fingerprint `6728ec4ec8166a0a`) fall back to a slow Python/Torch implementation on GPUs because `aria_core` kernels are currently CPU-only.
*   **Debt:** The Python fallback uses O(B*S*S*D) memory and compute via broadcasting, which is extremely inefficient on CUDA.
*   **Fix:** Implement a dedicated CUDA kernel for `tropical_matmul` and `tropical_add` to allow native GPU execution of tropical semiring operations.

### 4. Blocking Benchmark Evaluations
*   **Problem:** `_evaluate_investigation_benchmarks` blocks the main investigation thread for an additional **400 steps** per survivor across two model re-compilations.
*   **Debt:** This is a serial process that could be performed in a separate background validation worker.
*   **Fix:** Offload benchmark evaluations to the `ValidationWorker` or a dedicated benchmark queue.

---

## Implementation Roadmap

### Phase 1: Fingerprint Vectorization (Immediate Speedup)
*   Refactor `research/eval/fingerprint.py` to use `torch.func` for sensitivity analysis.
*   **Goal:** Reduce sensitivity probe time by ~75% (1 backward pass instead of 4 per probe).

### Phase 2: Native Kernel Expansion
*   Add missing metric kernels to `aria_core`.
*   Remove the `if not hasattr(aria_core, ...)` polyfill guards in the Python code.
*   **Goal:** 10-20x faster reduction logic for fingerprint statistics.

### Phase 3: CUDA Acceleration
*   Port key tropical and clifford kernels to CUDA.
*   Remove the `device.type == "cpu"` check in `bindings.cpp` for these operations.
*   **Goal:** Enable native GPU training for exotic math-space architectures.

### Phase 4: Investigation Gating
*   Enforce the `compute_gated_fingerprint` logic in `execution_training.py`.
*   **Goal:** Skip the full 32-probe analysis for any architecture that fails the 1-probe "lightning" novelty check.
