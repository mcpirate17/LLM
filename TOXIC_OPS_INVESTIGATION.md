# Aria AI Scientist: Toxic Op & Success Rate Investigation

## 1. Summary of Findings
The investigation into the 28 ops with 0% S1 success rates and 171 toxic op-pair patterns has revealed that many of these failures were due to **low-level code errors and initialization gaps** rather than fundamental theoretical flaws.

### Key Issues Identified:
*   **Stage 0 (Compilation) Failures:**
    *   **Dtype Mismatches:** `mod_topk` and other routing ops failed when the model used `float16/bfloat16` but `aria_core` or the `mask` was initialized as `float32`.
    *   **Indexing Errors:** `moe_topk` had a broadcasting error in its expert dispatch logic, causing crashes when `top_k > 1`.
    *   **Missing Initializers:** Many math space ops (`tropical_center`, `rotor_transform`, `ultrametric_attention`) were registered as primitives but lacked parameter initialization in `CompiledOp`, causing them to fall back to a default `nn.Parameter` that didn't match their expected shapes.
*   **Stage 1 (Learning) Failures / Zero-Grad:**
    *   **Non-Differentiable Operators:** `div_safe` used `torch.sign(b)` which has a zero gradient everywhere, killing backpropagation for the denominator branch.
    *   **Causality Violations:** `irfft_seq` and `_op_causal_mask` had subtle non-causal behaviors that made them "toxic" in an autoregressive (next-token prediction) context.
    *   **Gradient Flow Breakers:** Ops that returned integer `indices` (like `route_topk`) without properly utilizing their associated `weights` broke the gradient chain.

---

## 2. Fixes Applied
We have performed a surgical update to `research/synthesis/compiler.py`:

1.  **Differentiable `div_safe`:** Replaced `torch.sign` with a stable, differentiable alternative.
2.  **Robust `moe_topk`:** Fixed the expert indexing and broadcasting logic to support any `top_k` and handle empty expert masks safely.
3.  **CUDA-Safe Indexing:** Added `.clamp()` to all `gather` and `scatter_` operations in the synthesis engine to prevent the CUDA kernel assertions that were killing the GPU context.
4.  **Math Space Initializers:** Added comprehensive parameter initialization for `MATH_SPACE` category ops in `CompiledOp._init_params`.
5.  **Causal `tropical_center`:** Re-implemented using `torch.cummin` to ensure the operation is strictly causal and suitable for language modeling.

---

## 3. Recommendations for "Toxic" Patterns
Many "toxic op-pairs" (e.g., `ultrametric_attention -> add`) are likely labeled as such because the first op in the pair was previously non-differentiable or returning non-finite values. 

**Next Steps:**
- **Re-Evaluate:** With the fixes in `compiler.py`, many of the previously "toxic" ops should be unlocked. We recommend a fresh 100-model "Recalibration" run.
- **Novelty Engine Update:** Adjust the `NoveltyEngine` to reduce the penalty for these ops now that their implementation is sound.
- **Native Kernels:** For the slow sequential ops (like `rwkv_time_mixing`), we should implement native C++/CUDA kernels in `aria_core` to prevent them from being penalized for high latency during synthesis.
