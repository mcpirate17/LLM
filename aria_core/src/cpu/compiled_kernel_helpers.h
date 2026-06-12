#ifndef ARIA_COMPILED_KERNEL_HELPERS_H
#define ARIA_COMPILED_KERNEL_HELPERS_H

#include <cmath>
#include <cstdint>

/* Shared helpers for the hand-compiled sequence kernels (gated_delta,
 * selective_scan, state_space, softmax_attention — forward and backward).
 * Forward/backward pairs previously carried per-file renamed copies of these
 * (e.g. sigmoid_scalar vs gated_delta_backward_sigmoid_scalar); identical
 * math duplicated per file is exactly how forward/backward drift starts. */

// Positive forget-gate-bias init on the gated_delta retention gate: the decay
// is sigmoid(alpha_logit + kGatedDeltaDecayBias), so at init (logits ≈ 0) the
// gate is ≈0.92 and the recurrent state is *kept*, not wiped. Without it the
// gate sits at 0.5 and — combined with decay = alpha (a true retention gate,
// not the old alpha - beta which centred decay at 0) — the state dies before
// training can bias it (mamba2 baseline scored 0.0 everywhere; diagnosed
// 2026-06-07). Must stay in lockstep with the torch reference
// `_op_gated_delta` (_GATED_DELTA_DECAY_BIAS in
// research/synthesis/compiler_ops_sequence.py).
constexpr float kGatedDeltaDecayBias = 2.5f;

inline float aria_ck_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/* out[o] = sum_i x_row[i] * weight[o * dim + i] (row-major [out, in] weight) */
inline void aria_ck_linear_project(const float *x_row,
                                   const float *weight,
                                   float *out,
                                   int64_t dim) {
    for (int64_t o = 0; o < dim; o++) {
        float sum = 0.0f;
        const float *w_row = weight + o * dim;
        for (int64_t i = 0; i < dim; i++) {
            sum += x_row[i] * w_row[i];
        }
        out[o] = sum;
    }
}

/* lo < value < hi — true where a forward clamp(value, lo, hi) was inactive,
 * i.e. where gradient should pass. */
inline bool aria_ck_unclamped(float value, float lo, float hi) {
    return value > lo && value < hi;
}

#endif /* ARIA_COMPILED_KERNEL_HELPERS_H */
