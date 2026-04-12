/**
 * routing_ops.c — Native CPU kernels for routing operators.
 *
 * transpose_sd:             Even/odd channel interleave (pure memory shuffle)
 * gated_lane_blend:         Soft-routed multi-lane transform
 * depth_gated_transform:    Variable-depth per-token transform
 * calibrated_branch_merge:  Two-branch merge with RMS calibration
 */

#include "kernels_common.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ── transpose_sd: interleave even/odd channels ────────────────────── */

void aria_transpose_sd_f32(const float *x, float *y,
                           int64_t batch, int64_t seq, int64_t dim) {
    if (!x || !y || batch <= 0 || seq <= 0 || dim <= 0 || dim % 2 != 0) return;
    const int64_t half = dim / 2;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const float *src = x + (b * seq + s) * dim;
            float *dst = y + (b * seq + s) * dim;
            for (int64_t h = 0; h < half; ++h) {
                dst[h * 2]     = src[h];
                dst[h * 2 + 1] = src[half + h];
            }
        }
    }
}

/* ── Shared softmax over contiguous K-length logit vectors ─────────── */

static void _softmax_row(float *logits, int64_t k) {
    float mx = logits[0];
    for (int64_t i = 1; i < k; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        logits[i] = expf(logits[i] - mx);
        sum += logits[i];
    }
    const float inv = 1.0f / (sum + 1e-8f);
    for (int64_t i = 0; i < k; ++i) {
        logits[i] *= inv;
    }
}

/* ── gated_lane_blend: soft-routed multi-lane transform ────────────── */
/*
 * scorer:  [n_lanes, D]           lane scoring weights
 * projs:   [n_lanes, D_out, D]    per-lane projection weights
 *
 * For each token:
 *   logits = x @ scorer^T                    → [n_lanes]
 *   weights = softmax(logits)                → [n_lanes]
 *   per_lane = x @ projs[l]^T for each l    → [n_lanes, D_out]
 *   output = sum(weights[l] * per_lane[l])   → [D_out]
 */

void aria_gated_lane_blend_f32(const float *x,
                               const float *scorer,
                               const float *projs,
                               float *y,
                               int64_t batch, int64_t seq,
                               int64_t dim, int64_t n_lanes) {
    if (!x || !scorer || !projs || !y ||
        batch <= 0 || seq <= 0 || dim <= 0 || n_lanes <= 0) return;

    arena_reset();
    float *logits = (float *)arena_alloc((size_t)n_lanes * sizeof(float));
    float *lane_out = (float *)arena_alloc((size_t)dim * sizeof(float));
    if (!logits || !lane_out) return;

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const float *xr = x + (b * seq + s) * dim;
            float *yr = y + (b * seq + s) * dim;

            /* Score: logits[l] = dot(x, scorer[l]) */
            for (int64_t l = 0; l < n_lanes; ++l) {
                const float *sw = scorer + l * dim;
                float v = 0.0f;
                for (int64_t d = 0; d < dim; ++d) v += xr[d] * sw[d];
                logits[l] = v;
            }
            _softmax_row(logits, n_lanes);

            /* Weighted sum of per-lane projections */
            memset(yr, 0, (size_t)dim * sizeof(float));
            for (int64_t l = 0; l < n_lanes; ++l) {
                const float w = logits[l];
                const float *pw = projs + l * dim * dim;
                for (int64_t o = 0; o < dim; ++o) {
                    float v = 0.0f;
                    for (int64_t d = 0; d < dim; ++d) v += xr[d] * pw[o * dim + d];
                    yr[o] += w * v;
                }
            }
        }
    }
    arena_free(lane_out);
    arena_free(logits);
}

/* ── depth_gated_transform: variable-depth per-token transform ─────── */
/*
 * Identical structure to gated_lane_blend but with depth semantics.
 * scorer:  [max_depth, D]           depth scoring weights
 * projs:   [max_depth, D_out, D]    per-depth projection weights
 */

void aria_depth_gated_transform_f32(const float *x,
                                    const float *scorer,
                                    const float *projs,
                                    float *y,
                                    int64_t batch, int64_t seq,
                                    int64_t dim, int64_t max_depth) {
    /* Same math as lane blend — parameterised by depth instead of lanes. */
    aria_gated_lane_blend_f32(x, scorer, projs, y,
                              batch, seq, dim, max_depth);
}

/* ── calibrated_branch_merge: two-branch weighted merge ────────────── */
/*
 * a, b:         [B, S, D]    two branch inputs
 * score_proj:   [2, 1, D]    per-branch scoring projection
 * branch_bias:  [2]          per-branch bias
 * branch_gain:  [2]          per-branch gain (pre-sigmoid)
 *
 * For each token:
 *   rms_a, rms_b = RMS(a), RMS(b)
 *   norm_a, norm_b = a / rms_a, b / rms_b
 *   score_0 = dot(norm_a, score_proj[0]) + bias[0]
 *   score_1 = dot(norm_b, score_proj[1]) + bias[1]
 *   weights = softmax([score_0, score_1] / temperature)
 *   w1 = clamp(weights[1], min_share, max_share)
 *   w0 = 1 - w1
 *   gain_i = 0.5 + sigmoid(branch_gain[i])
 *   output = (norm_a * w0 * gain_0 + norm_b * w1 * gain_1) * rms_a
 */

static float _rms_f32(const float *v, int64_t d) {
    float sum = 0.0f;
    for (int64_t i = 0; i < d; ++i) sum += v[i] * v[i];
    return sqrtf(sum / (float)d + 1e-8f);
}

static float _sigmoid_f32(float x) {
    if (x >= 0.0f) { float z = expf(-x); return 1.0f / (1.0f + z); }
    float z = expf(x); return z / (1.0f + z);
}

void aria_calibrated_branch_merge_f32(const float *a, const float *b,
                                      const float *score_proj,
                                      const float *branch_bias,
                                      const float *branch_gain,
                                      float *y,
                                      int64_t batch, int64_t seq, int64_t dim,
                                      float temperature,
                                      float min_secondary,
                                      float max_secondary) {
    if (!a || !b || !y || batch <= 0 || seq <= 0 || dim <= 0) return;
    if (temperature < 1e-4f) temperature = 1.0f;

    for (int64_t bs = 0; bs < batch * seq; ++bs) {
        const float *ar = a + bs * dim;
        const float *br = b + bs * dim;
        float *yr = y + bs * dim;

        float rms_a = _rms_f32(ar, dim);
        float rms_b = _rms_f32(br, dim);
        float inv_a = 1.0f / rms_a;
        float inv_b = 1.0f / rms_b;

        /* Branch scores */
        float s0 = branch_bias ? branch_bias[0] : 0.0f;
        float s1 = branch_bias ? branch_bias[1] : 0.0f;
        if (score_proj) {
            for (int64_t d = 0; d < dim; ++d) {
                s0 += (ar[d] * inv_a) * score_proj[d];
                s1 += (br[d] * inv_b) * score_proj[dim + d];
            }
        }

        /* Softmax over 2 scores */
        float logits[2] = {s0 / temperature, s1 / temperature};
        _softmax_row(logits, 2);

        /* Clamp secondary share */
        float w1 = logits[1];
        if (w1 < min_secondary) w1 = min_secondary;
        if (w1 > max_secondary) w1 = max_secondary;
        float w0 = 1.0f - w1;

        /* Gains: 0.5 + sigmoid(raw) */
        float g0 = branch_gain ? 0.5f + _sigmoid_f32(branch_gain[0]) : 1.0f;
        float g1 = branch_gain ? 0.5f + _sigmoid_f32(branch_gain[1]) : 1.0f;

        /* Weighted merge with RMS anchor */
        for (int64_t d = 0; d < dim; ++d) {
            yr[d] = ((ar[d] * inv_a) * w0 * g0 +
                     (br[d] * inv_b) * w1 * g1) * rms_a;
        }
    }
}
