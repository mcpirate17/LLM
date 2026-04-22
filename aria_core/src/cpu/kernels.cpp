/**
 * kernels.cpp — Master CPU kernel inclusion file.
 *
 * This file acts as the primary compilation unit for CPU kernels,
 * including optimized modular implementations while maintaining
 * a clean build structure.
 */

#include "kernels_common.h"

// ── Master Inclusion List (DRY Canonical Sources) ──────────────────

#include "unary.cpp"
#include "binary.cpp"
#include "linalg.cpp"
#include "norm.cpp"
#include "mixing.cpp"
#include "math_space.cpp"
#include "structural.cpp"
#include "backward.cpp"
#include "rwkv_time_mixing_backward.cpp"
#include "selective_scan_compiled.cpp"
#include "selective_scan_backward_compiled.cpp"
#include "state_space_compiled.cpp"
#include "state_space_backward_compiled.cpp"
#include "gated_delta_compiled.cpp"
#include "gated_delta_backward_compiled.cpp"
#include "softmax_attention_backward_compiled.cpp"
#include "depth_weighted_proj.cpp"
#include "io.cpp"
#include "adaptive_routing.cpp"
#include "routing.c"
#include "routing_ops.c"
#include "dispatch.c"
#include "binding_stubs.cpp"
#include "fp16.cpp"
#include "smoke_test.cpp"
#include "fingerprint_metrics.cpp"

extern "C" {
void aria_argsort_seq_f32(const float *x, int64_t *indices, int64_t batch, int64_t seq, int64_t dim) {
    /* Argsort along sequence dim by L2 norm of each token. Stable insertion sort. */
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        int64_t *ib = indices + b * seq;
        /* Initialize indices */
        for (int64_t s = 0; s < seq; s++) ib[s] = s;
        /* Compute norms inline and sort by them (stable insertion sort) */
        for (int64_t i = 1; i < seq; i++) {
            int64_t ki = ib[i];
            /* Compute norm for key element */
            float key_norm = 0.0f;
            const float *ki_row = xb + ki * dim;
            for (int64_t d = 0; d < dim; d++) key_norm += ki_row[d] * ki_row[d];
            int64_t j = i - 1;
            while (j >= 0) {
                float j_norm = 0.0f;
                const float *ji_row = xb + ib[j] * dim;
                for (int64_t d = 0; d < dim; d++) j_norm += ji_row[d] * ji_row[d];
                if (j_norm <= key_norm) break;
                ib[j + 1] = ib[j];
                j--;
            }
            ib[j + 1] = ki;
        }
    }
}
void aria_embedding_lookup_f32(const float *table, const int32_t *indices, const float *pos_embed, float *y, int64_t batch, int64_t dim, int64_t vocab_size) {
    for (int64_t i = 0; i < batch; i++) {
        int32_t idx = indices[i];
        if (idx < 0 || idx >= (int32_t)vocab_size) idx = 0;
        const float *row = table + (int64_t)idx * dim;
        float *yi = y + i * dim;
        memcpy(yi, row, (size_t)dim * sizeof(float));
        if (pos_embed) {
            const float *pe = pos_embed + i * dim;
            for (int64_t d = 0; d < dim; d++) yi[d] += pe[d];
        }
    }
}
void aria_rope_rotate_f32(const float *x, float *y, int64_t batch, int64_t seq, int64_t dim, float theta_base) {
    int64_t half_dim = dim / 2;
    /* Precompute frequencies on the stack (invariant across batch and seq).
       Use a fixed-size buffer for typical dims; fall back to alloca for larger. */
    float freqs_buf[1024];
    float *freqs_heap = NULL;
    float *freqs;
    if (half_dim <= 1024) {
        freqs = freqs_buf;
    } else {
        freqs_heap = (float *)malloc((size_t)half_dim * sizeof(float));
        freqs = freqs_heap ? freqs_heap : freqs_buf;
        if (!freqs_heap) half_dim = 1024; /* cap to stack buffer on OOM */
    }
    for (int64_t d = 0; d < half_dim; d++)
        freqs[d] = 1.0f / powf(theta_base, (float)(2 * d) / (float)dim);

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(batch * seq > 64)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            int64_t base = (b * seq + s) * dim;
            for (int64_t d = 0; d < half_dim; d++) {
                float angle = (float)s * freqs[d];
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x_even = x[base + 2 * d];
                float x_odd  = x[base + 2 * d + 1];
                y[base + 2 * d]     = x_even * cos_a - x_odd * sin_a;
                y[base + 2 * d + 1] = x_even * sin_a + x_odd * cos_a;
            }
            if (dim % 2 != 0)
                y[base + dim - 1] = x[base + dim - 1];
        }
    }
    free(freqs_heap);
}
}
