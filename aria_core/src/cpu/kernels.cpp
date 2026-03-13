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
#include "backward.cpp"
#include "io.cpp"
#include "adaptive_routing.cpp"
#include "routing.c"
#include "dispatch.c"
#include "binding_stubs.cpp"
#include "fp16.cpp"
#include "smoke_test.cpp"
#include "fingerprint_metrics.cpp"

extern "C" {
void aria_argsort_seq_f32(const float *x, int64_t *indices, int64_t batch, int64_t seq, int64_t dim) {}
void aria_embedding_lookup_f32(const float *table, const int32_t *indices, const float *pos_embed, float *y, int64_t batch, int64_t dim, int64_t vocab_size) {}
void aria_rope_rotate_f32(const float *x, float *y, int64_t batch, int64_t seq, int64_t dim, float theta_base) {
    int64_t half_dim = dim / 2;
    /* Precompute frequencies on the stack (invariant across batch and seq).
       Use a fixed-size buffer for typical dims; fall back to alloca for larger. */
    float freqs_buf[1024];
    float *freqs = (half_dim <= 1024) ? freqs_buf : (float *)alloca((size_t)half_dim * sizeof(float));
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
}
}
