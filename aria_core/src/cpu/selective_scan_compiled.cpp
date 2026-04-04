#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void aria_selective_scan_compiled_f32(const float *x,
                                      const float *A_log,
                                      const float *dt_proj,
                                      const float *B_weight,
                                      const float *C_weight,
                                      float *y,
                                      int64_t batch,
                                      int64_t seq,
                                      int64_t dim) {
    if (!x || !A_log || !dt_proj || !B_weight || !C_weight || !y) {
        return;
    }

    std::vector<float> a(dim);
    for (int64_t d = 0; d < dim; d++) {
        const float a_log = std::clamp(A_log[d], -10.0f, 10.0f);
        const float A = -std::exp(a_log);
        const float dt = std::log1pf(std::exp(dt_proj[d]));
        const float log_a = std::clamp(A * dt, -10.0f, -0.05f);
        a[d] = std::exp(log_a);
    }

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) if(batch * dim > 64) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float h = 0.0f;
            float u_prev = 0.0f;
            const float a_d = a[d];
            for (int64_t t = 0; t < seq; t++) {
                const float *x_row = x + (b * seq + t) * dim;
                float b_proj = 0.0f;
                float c_proj = 0.0f;
                const float *b_row = B_weight + d * dim;
                const float *c_row = C_weight + d * dim;
                for (int64_t i = 0; i < dim; i++) {
                    const float xv = x_row[i];
                    b_proj += xv * b_row[i];
                    c_proj += xv * c_row[i];
                }
                const float gate_b = 1.0f / (1.0f + std::exp(-b_proj));
                const float u = gate_b * x_row[d];
                const float u_trap = 0.5f * (u + a_d * u_prev);
                h = a_d * h + u_trap;
                const float gate_c = 1.0f / (1.0f + std::exp(-c_proj));
                y[(b * seq + t) * dim + d] = gate_c * h;
                u_prev = u;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
