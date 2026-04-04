#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void aria_state_space_compiled_f32(const float *x,
                                   const float *ssm_A,
                                   const float *ssm_B_weight,
                                   const float *ssm_C_weight,
                                   const float *ssm_D,
                                   const float *ssm_dt_weight,
                                   const float *ssm_dt_bias,
                                   float *y,
                                   int64_t batch,
                                   int64_t seq,
                                   int64_t dim,
                                   int64_t state_dim) {
    if (!x || !ssm_A || !ssm_B_weight || !ssm_C_weight || !ssm_D || !ssm_dt_weight ||
        !ssm_dt_bias || !y || dim <= 0 || state_dim <= 0) {
        return;
    }

    const float inv_sqrt_state = 1.0f / std::sqrt(static_cast<float>(state_dim));

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > 1) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        std::vector<float> state(dim * state_dim, 0.0f);
        std::vector<float> dt(dim, 0.0f);
        std::vector<float> b_proj(dim * state_dim, 0.0f);
        std::vector<float> h_clamped(dim * state_dim, 0.0f);

        for (int64_t t = 0; t < seq; t++) {
            const float *x_row = x + (b * seq + t) * dim;
            float *y_row = y + (b * seq + t) * dim;

            for (int64_t d = 0; d < dim; d++) {
                float dt_linear = ssm_dt_bias[d];
                const float *dt_row = ssm_dt_weight + d * dim;
                for (int64_t i = 0; i < dim; i++) {
                    dt_linear += x_row[i] * dt_row[i];
                }
                dt[d] = std::log1pf(std::exp(dt_linear));
            }

            for (int64_t lane = 0; lane < dim * state_dim; lane++) {
                float sum = 0.0f;
                const float *b_row = ssm_B_weight + lane * dim;
                for (int64_t i = 0; i < dim; i++) {
                    sum += x_row[i] * b_row[i];
                }
                b_proj[lane] = sum;
            }

            for (int64_t d = 0; d < dim; d++) {
                const float dt_d = dt[d];
                for (int64_t n = 0; n < state_dim; n++) {
                    const int64_t idx = d * state_dim + n;
                    const float log_a =
                        std::clamp(ssm_A[idx] * dt_d, -10.0f, 0.0f);
                    state[idx] = std::exp(log_a) * state[idx] + b_proj[idx];
                    h_clamped[idx] = std::clamp(state[idx], -50.0f, 50.0f);
                }
            }

            for (int64_t out_d = 0; out_d < dim; out_d++) {
                float sum = 0.0f;
                const float *c_row = ssm_C_weight + out_d * (dim * state_dim);
                for (int64_t i = 0; i < dim * state_dim; i++) {
                    sum += c_row[i] * h_clamped[i];
                }
                y_row[out_d] = sum * inv_sqrt_state + x_row[out_d] * ssm_D[out_d];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
