#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif
#include "compiled_kernel_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void aria_state_space_compiled_backward_f32(const float *grad_out,
                                            const float *x,
                                            const float *ssm_A,
                                            const float *ssm_B_weight,
                                            const float *ssm_C_weight,
                                            const float *ssm_D,
                                            const float *ssm_dt_weight,
                                            const float *ssm_dt_bias,
                                            float *grad_x,
                                            float *grad_ssm_A,
                                            float *grad_ssm_B_weight,
                                            float *grad_ssm_C_weight,
                                            float *grad_ssm_D,
                                            float *grad_ssm_dt_weight,
                                            float *grad_ssm_dt_bias,
                                            int64_t batch,
                                            int64_t seq,
                                            int64_t dim,
                                            int64_t state_dim) {
    if (!grad_out || !x || !ssm_A || !ssm_B_weight || !ssm_C_weight || !ssm_D ||
        !ssm_dt_weight || !ssm_dt_bias || !grad_x || !grad_ssm_A ||
        !grad_ssm_B_weight || !grad_ssm_C_weight || !grad_ssm_D ||
        !grad_ssm_dt_weight || !grad_ssm_dt_bias || state_dim <= 0) {
        return;
    }

    const int64_t state_size = dim * state_dim;
    const float inv_sqrt_state = 1.0f / std::sqrt(static_cast<float>(state_dim));
    std::memset(grad_x, 0, sizeof(float) * batch * seq * dim);
    std::memset(grad_ssm_A, 0, sizeof(float) * state_size);
    std::memset(grad_ssm_B_weight, 0, sizeof(float) * state_size * dim);
    std::memset(grad_ssm_C_weight, 0, sizeof(float) * dim * state_size);
    std::memset(grad_ssm_D, 0, sizeof(float) * dim);
    std::memset(grad_ssm_dt_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_ssm_dt_bias, 0, sizeof(float) * dim);

    std::vector<float> dt_linear(seq * dim);
    std::vector<float> dt(seq * dim);
    std::vector<float> state(seq * state_size);
    std::vector<float> h_clamped(seq * state_size);
    std::vector<float> b_proj(seq * state_size);
    std::vector<float> grad_state_carry(state_size, 0.0f);
    std::vector<float> grad_state_now(state_size, 0.0f);
    std::vector<float> grad_dt_acc(seq * dim, 0.0f);

    for (int64_t b = 0; b < batch; b++) {
        std::fill(grad_state_carry.begin(), grad_state_carry.end(), 0.0f);
        std::fill(grad_dt_acc.begin(), grad_dt_acc.end(), 0.0f);
        std::vector<float> prev_state(state_size, 0.0f);

        for (int64_t t = 0; t < seq; t++) {
            const float *x_row = x + (b * seq + t) * dim;
            float *state_t = state.data() + t * state_size;
            float *h_t = h_clamped.data() + t * state_size;
            float *bproj_t = b_proj.data() + t * state_size;
            float *dt_linear_t = dt_linear.data() + t * dim;
            float *dt_t = dt.data() + t * dim;

            for (int64_t d = 0; d < dim; d++) {
                float dt_sum = ssm_dt_bias[d];
                const float *dt_row = ssm_dt_weight + d * dim;
                for (int64_t i = 0; i < dim; i++) {
                    dt_sum += x_row[i] * dt_row[i];
                }
                dt_linear_t[d] = dt_sum;
                dt_t[d] = std::log1pf(std::exp(dt_sum));
            }

            for (int64_t idx = 0; idx < state_size; idx++) {
                const float *b_row = ssm_B_weight + idx * dim;
                float sum = 0.0f;
                for (int64_t i = 0; i < dim; i++) {
                    sum += x_row[i] * b_row[i];
                }
                bproj_t[idx] = sum;
            }

            for (int64_t d = 0; d < dim; d++) {
                for (int64_t n = 0; n < state_dim; n++) {
                    const int64_t idx = d * state_dim + n;
                    const float raw_log_a = ssm_A[idx] * dt_t[d];
                    const float log_a = std::clamp(raw_log_a, -10.0f, 0.0f);
                    const float a = std::exp(log_a);
                    state_t[idx] = a * prev_state[idx] + bproj_t[idx];
                    h_t[idx] = std::clamp(state_t[idx], -50.0f, 50.0f);
                    prev_state[idx] = state_t[idx];
                }
            }
        }

        for (int64_t t = seq - 1; t >= 0; t--) {
            const float *x_row = x + (b * seq + t) * dim;
            const float *go_row = grad_out + (b * seq + t) * dim;
            float *gx_row = grad_x + (b * seq + t) * dim;
            const float *state_t = state.data() + t * state_size;
            const float *h_t = h_clamped.data() + t * state_size;
            const float *dt_t = dt.data() + t * dim;
            const float *dt_linear_t = dt_linear.data() + t * dim;
            const float *prev_state_t = (t > 0) ? (state.data() + (t - 1) * state_size) : nullptr;

            std::fill(grad_state_now.begin(), grad_state_now.end(), 0.0f);

            for (int64_t out_d = 0; out_d < dim; out_d++) {
                grad_ssm_D[out_d] += go_row[out_d] * x_row[out_d];
                gx_row[out_d] += go_row[out_d] * ssm_D[out_d];
                float *gc_row = grad_ssm_C_weight + out_d * state_size;
                const float *c_row = ssm_C_weight + out_d * state_size;
                for (int64_t idx = 0; idx < state_size; idx++) {
                    gc_row[idx] += go_row[out_d] * h_t[idx] * inv_sqrt_state;
                    grad_state_now[idx] += go_row[out_d] * c_row[idx] * inv_sqrt_state;
                }
            }

            for (int64_t idx = 0; idx < state_size; idx++) {
                float total_grad = grad_state_now[idx] + grad_state_carry[idx];
                if (!(state_t[idx] > -50.0f && state_t[idx] < 50.0f)) {
                    total_grad = 0.0f;
                }

                const int64_t d = idx / state_dim;
                const float prev = prev_state_t ? prev_state_t[idx] : 0.0f;
                const float raw_log_a = ssm_A[idx] * dt_t[d];
                const float log_a = std::clamp(raw_log_a, -10.0f, 0.0f);
                const float a = std::exp(log_a);

                if (aria_ck_unclamped(raw_log_a, -10.0f, 0.0f)) {
                    const float grad_log_a = total_grad * a * prev;
                    grad_ssm_A[idx] += grad_log_a * dt_t[d];
                    grad_dt_acc[t * dim + d] += grad_log_a * ssm_A[idx];
                }

                const float *b_row = ssm_B_weight + idx * dim;
                float *gb_row = grad_ssm_B_weight + idx * dim;
                for (int64_t i = 0; i < dim; i++) {
                    gb_row[i] += total_grad * x_row[i];
                    gx_row[i] += total_grad * b_row[i];
                }
                grad_state_carry[idx] = total_grad * a;
            }

            for (int64_t d = 0; d < dim; d++) {
                const float g_dt_lin =
                    grad_dt_acc[t * dim + d] * aria_ck_sigmoid(dt_linear_t[d]);
                grad_ssm_dt_bias[d] += g_dt_lin;
                float *gdt_row = grad_ssm_dt_weight + d * dim;
                const float *dt_row = ssm_dt_weight + d * dim;
                for (int64_t i = 0; i < dim; i++) {
                    gdt_row[i] += g_dt_lin * x_row[i];
                    gx_row[i] += g_dt_lin * dt_row[i];
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
