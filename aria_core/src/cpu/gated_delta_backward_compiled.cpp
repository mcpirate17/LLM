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

void aria_gated_delta_compiled_backward_f32(const float *grad_out,
                                            const float *x,
                                            const float *q_weight,
                                            const float *k_weight,
                                            const float *v_weight,
                                            const float *alpha_weight,
                                            const float *beta_weight,
                                            const float *o_weight,
                                            float *grad_x,
                                            float *grad_q_weight,
                                            float *grad_k_weight,
                                            float *grad_v_weight,
                                            float *grad_alpha_weight,
                                            float *grad_beta_weight,
                                            float *grad_o_weight,
                                            int64_t batch,
                                            int64_t seq,
                                            int64_t dim,
                                            int64_t n_heads) {
    if (!grad_out || !x || !q_weight || !k_weight || !v_weight || !alpha_weight ||
        !beta_weight || !o_weight || !grad_x || !grad_q_weight || !grad_k_weight ||
        !grad_v_weight || !grad_alpha_weight || !grad_beta_weight || !grad_o_weight ||
        batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    int64_t heads = n_heads > 0 ? n_heads : std::min<int64_t>(8, dim);
    if (heads <= 0 || dim % heads != 0) {
        heads = 1;
    }
    const int64_t head_dim = dim / heads;
    const int64_t state_size = heads * head_dim * head_dim;
    const int64_t chunk = std::min<int64_t>(32, seq);

    std::memset(grad_x, 0, sizeof(float) * batch * seq * dim);
    std::memset(grad_q_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_k_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_v_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_alpha_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_beta_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_o_weight, 0, sizeof(float) * dim * dim);

    std::vector<float> q(seq * dim);
    std::vector<float> k(seq * dim);
    std::vector<float> v(seq * dim);
    std::vector<float> alpha(seq * dim);
    std::vector<float> beta(seq * dim);
    std::vector<float> decay(seq * dim);
    std::vector<float> pre_o(seq * dim);
    std::vector<float> state_history(seq * state_size);
    std::vector<float> prev_history(seq * state_size);
    std::vector<float> grad_pre(seq * dim);
    std::vector<float> grad_q(seq * dim, 0.0f);
    std::vector<float> grad_k(seq * dim, 0.0f);
    std::vector<float> grad_v(seq * dim, 0.0f);
    std::vector<float> grad_alpha(seq * dim, 0.0f);
    std::vector<float> grad_beta(seq * dim, 0.0f);
    std::vector<float> grad_state_carry(state_size, 0.0f);
    std::vector<float> state(state_size, 0.0f);

    for (int64_t b = 0; b < batch; b++) {
        std::fill(state.begin(), state.end(), 0.0f);
        for (int64_t c_start = 0; c_start < seq; c_start += chunk) {
            const int64_t c_end = std::min<int64_t>(c_start + chunk, seq);
            for (int64_t t = c_start; t < c_end; t++) {
                const float *x_row = x + (b * seq + t) * dim;
                aria_ck_linear_project(x_row, q_weight, q.data() + t * dim, dim);
                aria_ck_linear_project(x_row, k_weight, k.data() + t * dim, dim);
                aria_ck_linear_project(x_row, v_weight, v.data() + t * dim, dim);
                aria_ck_linear_project(
                    x_row, alpha_weight, alpha.data() + t * dim, dim);
                aria_ck_linear_project(
                    x_row, beta_weight, beta.data() + t * dim, dim);
                std::memcpy(prev_history.data() + t * state_size, state.data(), sizeof(float) * state_size);

                for (int64_t i = 0; i < dim; i++) {
                    alpha[t * dim + i] = aria_ck_sigmoid(
                        alpha[t * dim + i] + kGatedDeltaDecayBias);
                    beta[t * dim + i] = aria_ck_sigmoid(beta[t * dim + i]);
                    decay[t * dim + i] = alpha[t * dim + i];
                }

                for (int64_t h = 0; h < heads; h++) {
                    const int64_t off = h * head_dim;
                    float *state_h = state.data() + h * head_dim * head_dim;
                    const bool chunk_start = (t == c_start);
                    for (int64_t row = 0; row < head_dim; row++) {
                        const float beta_v = beta[t * dim + off + row] * v[t * dim + off + row];
                        const float decay_val = chunk_start
                                                    ? decay[t * dim + off + row]
                                                    : std::max(decay[t * dim + off + row], 1e-8f);
                        float *state_row = state_h + row * head_dim;
                        for (int64_t col = 0; col < head_dim; col++) {
                            state_row[col] =
                                beta_v * k[t * dim + off + col] + decay_val * state_row[col];
                        }
                    }
                    for (int64_t col = 0; col < head_dim; col++) {
                        float sum = 0.0f;
                        for (int64_t row = 0; row < head_dim; row++) {
                            sum += q[t * dim + off + row] * state_h[row * head_dim + col];
                        }
                        pre_o[t * dim + off + col] = sum;
                    }
                }
                std::memcpy(state_history.data() + t * state_size, state.data(), sizeof(float) * state_size);
            }
        }

        std::fill(grad_pre.begin(), grad_pre.end(), 0.0f);
        for (int64_t t = 0; t < seq; t++) {
            const float *go_row = grad_out + (b * seq + t) * dim;
            const float *pre_row = pre_o.data() + t * dim;
            for (int64_t o = 0; o < dim; o++) {
                const float *wo_row = o_weight + o * dim;
                for (int64_t i = 0; i < dim; i++) {
                    grad_pre[t * dim + i] += go_row[o] * wo_row[i];
                    grad_o_weight[o * dim + i] += go_row[o] * pre_row[i];
                }
            }
        }

        std::fill(grad_q.begin(), grad_q.end(), 0.0f);
        std::fill(grad_k.begin(), grad_k.end(), 0.0f);
        std::fill(grad_v.begin(), grad_v.end(), 0.0f);
        std::fill(grad_alpha.begin(), grad_alpha.end(), 0.0f);
        std::fill(grad_beta.begin(), grad_beta.end(), 0.0f);
        std::fill(grad_state_carry.begin(), grad_state_carry.end(), 0.0f);

        for (int64_t t = seq - 1; t >= 0; t--) {
            const float *x_row = x + (b * seq + t) * dim;
            float *gx_row = grad_x + (b * seq + t) * dim;
            const float *state_t = state_history.data() + t * state_size;
            const float *prev_t = prev_history.data() + t * state_size;

            for (int64_t h = 0; h < heads; h++) {
                const int64_t off = h * head_dim;
                const bool chunk_start = (t % chunk == 0);
                const float *state_h = state_t + h * head_dim * head_dim;
                const float *prev_h = prev_t + h * head_dim * head_dim;
                std::vector<float> grad_state(head_dim * head_dim, 0.0f);

                for (int64_t row = 0; row < head_dim; row++) {
                    float gq = 0.0f;
                    for (int64_t col = 0; col < head_dim; col++) {
                        const float gpre = grad_pre[t * dim + off + col];
                        gq += gpre * state_h[row * head_dim + col];
                        grad_state[row * head_dim + col] += gpre * q[t * dim + off + row];
                    }
                    grad_q[t * dim + off + row] += gq;
                }

                for (int64_t row = 0; row < head_dim; row++) {
                    float grad_beta_v = 0.0f;
                    float grad_decay_row = 0.0f;
                    const float beta_val = beta[t * dim + off + row];
                    const float v_val = v[t * dim + off + row];
                    const float alpha_val = alpha[t * dim + off + row];
                    const float raw_decay = decay[t * dim + off + row];
                    for (int64_t col = 0; col < head_dim; col++) {
                        const float total_state_grad =
                            grad_state[row * head_dim + col] + grad_state_carry[h * head_dim * head_dim + row * head_dim + col];
                        grad_beta_v += total_state_grad * k[t * dim + off + col];
                        grad_k[t * dim + off + col] += total_state_grad * beta_val * v_val;
                        grad_decay_row += total_state_grad * prev_h[row * head_dim + col];
                        grad_state_carry[h * head_dim * head_dim + row * head_dim + col] =
                            total_state_grad * (chunk_start ? raw_decay : std::max(raw_decay, 1e-8f));
                    }
                    grad_v[t * dim + off + row] += grad_beta_v * beta_val;
                    float grad_beta_total = grad_beta_v * v_val;
                    float grad_decay_total = grad_decay_row;
                    if (!chunk_start && raw_decay <= 1e-8f) {
                        grad_decay_total = 0.0f;
                    }
                    // decay = alpha (retention gate): ∂decay/∂alpha = 1,
                    // ∂decay/∂beta = 0. (Old alpha - beta had ∂/∂beta = -1.)
                    grad_alpha[t * dim + off + row] += grad_decay_total;
                    grad_beta[t * dim + off + row] += grad_beta_total;
                    grad_alpha[t * dim + off + row] *= alpha_val * (1.0f - alpha_val);
                    grad_beta[t * dim + off + row] *= beta_val * (1.0f - beta_val);
                }
            }

            for (int64_t o = 0; o < dim; o++) {
                const float gq = grad_q[t * dim + o];
                const float gk = grad_k[t * dim + o];
                const float gv = grad_v[t * dim + o];
                const float ga = grad_alpha[t * dim + o];
                const float gb = grad_beta[t * dim + o];
                const float *q_row = q_weight + o * dim;
                const float *k_row = k_weight + o * dim;
                const float *v_row = v_weight + o * dim;
                const float *a_row = alpha_weight + o * dim;
                const float *b_row = beta_weight + o * dim;
                float *gq_row = grad_q_weight + o * dim;
                float *gk_row = grad_k_weight + o * dim;
                float *gv_row = grad_v_weight + o * dim;
                float *ga_row = grad_alpha_weight + o * dim;
                float *gb_row = grad_beta_weight + o * dim;
                for (int64_t i = 0; i < dim; i++) {
                    gq_row[i] += gq * x_row[i];
                    gk_row[i] += gk * x_row[i];
                    gv_row[i] += gv * x_row[i];
                    ga_row[i] += ga * x_row[i];
                    gb_row[i] += gb * x_row[i];
                    gx_row[i] += gq * q_row[i] + gk * k_row[i] + gv * v_row[i] + ga * a_row[i] + gb * b_row[i];
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
