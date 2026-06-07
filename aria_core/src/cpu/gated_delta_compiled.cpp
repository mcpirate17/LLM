#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// Positive forget-gate-bias init on the retention gate: the decay is
// sigmoid(alpha_logit + kGatedDeltaDecayBias), so at init (logits ≈ 0) the gate
// is ≈0.92 and the recurrent state is *kept*, not wiped. Without it the gate sits
// at 0.5 and — combined with decay = alpha (a true retention gate, not the old
// alpha - beta which centred decay at 0) — the state dies before training can
// bias it (mamba2 baseline scored 0.0 everywhere; diagnosed 2026-06-07). Must
// stay in lockstep with the torch reference `_op_gated_delta`
// (_GATED_DELTA_DECAY_BIAS in research/synthesis/compiler_ops_sequence.py).
constexpr float kGatedDeltaDecayBias = 2.5f;

inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline void linear_project_token(const float *x_row,
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

}  // namespace

void aria_gated_delta_compiled_f32(const float *x,
                                   const float *q_weight,
                                   const float *k_weight,
                                   const float *v_weight,
                                   const float *alpha_weight,
                                   const float *beta_weight,
                                   const float *o_weight,
                                   float *y,
                                   int64_t batch,
                                   int64_t seq,
                                   int64_t dim,
                                   int64_t n_heads) {
    if (!x || !q_weight || !k_weight || !v_weight || !alpha_weight || !beta_weight ||
        !o_weight || !y || batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    int64_t heads = n_heads;
    if (heads <= 0) {
        heads = std::min<int64_t>(8, dim);
    }
    if (heads <= 0 || dim % heads != 0) {
        heads = 1;
    }
    const int64_t head_dim = dim / heads;
    const int64_t chunk = std::min<int64_t>(32, seq);

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > 1) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        std::vector<float> q(dim);
        std::vector<float> k(dim);
        std::vector<float> v(dim);
        std::vector<float> alpha(dim);
        std::vector<float> beta(dim);
        std::vector<float> decay(dim);
        std::vector<float> pre_out(dim);
        std::vector<float> state(heads * head_dim * head_dim, 0.0f);

        for (int64_t c_start = 0; c_start < seq; c_start += chunk) {
            const int64_t c_end = std::min<int64_t>(c_start + chunk, seq);
            for (int64_t t = c_start; t < c_end; t++) {
                const float *x_row = x + (b * seq + t) * dim;
                float *y_row = y + (b * seq + t) * dim;

                linear_project_token(x_row, q_weight, q.data(), dim);
                linear_project_token(x_row, k_weight, k.data(), dim);
                linear_project_token(x_row, v_weight, v.data(), dim);
                linear_project_token(x_row, alpha_weight, alpha.data(), dim);
                linear_project_token(x_row, beta_weight, beta.data(), dim);

                for (int64_t i = 0; i < dim; i++) {
                    alpha[i] = sigmoid_scalar(alpha[i] + kGatedDeltaDecayBias);
                    beta[i] = sigmoid_scalar(beta[i]);
                    decay[i] = alpha[i];
                }

                for (int64_t h = 0; h < heads; h++) {
                    const int64_t head_off = h * head_dim;
                    float *state_h = state.data() + h * head_dim * head_dim;
                    const bool chunk_start = (t == c_start);

                    for (int64_t row = 0; row < head_dim; row++) {
                        const float beta_v = beta[head_off + row] * v[head_off + row];
                        const float decay_val = chunk_start
                                                    ? decay[head_off + row]
                                                    : std::max(decay[head_off + row], 1e-8f);
                        float *state_row = state_h + row * head_dim;
                        for (int64_t col = 0; col < head_dim; col++) {
                            state_row[col] =
                                beta_v * k[head_off + col] + decay_val * state_row[col];
                        }
                    }

                    for (int64_t col = 0; col < head_dim; col++) {
                        float sum = 0.0f;
                        for (int64_t row = 0; row < head_dim; row++) {
                            sum += q[head_off + row] * state_h[row * head_dim + col];
                        }
                        pre_out[head_off + col] = sum;
                    }
                }

                linear_project_token(pre_out.data(), o_weight, y_row, dim);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
