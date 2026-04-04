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

namespace {

inline void linear_project(const float *x_row,
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

void aria_softmax_attention_backward_f32(const float *grad_out,
                                         const float *x,
                                         const float *Wq,
                                         const float *Wk,
                                         const float *Wv,
                                         const float *Wo,
                                         float *grad_x,
                                         float *grad_Wq,
                                         float *grad_Wk,
                                         float *grad_Wv,
                                         float *grad_Wo,
                                         int64_t batch,
                                         int64_t seq,
                                         int64_t dim,
                                         int64_t n_heads) {
    if (!grad_out || !x || !Wq || !Wk || !Wv || !Wo || !grad_x || !grad_Wq || !grad_Wk ||
        !grad_Wv || !grad_Wo || batch <= 0 || seq <= 0 || dim <= 0 || n_heads <= 0) {
        return;
    }

    const int64_t head_dim = dim / n_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::memset(grad_x, 0, sizeof(float) * batch * seq * dim);
    std::memset(grad_Wq, 0, sizeof(float) * dim * dim);
    std::memset(grad_Wk, 0, sizeof(float) * dim * dim);
    std::memset(grad_Wv, 0, sizeof(float) * dim * dim);
    std::memset(grad_Wo, 0, sizeof(float) * dim * dim);

    std::vector<float> q(batch * seq * dim);
    std::vector<float> k(batch * seq * dim);
    std::vector<float> v(batch * seq * dim);
    std::vector<float> pre_o(batch * seq * dim);
    std::vector<float> probs(seq * seq);
    std::vector<float> grad_pre(seq * dim);
    std::vector<float> grad_q(seq * dim, 0.0f);
    std::vector<float> grad_k(seq * dim, 0.0f);
    std::vector<float> grad_v(seq * dim, 0.0f);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t t = 0; t < seq; t++) {
            const float *x_row = x + (b * seq + t) * dim;
            linear_project(x_row, Wq, q.data() + (b * seq + t) * dim, dim);
            linear_project(x_row, Wk, k.data() + (b * seq + t) * dim, dim);
            linear_project(x_row, Wv, v.data() + (b * seq + t) * dim, dim);
        }

        for (int64_t h = 0; h < n_heads; h++) {
            const int64_t off = h * head_dim;
            std::fill(probs.begin(), probs.end(), 0.0f);
            for (int64_t i = 0; i < seq; i++) {
                float max_score = -INFINITY;
                for (int64_t j = 0; j <= i; j++) {
                    float score = 0.0f;
                    const float *q_row = q.data() + (b * seq + i) * dim + off;
                    const float *k_row = k.data() + (b * seq + j) * dim + off;
                    for (int64_t d = 0; d < head_dim; d++) {
                        score += q_row[d] * k_row[d];
                    }
                    score *= scale;
                    probs[i * seq + j] = score;
                    max_score = std::max(max_score, score);
                }
                float denom = 0.0f;
                for (int64_t j = 0; j <= i; j++) {
                    float p = std::exp(probs[i * seq + j] - max_score);
                    probs[i * seq + j] = p;
                    denom += p;
                }
                for (int64_t j = 0; j <= i; j++) {
                    probs[i * seq + j] /= denom;
                }
                float *pre_row = pre_o.data() + (b * seq + i) * dim + off;
                for (int64_t d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int64_t j = 0; j <= i; j++) {
                        sum += probs[i * seq + j] * v[(b * seq + j) * dim + off + d];
                    }
                    pre_row[d] = sum;
                }
            }
        }

        std::fill(grad_pre.begin(), grad_pre.end(), 0.0f);
        for (int64_t t = 0; t < seq; t++) {
            const float *go_row = grad_out + (b * seq + t) * dim;
            const float *pre_row = pre_o.data() + (b * seq + t) * dim;
            for (int64_t o = 0; o < dim; o++) {
                const float *wo_row = Wo + o * dim;
                for (int64_t i = 0; i < dim; i++) {
                    grad_pre[t * dim + i] += go_row[o] * wo_row[i];
                }
                for (int64_t i = 0; i < dim; i++) {
                    grad_Wo[o * dim + i] += go_row[o] * pre_row[i];
                }
            }
        }

        std::fill(grad_q.begin(), grad_q.end(), 0.0f);
        std::fill(grad_k.begin(), grad_k.end(), 0.0f);
        std::fill(grad_v.begin(), grad_v.end(), 0.0f);

        for (int64_t h = 0; h < n_heads; h++) {
            const int64_t off = h * head_dim;
            for (int64_t i = 0; i < seq; i++) {
                std::vector<float> grad_prob(i + 1, 0.0f);
                float dot = 0.0f;
                const float *gpre_row = grad_pre.data() + i * dim + off;
                for (int64_t j = 0; j <= i; j++) {
                    float gp = 0.0f;
                    const float *v_row = v.data() + (b * seq + j) * dim + off;
                    for (int64_t d = 0; d < head_dim; d++) {
                        gp += gpre_row[d] * v_row[d];
                        grad_v[j * dim + off + d] += probs[i * seq + j] * gpre_row[d];
                    }
                    grad_prob[j] = gp;
                    dot += gp * probs[i * seq + j];
                }
                for (int64_t j = 0; j <= i; j++) {
                    const float gscore = probs[i * seq + j] * (grad_prob[j] - dot);
                    const float *q_row = q.data() + (b * seq + i) * dim + off;
                    const float *k_row = k.data() + (b * seq + j) * dim + off;
                    for (int64_t d = 0; d < head_dim; d++) {
                        grad_q[i * dim + off + d] += scale * gscore * k_row[d];
                        grad_k[j * dim + off + d] += scale * gscore * q_row[d];
                    }
                }
            }
        }

        for (int64_t t = 0; t < seq; t++) {
            const float *x_row = x + (b * seq + t) * dim;
            float *gx_row = grad_x + (b * seq + t) * dim;
            for (int64_t o = 0; o < dim; o++) {
                const float gq = grad_q[t * dim + o];
                const float gk = grad_k[t * dim + o];
                const float gv = grad_v[t * dim + o];
                float *gwq_row = grad_Wq + o * dim;
                float *gwk_row = grad_Wk + o * dim;
                float *gwv_row = grad_Wv + o * dim;
                const float *wq_row = Wq + o * dim;
                const float *wk_row = Wk + o * dim;
                const float *wv_row = Wv + o * dim;
                for (int64_t i = 0; i < dim; i++) {
                    gwq_row[i] += gq * x_row[i];
                    gwk_row[i] += gk * x_row[i];
                    gwv_row[i] += gv * x_row[i];
                    gx_row[i] += gq * wq_row[i] + gk * wk_row[i] + gv * wv_row[i];
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
