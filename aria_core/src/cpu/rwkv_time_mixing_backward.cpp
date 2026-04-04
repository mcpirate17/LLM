#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void aria_rwkv_time_mixing_backward_f32(const float *grad_out,
                                        const float *x,
                                        const float *w_decay,
                                        const float *u_bonus,
                                        const float *W_k,
                                        const float *W_v,
                                        const float *W_r,
                                        float *grad_x,
                                        float *grad_w_decay,
                                        float *grad_u_bonus,
                                        float *grad_W_k,
                                        float *grad_W_v,
                                        float *grad_W_r,
                                        int64_t batch,
                                        int64_t seq,
                                        int64_t dim) {
    if (!grad_out || !x || !w_decay || !u_bonus || !W_k || !W_v || !W_r || !grad_x ||
        !grad_w_decay || !grad_u_bonus || !grad_W_k || !grad_W_v || !grad_W_r ||
        batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    const int64_t total = batch * seq * dim;
    std::fill(grad_x, grad_x + total, 0.0f);
    std::fill(grad_w_decay, grad_w_decay + dim, 0.0f);
    std::fill(grad_u_bonus, grad_u_bonus + dim, 0.0f);
    std::fill(grad_W_k, grad_W_k + dim * dim, 0.0f);
    std::fill(grad_W_v, grad_W_v + dim * dim, 0.0f);
    std::fill(grad_W_r, grad_W_r + dim * dim, 0.0f);

    std::vector<float> k_buf(total, 0.0f);
    std::vector<float> v_buf(total, 0.0f);
    std::vector<float> r_raw_buf(total, 0.0f);
    std::vector<float> r_sig_buf(total, 0.0f);
    std::vector<float> a_prev_buf(total, 0.0f);
    std::vector<float> b_prev_buf(total, 0.0f);
    std::vector<float> ek_buf(total, 0.0f);
    std::vector<float> eku_buf(total, 0.0f);
    std::vector<float> grad_k_buf(total, 0.0f);
    std::vector<float> grad_v_buf(total, 0.0f);
    std::vector<float> grad_rraw_buf(total, 0.0f);

    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *kb = k_buf.data() + b * seq * dim;
        float *vb = v_buf.data() + b * seq * dim;
        float *rb = r_raw_buf.data() + b * seq * dim;
        aria_linear_f32(xb, W_k, NULL, kb, seq, dim, dim);
        aria_linear_f32(xb, W_v, NULL, vb, seq, dim, dim);
        aria_linear_f32(xb, W_r, NULL, rb, seq, dim, dim);
        aria_sigmoid_f32(rb, r_sig_buf.data() + b * seq * dim, seq * dim);

        for (int64_t d = 0; d < dim; d++) {
            const float ew = expf(-w_decay[d]);
            float a_prev = 0.0f;
            float b_prev = 0.0f;
            for (int64_t t = 0; t < seq; t++) {
                const int64_t idx = (b * seq + t) * dim + d;
                const float k = k_buf[idx];
                const float v = v_buf[idx];
                const float ku = u_bonus[d] + k;
                const float eku = expf(fminf(ku, 30.0f));
                const float ek = expf(fminf(k, 30.0f));
                a_prev_buf[idx] = a_prev;
                b_prev_buf[idx] = b_prev;
                eku_buf[idx] = eku;
                ek_buf[idx] = ek;
                a_prev = ew * a_prev + ek * v;
                b_prev = ew * b_prev + ek;
            }
        }
    }

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            const float ew = expf(-w_decay[d]);
            float grad_a_next = 0.0f;
            float grad_b_next = 0.0f;
            float grad_w = 0.0f;
            float grad_u = 0.0f;
            for (int64_t t = seq - 1; t >= 0; t--) {
                const int64_t idx = (b * seq + t) * dim + d;
                const float k = k_buf[idx];
                const float v = v_buf[idx];
                const float r_sig = r_sig_buf[idx];
                const float a_prev = a_prev_buf[idx];
                const float b_prev = b_prev_buf[idx];
                const float eku = eku_buf[idx];
                const float ek = ek_buf[idx];
                const float numer = a_prev + eku * v;
                const float denom = b_prev + eku + 1e-8f;
                const float wkv = numer / denom;

                const float grad_y = grad_out[idx];
                const float grad_wkv = grad_y * r_sig;
                const float grad_rsig = grad_y * wkv;
                const float grad_rraw = grad_rsig * r_sig * (1.0f - r_sig);

                float grad_a_prev = grad_wkv / denom;
                float grad_b_prev = -grad_wkv * numer / (denom * denom);
                float grad_eku =
                    (grad_wkv / denom) * v - (grad_wkv * numer) / (denom * denom);
                float grad_v = (grad_wkv / denom) * eku;

                grad_w += grad_a_next * (-ew * a_prev) + grad_b_next * (-ew * b_prev);
                grad_a_prev += grad_a_next * ew;
                grad_b_prev += grad_b_next * ew;

                float grad_k = 0.0f;
                if (k < 30.0f) {
                    grad_k += (grad_a_next * v + grad_b_next) * ek;
                }
                grad_v += grad_a_next * ek;
                if (u_bonus[d] + k < 30.0f) {
                    const float grad_ku = grad_eku * eku;
                    grad_k += grad_ku;
                    grad_u += grad_ku;
                }

                grad_k_buf[idx] = grad_k;
                grad_v_buf[idx] = grad_v;
                grad_rraw_buf[idx] = grad_rraw;
                grad_a_next = grad_a_prev;
                grad_b_next = grad_b_prev;
            }
            grad_w_decay[d] += grad_w;
            grad_u_bonus[d] += grad_u;
        }
    }

    for (int64_t row = 0; row < batch * seq; row++) {
        const float *x_row = x + row * dim;
        float *grad_x_row = grad_x + row * dim;
        const float *grad_k_row = grad_k_buf.data() + row * dim;
        const float *grad_v_row = grad_v_buf.data() + row * dim;
        const float *grad_r_row = grad_rraw_buf.data() + row * dim;

        for (int64_t out_d = 0; out_d < dim; out_d++) {
            const float gk = grad_k_row[out_d];
            const float gv = grad_v_row[out_d];
            const float gr = grad_r_row[out_d];
            const float *wk_row = W_k + out_d * dim;
            const float *wv_row = W_v + out_d * dim;
            const float *wr_row = W_r + out_d * dim;
            float *grad_wk_row = grad_W_k + out_d * dim;
            float *grad_wv_row = grad_W_v + out_d * dim;
            float *grad_wr_row = grad_W_r + out_d * dim;
            for (int64_t in_d = 0; in_d < dim; in_d++) {
                const float x_val = x_row[in_d];
                grad_x_row[in_d] += gk * wk_row[in_d] + gv * wv_row[in_d] + gr * wr_row[in_d];
                grad_wk_row[in_d] += gk * x_val;
                grad_wv_row[in_d] += gv * x_val;
                grad_wr_row[in_d] += gr * x_val;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
