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

inline float selective_scan_backward_sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline bool clamp_active(float value, float lo, float hi) {
    return value > lo && value < hi;
}

}  // namespace

void aria_selective_scan_compiled_backward_f32(const float *grad_out,
                                               const float *x,
                                               const float *A_log,
                                               const float *dt_proj,
                                               const float *B_weight,
                                               const float *C_weight,
                                               float *grad_x,
                                               float *grad_A_log,
                                               float *grad_dt_proj,
                                               float *grad_B_weight,
                                               float *grad_C_weight,
                                               int64_t batch,
                                               int64_t seq,
                                               int64_t dim) {
    if (!grad_out || !x || !A_log || !dt_proj || !B_weight || !C_weight ||
        !grad_x || !grad_A_log || !grad_dt_proj || !grad_B_weight || !grad_C_weight) {
        return;
    }

    std::memset(grad_x, 0, sizeof(float) * batch * seq * dim);
    std::memset(grad_A_log, 0, sizeof(float) * dim);
    std::memset(grad_dt_proj, 0, sizeof(float) * dim);
    std::memset(grad_B_weight, 0, sizeof(float) * dim * dim);
    std::memset(grad_C_weight, 0, sizeof(float) * dim * dim);

    std::vector<float> a(dim);
    std::vector<float> dt(dim);
    std::vector<float> raw_a(dim);
    std::vector<float> clamped_a_log(dim);
    std::vector<float> raw_dt(dim);
    for (int64_t d = 0; d < dim; d++) {
        clamped_a_log[d] = std::clamp(A_log[d], -10.0f, 10.0f);
        raw_a[d] = -std::exp(clamped_a_log[d]);
        raw_dt[d] = dt_proj[d];
        dt[d] = std::log1pf(std::exp(raw_dt[d]));
        const float log_a = std::clamp(raw_a[d] * dt[d], -10.0f, -0.05f);
        a[d] = std::exp(log_a);
    }

    std::vector<float> h(seq);
    std::vector<float> u(seq);
    std::vector<float> gate_b(seq);
    std::vector<float> gate_c(seq);
    std::vector<float> b_proj(seq);
    std::vector<float> c_proj(seq);
    std::vector<float> grad_u(seq);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            const float a_d = a[d];
            const float *b_row = B_weight + d * dim;
            const float *c_row = C_weight + d * dim;
            float h_prev = 0.0f;
            float u_prev = 0.0f;

            for (int64_t t = 0; t < seq; t++) {
                const float *x_row = x + (b * seq + t) * dim;
                float b_sum = 0.0f;
                float c_sum = 0.0f;
                for (int64_t i = 0; i < dim; i++) {
                    b_sum += x_row[i] * b_row[i];
                    c_sum += x_row[i] * c_row[i];
                }
                b_proj[t] = b_sum;
                c_proj[t] = c_sum;
                gate_b[t] = selective_scan_backward_sigmoid_scalar(b_sum);
                gate_c[t] = selective_scan_backward_sigmoid_scalar(c_sum);
                u[t] = gate_b[t] * x_row[d];
                const float u_trap = 0.5f * (u[t] + a_d * u_prev);
                h[t] = a_d * h_prev + u_trap;
                h_prev = h[t];
                u_prev = u[t];
                grad_u[t] = 0.0f;
            }

            float grad_a_scalar = 0.0f;
            float grad_h_carry = 0.0f;
            for (int64_t t = seq - 1; t >= 0; t--) {
                const float *x_row = x + (b * seq + t) * dim;
                const float *go_row = grad_out + (b * seq + t) * dim;
                float *gx_row = grad_x + (b * seq + t) * dim;
                const float h_prev_t = (t > 0) ? h[t - 1] : 0.0f;
                const float u_prev_t = (t > 0) ? u[t - 1] : 0.0f;

                const float grad_y = go_row[d];
                const float gate_c_t = gate_c[t];
                const float grad_c_pre = grad_y * h[t] * gate_c_t * (1.0f - gate_c_t);
                for (int64_t i = 0; i < dim; i++) {
                    grad_C_weight[d * dim + i] += grad_c_pre * x_row[i];
                    gx_row[i] += grad_c_pre * c_row[i];
                }

                const float grad_h = grad_h_carry + grad_y * gate_c_t;
                grad_a_scalar += grad_h * (h_prev_t + 0.5f * u_prev_t);
                grad_h_carry = grad_h * a_d;
                grad_u[t] += 0.5f * grad_h;
                if (t > 0) {
                    grad_u[t - 1] += 0.5f * grad_h * a_d;
                }
            }

            for (int64_t t = 0; t < seq; t++) {
                const float *x_row = x + (b * seq + t) * dim;
                float *gx_row = grad_x + (b * seq + t) * dim;
                const float gb_pre = grad_u[t] * x_row[d] * gate_b[t] * (1.0f - gate_b[t]);
                for (int64_t i = 0; i < dim; i++) {
                    grad_B_weight[d * dim + i] += gb_pre * x_row[i];
                    gx_row[i] += gb_pre * b_row[i];
                }
                gx_row[d] += grad_u[t] * gate_b[t];
            }

            const float raw_log_a = raw_a[d] * dt[d];
            if (clamp_active(raw_log_a, -10.0f, -0.05f)) {
                const float grad_log_a = grad_a_scalar * a_d;
                const float grad_raw_a = grad_log_a * dt[d];
                const float grad_dt = grad_log_a * raw_a[d];
                if (clamp_active(A_log[d], -10.0f, 10.0f)) {
                    grad_A_log[d] += grad_raw_a * raw_a[d];
                }
                grad_dt_proj[d] += grad_dt * selective_scan_backward_sigmoid_scalar(raw_dt[d]);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
