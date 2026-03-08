#include "kernels.h"

#include <stdint.h>
#include <string.h>

void aria_conditional_dispatch_f32(const float *x, const int64_t *assignments, int64_t lane_id,
                                   float *lane_out, int64_t *index_map, int64_t *lane_counts,
                                   int64_t batch, int64_t seq, int64_t dim) {
    if (!x || !assignments || !lane_out || !index_map || !lane_counts ||
        batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    memset(lane_out, 0, (size_t)(batch * seq * dim) * sizeof(float));

    for (int64_t b = 0; b < batch; ++b) {
        int64_t write_pos = 0;
        for (int64_t s = 0; s < seq; ++s) {
            int64_t src_idx = b * seq + s;
            if (assignments[src_idx] == lane_id) {
                index_map[src_idx] = write_pos;
                const float *src = x + (src_idx * dim);
                float *dst = lane_out + ((b * seq + write_pos) * dim);
                memcpy(dst, src, (size_t)dim * sizeof(float));
                write_pos++;
            } else {
                index_map[src_idx] = -1;
            }
        }
        lane_counts[b] = write_pos;
    }
}

void aria_conditional_dispatch_backward_f32(const float *lane_grad, const int64_t *index_map,
                                            float *grad_x, int64_t batch, int64_t seq, int64_t dim) {
    if (!lane_grad || !index_map || !grad_x || batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    memset(grad_x, 0, (size_t)(batch * seq * dim) * sizeof(float));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            int64_t src_idx = b * seq + s;
            int64_t packed_pos = index_map[src_idx];
            if (packed_pos < 0 || packed_pos >= seq) {
                continue;
            }
            const float *src = lane_grad + ((b * seq + packed_pos) * dim);
            float *dst = grad_x + (src_idx * dim);
            memcpy(dst, src, (size_t)dim * sizeof(float));
        }
    }
}

void aria_conditional_gather_f32(const float *lane_out, const int64_t *index_map, const float *weights,
                                 float *y, int64_t batch, int64_t seq, int64_t dim) {
    if (!lane_out || !index_map || !weights || !y || batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            int64_t src_idx = b * seq + s;
            int64_t packed_pos = index_map[src_idx];
            if (packed_pos < 0 || packed_pos >= seq) {
                continue;
            }
            float w = weights[src_idx];
            const float *lane_vec = lane_out + ((b * seq + packed_pos) * dim);
            float *dst = y + (src_idx * dim);
            for (int64_t d = 0; d < dim; ++d) {
                dst[d] += w * lane_vec[d];
            }
        }
    }
}

void aria_conditional_gather_backward_f32(const float *grad_y, const float *lane_out,
                                          const int64_t *index_map, const float *weights,
                                          float *grad_lane, float *grad_weights,
                                          int64_t batch, int64_t seq, int64_t dim) {
    if (!grad_y || !lane_out || !index_map || !weights || !grad_lane || !grad_weights ||
        batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }

    memset(grad_lane, 0, (size_t)(batch * seq * dim) * sizeof(float));
    memset(grad_weights, 0, (size_t)(batch * seq) * sizeof(float));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            int64_t src_idx = b * seq + s;
            int64_t packed_pos = index_map[src_idx];
            if (packed_pos < 0 || packed_pos >= seq) {
                continue;
            }
            float w = weights[src_idx];
            const float *gy = grad_y + (src_idx * dim);
            const float *lane_vec = lane_out + ((b * seq + packed_pos) * dim);
            float *gl = grad_lane + ((b * seq + packed_pos) * dim);

            float gw = 0.0f;
            for (int64_t d = 0; d < dim; ++d) {
                gl[d] += w * gy[d];
                gw += gy[d] * lane_vec[d];
            }
            grad_weights[src_idx] = gw;
        }
    }
}
