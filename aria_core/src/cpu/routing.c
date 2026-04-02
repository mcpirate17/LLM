#include "kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float _aria_sigmoidf(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

void aria_difficulty_scorer_f32(const float *x,
                                const float *w1, const float *b1,
                                const float *w2, const float *b2,
                                float *scores,
                                int64_t batch, int64_t seq, int64_t dim, int64_t hidden_dim) {
    if (!x || !w1 || !w2 || !scores || batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }
    if (hidden_dim <= 0) {
        hidden_dim = dim / 8;
        if (hidden_dim < 1) {
            hidden_dim = 1;
        }
    }

    arena_reset();
    float *hidden = (float *)arena_alloc((size_t)hidden_dim * sizeof(float));
    if (!hidden) {
        return;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const float *x_row = x + ((b * seq + s) * dim);

            for (int64_t h = 0; h < hidden_dim; ++h) {
                const float *w1_row = w1 + (h * dim);
                float v = b1 ? b1[h] : 0.0f;
                for (int64_t d = 0; d < dim; ++d) {
                    v += x_row[d] * w1_row[d];
                }
                hidden[h] = v > 0.0f ? v : 0.0f; /* ReLU */
            }

            float z = b2 ? b2[0] : 0.0f;
            for (int64_t h = 0; h < hidden_dim; ++h) {
                z += hidden[h] * w2[h];
            }
            scores[b * seq + s] = _aria_sigmoidf(z);
        }
    }

    arena_free(hidden);
}

void aria_lane_router_threshold_f32(const float *scores,
                                    int64_t *assignments,
                                    float *weights,
                                    int64_t batch, int64_t seq, int64_t lanes,
                                    const float *thresholds) {
    if (!scores || !assignments || batch <= 0 || seq <= 0 || lanes <= 0) {
        return;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const float score = scores[b * seq + s];
            int64_t lane = 0;

            if (thresholds) {
                while (lane < lanes - 1 && score >= thresholds[lane]) {
                    lane++;
                }
            } else {
                /* Default equal-width bins over [0,1]. */
                float clamped = score;
                if (clamped < 0.0f) clamped = 0.0f;
                if (clamped > 1.0f) clamped = 1.0f;
                lane = (int64_t)(clamped * (float)lanes);
                if (lane >= lanes) lane = lanes - 1;
            }

            assignments[b * seq + s] = lane;

            if (weights) {
                float *w = weights + ((b * seq + s) * lanes);
                memset(w, 0, (size_t)lanes * sizeof(float));
                w[lane] = 1.0f;
            }
        }
    }
}

void aria_load_balance_loss_f32(const int64_t *assignments,
                                const float *target_distribution,
                                float *lane_fractions,
                                float *loss_out,
                                int64_t batch, int64_t seq, int64_t lanes,
                                float loss_weight) {
    if (!assignments || !loss_out || batch <= 0 || seq <= 0 || lanes <= 0) {
        return;
    }

    int64_t total = batch * seq;
    if (total <= 0) {
        *loss_out = 0.0f;
        return;
    }

    arena_reset();
    float *counts = (float *)arena_alloc((size_t)lanes * sizeof(float));
    if (!counts) {
        return;
    }
    memset(counts, 0, (size_t)lanes * sizeof(float));

    for (int64_t i = 0; i < total; ++i) {
        int64_t lane = assignments[i];
        if (lane < 0) lane = 0;
        if (lane >= lanes) lane = lanes - 1;
        counts[lane] += 1.0f;
    }

    float loss = 0.0f;
    for (int64_t lane = 0; lane < lanes; ++lane) {
        float frac = counts[lane] / (float)total;
        if (lane_fractions) {
            lane_fractions[lane] = frac;
        }
        float target = target_distribution ? target_distribution[lane] : (1.0f / (float)lanes);
        float delta = frac - target;
        loss += delta * delta;
    }

    *loss_out = loss_weight * loss;
    arena_free(counts);
}
