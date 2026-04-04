#include "kernels.h"

#include <cmath>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void aria_depth_weighted_proj_f32(const float *x,
                                  const float *depth_scorer,
                                  const float *step_projs,
                                  float *y,
                                  int64_t batch,
                                  int64_t seq,
                                  int64_t dim,
                                  int64_t max_depth) {
    if (!x || !depth_scorer || !step_projs || !y || batch <= 0 || seq <= 0 || dim <= 0 ||
        max_depth <= 0) {
        return;
    }

    std::vector<float> logits(static_cast<size_t>(max_depth));
    std::vector<float> probs(static_cast<size_t>(max_depth));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const float *x_row = x + ((b * seq + s) * dim);
            float *y_row = y + ((b * seq + s) * dim);

            float max_logit = -INFINITY;
            for (int64_t k = 0; k < max_depth; ++k) {
                const float *score_row = depth_scorer + k * dim;
                float logit = 0.0f;
                for (int64_t d = 0; d < dim; ++d) {
                    logit += x_row[d] * score_row[d];
                }
                logits[static_cast<size_t>(k)] = logit;
                if (logit > max_logit) {
                    max_logit = logit;
                }
            }

            float sum_exp = 0.0f;
            for (int64_t k = 0; k < max_depth; ++k) {
                float p = expf(logits[static_cast<size_t>(k)] - max_logit);
                probs[static_cast<size_t>(k)] = p;
                sum_exp += p;
            }
            float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
            for (int64_t k = 0; k < max_depth; ++k) {
                probs[static_cast<size_t>(k)] *= inv_sum;
            }

            for (int64_t out_d = 0; out_d < dim; ++out_d) {
                float acc = 0.0f;
                for (int64_t k = 0; k < max_depth; ++k) {
                    const float *proj_row =
                        step_projs + ((k * dim + out_d) * dim);
                    float proj = 0.0f;
                    for (int64_t in_d = 0; in_d < dim; ++in_d) {
                        proj += x_row[in_d] * proj_row[in_d];
                    }
                    acc += probs[static_cast<size_t>(k)] * proj;
                }
                y_row[out_d] = acc;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
