#include "kernels.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void aria_adaptive_route_dispatch_f32(const float *x,
                                                 const float *w1, const float *b1,
                                                 const float *w2, const float *b2,
                                                 int64_t lanes, const float *thresholds,
                                                 float *scores, int64_t *assignments, float *weights,
                                                 float *lane_out, int64_t *index_map, int64_t *lane_counts,
                                                 int64_t batch, int64_t seq, int64_t dim, int64_t hidden_dim) {
    if (!x || !w1 || !w2 || !scores || !assignments || !weights ||
        !lane_out || !index_map || !lane_counts ||
        batch <= 0 || seq <= 0 || dim <= 0 || lanes <= 0) {
        return;
    }

    // 1) Difficulty scoring.
    aria_difficulty_scorer_f32(x, w1, b1, w2, b2, scores, batch, seq, dim, hidden_dim);

    // 2) Threshold routing (assignments + one-hot weights).
    aria_lane_router_threshold_f32(scores, assignments, weights, batch, seq, lanes, thresholds);

    // 3) Per-lane dispatch into packed buffers.
    const int64_t lane_stride = batch * seq * dim;
    const int64_t map_stride = batch * seq;
    const int64_t counts_stride = batch;
    for (int64_t lane_id = 0; lane_id < lanes; ++lane_id) {
        float *lane_ptr = lane_out + lane_id * lane_stride;
        int64_t *map_ptr = index_map + lane_id * map_stride;
        int64_t *count_ptr = lane_counts + lane_id * counts_stride;
        aria_conditional_dispatch_f32(x, assignments, lane_id, lane_ptr, map_ptr, count_ptr, batch, seq, dim);
    }
}

#ifdef __cplusplus
}
#endif
