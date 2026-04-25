#ifndef ARIA_SEARCH_METRICS_H
#define ARIA_SEARCH_METRICS_H

#include <stdint.h>

int aria_behavior_mean_k_nearest(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    const float* target,
    int32_t k,
    float* out_mean
);

int aria_behavior_topk_nearest_indices(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    const float* target,
    int32_t k,
    int32_t* out_indices,
    float* out_distances
);

int aria_behavior_pairwise_median(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    float* out_median
);

int aria_behavior_neighbor_counts(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    float radius,
    int32_t* out_counts
);

int aria_behavior_pairwise_median_neighbor_counts(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    float* out_median,
    int32_t* out_counts
);

#endif
