#include "search_metrics.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    float distance;
    int32_t index;
} aria_distance_pair_t;

static float aria_rms_distance(const float* a, const float* b, int32_t dim) {
    double acc = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        const double diff = (double)a[i] - (double)b[i];
        acc += diff * diff;
    }
    return (float)sqrt(acc / (double)dim);
}

static int aria_compare_float(const void* lhs, const void* rhs) {
    const float a = *(const float*)lhs;
    const float b = *(const float*)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static int aria_compare_pair(const void* lhs, const void* rhs) {
    const aria_distance_pair_t* a = (const aria_distance_pair_t*)lhs;
    const aria_distance_pair_t* b = (const aria_distance_pair_t*)rhs;
    if (a->distance < b->distance) {
        return -1;
    }
    if (a->distance > b->distance) {
        return 1;
    }
    if (a->index < b->index) {
        return -1;
    }
    if (a->index > b->index) {
        return 1;
    }
    return 0;
}

int aria_behavior_mean_k_nearest(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    const float* target,
    int32_t k,
    float* out_mean
) {
    if (
        feature_matrix == NULL || target == NULL || out_mean == NULL || n_rows <= 0 ||
        dim <= 0 || k <= 0
    ) {
        return -1;
    }

    float* distances = (float*)malloc((size_t)n_rows * sizeof(float));
    if (distances == NULL) {
        return -2;
    }

    for (int32_t row = 0; row < n_rows; ++row) {
        distances[row] = aria_rms_distance(
            feature_matrix + ((size_t)row * (size_t)dim),
            target,
            dim
        );
    }

    qsort(distances, (size_t)n_rows, sizeof(float), aria_compare_float);
    const int32_t used_k = k < n_rows ? k : n_rows;
    double total = 0.0;
    for (int32_t i = 0; i < used_k; ++i) {
        total += distances[i];
    }
    *out_mean = (float)(total / (double)used_k);
    free(distances);
    return 0;
}

int aria_behavior_topk_nearest_indices(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    const float* target,
    int32_t k,
    int32_t* out_indices,
    float* out_distances
) {
    if (
        feature_matrix == NULL || target == NULL || out_indices == NULL ||
        out_distances == NULL || n_rows <= 0 || dim <= 0 || k <= 0
    ) {
        return -1;
    }

    aria_distance_pair_t* pairs =
        (aria_distance_pair_t*)malloc((size_t)n_rows * sizeof(aria_distance_pair_t));
    if (pairs == NULL) {
        return -2;
    }

    for (int32_t row = 0; row < n_rows; ++row) {
        pairs[row].distance = aria_rms_distance(
            feature_matrix + ((size_t)row * (size_t)dim),
            target,
            dim
        );
        pairs[row].index = row;
    }

    qsort(pairs, (size_t)n_rows, sizeof(aria_distance_pair_t), aria_compare_pair);
    const int32_t used_k = k < n_rows ? k : n_rows;
    for (int32_t i = 0; i < used_k; ++i) {
        out_indices[i] = pairs[i].index;
        out_distances[i] = pairs[i].distance;
    }
    free(pairs);
    return used_k;
}

int aria_behavior_pairwise_median(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    float* out_median
) {
    if (feature_matrix == NULL || out_median == NULL || n_rows < 2 || dim <= 0) {
        return -1;
    }

    const size_t pair_count = ((size_t)n_rows * (size_t)(n_rows - 1)) / 2u;
    float* distances = (float*)malloc(pair_count * sizeof(float));
    if (distances == NULL) {
        return -2;
    }

    size_t offset = 0;
    for (int32_t i = 0; i < n_rows; ++i) {
        const float* left = feature_matrix + ((size_t)i * (size_t)dim);
        for (int32_t j = i + 1; j < n_rows; ++j) {
            const float* right = feature_matrix + ((size_t)j * (size_t)dim);
            distances[offset++] = aria_rms_distance(left, right, dim);
        }
    }

    qsort(distances, pair_count, sizeof(float), aria_compare_float);
    if ((pair_count & 1u) != 0u) {
        *out_median = distances[pair_count / 2u];
    } else {
        const size_t right = pair_count / 2u;
        const size_t left = right - 1u;
        *out_median = (distances[left] + distances[right]) * 0.5f;
    }
    free(distances);
    return 0;
}

int aria_behavior_neighbor_counts(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    float radius,
    int32_t* out_counts
) {
    if (
        feature_matrix == NULL || out_counts == NULL || n_rows <= 0 || dim <= 0 ||
        radius < 0.0f
    ) {
        return -1;
    }

    for (int32_t i = 0; i < n_rows; ++i) {
        out_counts[i] = 0;
    }

    for (int32_t i = 0; i < n_rows; ++i) {
        const float* left = feature_matrix + ((size_t)i * (size_t)dim);
        for (int32_t j = i + 1; j < n_rows; ++j) {
            const float* right = feature_matrix + ((size_t)j * (size_t)dim);
            if (aria_rms_distance(left, right, dim) < radius) {
                out_counts[i] += 1;
                out_counts[j] += 1;
            }
        }
    }

    return 0;
}
