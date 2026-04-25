#include "search_metrics.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float distance;
    int32_t index;
} aria_distance_pair_t;

static float aria_mean_square_distance(const float* a, const float* b, int32_t dim) {
    double acc = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        const double diff = (double)a[i] - (double)b[i];
        acc += diff * diff;
    }
    return (float)(acc / (double)dim);
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

static int32_t aria_max_index(const float* values, int32_t count) {
    int32_t max_idx = 0;
    float max_value = values[0];
    for (int32_t i = 1; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
            max_idx = i;
        }
    }
    return max_idx;
}

static void aria_swap_float(float* a, float* b) {
    const float tmp = *a;
    *a = *b;
    *b = tmp;
}

static float aria_select_kth(float* values, size_t left, size_t right, size_t k) {
    while (left < right) {
        const float pivot = values[left + ((right - left) / 2u)];
        size_t i = left;
        size_t j = right;
        while (i <= j) {
            while (values[i] < pivot) {
                ++i;
            }
            while (values[j] > pivot) {
                if (j == 0u) {
                    break;
                }
                --j;
            }
            if (i <= j) {
                aria_swap_float(&values[i], &values[j]);
                ++i;
                if (j == 0u) {
                    break;
                }
                --j;
            }
        }
        if (k <= j) {
            right = j;
        } else if (k >= i) {
            left = i;
        } else {
            return values[k];
        }
    }
    return values[left];
}

static float aria_median_rms_from_mse_in_place(float* values, size_t count) {
    if ((count & 1u) != 0u) {
        const float mse = aria_select_kth(values, 0u, count - 1u, count / 2u);
        return (float)sqrt((double)mse);
    }
    const size_t right_idx = count / 2u;
    const size_t left_idx = right_idx - 1u;
    const float right_mse = aria_select_kth(values, 0u, count - 1u, right_idx);
    const float left_mse = aria_select_kth(values, 0u, right_idx - 1u, left_idx);
    return (float)((sqrt((double)left_mse) + sqrt((double)right_mse)) * 0.5);
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

    const int32_t used_k = k < n_rows ? k : n_rows;
    if (used_k == n_rows) {
        double total = 0.0;
        for (int32_t row = 0; row < n_rows; ++row) {
            total += sqrt((double)aria_mean_square_distance(
                feature_matrix + ((size_t)row * (size_t)dim),
                target,
                dim
            ));
        }
        *out_mean = (float)(total / (double)n_rows);
        return 0;
    }

    float* nearest = (float*)malloc((size_t)used_k * sizeof(float));
    if (nearest == NULL) {
        return -2;
    }

    for (int32_t row = 0; row < used_k; ++row) {
        nearest[row] = aria_mean_square_distance(
            feature_matrix + ((size_t)row * (size_t)dim),
            target,
            dim
        );
    }
    int32_t max_idx = aria_max_index(nearest, used_k);
    float max_distance = nearest[max_idx];

    for (int32_t row = used_k; row < n_rows; ++row) {
        const float distance = aria_mean_square_distance(
            feature_matrix + ((size_t)row * (size_t)dim),
            target,
            dim
        );
        if (distance < max_distance) {
            nearest[max_idx] = distance;
            max_idx = aria_max_index(nearest, used_k);
            max_distance = nearest[max_idx];
        }
    }

    double total = 0.0;
    for (int32_t i = 0; i < used_k; ++i) {
        total += sqrt((double)nearest[i]);
    }
    *out_mean = (float)(total / (double)used_k);
    free(nearest);
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
        pairs[row].distance = aria_mean_square_distance(
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
        out_distances[i] = (float)sqrt((double)pairs[i].distance);
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
            distances[offset++] = aria_mean_square_distance(left, right, dim);
        }
    }

    qsort(distances, pair_count, sizeof(float), aria_compare_float);
    if ((pair_count & 1u) != 0u) {
        *out_median = (float)sqrt((double)distances[pair_count / 2u]);
    } else {
        const size_t right = pair_count / 2u;
        const size_t left = right - 1u;
        *out_median =
            (float)((sqrt((double)distances[left]) + sqrt((double)distances[right])) * 0.5);
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

    const float radius_sq = radius * radius;
    for (int32_t i = 0; i < n_rows; ++i) {
        const float* left = feature_matrix + ((size_t)i * (size_t)dim);
        for (int32_t j = i + 1; j < n_rows; ++j) {
            const float* right = feature_matrix + ((size_t)j * (size_t)dim);
            if (aria_mean_square_distance(left, right, dim) < radius_sq) {
                out_counts[i] += 1;
                out_counts[j] += 1;
            }
        }
    }

    return 0;
}

int aria_behavior_pairwise_median_neighbor_counts(
    const float* feature_matrix,
    int32_t n_rows,
    int32_t dim,
    float* out_median,
    int32_t* out_counts
) {
    if (
        feature_matrix == NULL || out_median == NULL || out_counts == NULL ||
        n_rows < 2 || dim <= 0
    ) {
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
            distances[offset++] = aria_mean_square_distance(left, right, dim);
        }
        out_counts[i] = 0;
    }

    float* scratch = (float*)malloc(pair_count * sizeof(float));
    if (scratch == NULL) {
        free(distances);
        return -2;
    }
    memcpy(scratch, distances, pair_count * sizeof(float));
    const float median = aria_median_rms_from_mse_in_place(scratch, pair_count);
    free(scratch);
    *out_median = median;
    const float median_sq = median * median;

    offset = 0;
    for (int32_t i = 0; i < n_rows; ++i) {
        for (int32_t j = i + 1; j < n_rows; ++j) {
            if (distances[offset] < median_sq) {
                out_counts[i] += 1;
                out_counts[j] += 1;
            }
            ++offset;
        }
    }

    free(distances);
    return 0;
}
