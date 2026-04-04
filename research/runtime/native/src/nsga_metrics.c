#include "nsga_metrics.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    float value;
    int32_t index;
} aria_nsga_value_index_t;

static int aria_compare_value_index(const void* lhs, const void* rhs) {
    const aria_nsga_value_index_t* a = (const aria_nsga_value_index_t*)lhs;
    const aria_nsga_value_index_t* b = (const aria_nsga_value_index_t*)rhs;
    if (a->value < b->value) {
        return -1;
    }
    if (a->value > b->value) {
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

int aria_nsga_crowding_distance(
    const float* objective_matrix,
    int32_t n_rows,
    int32_t n_objectives,
    float* out_distances
) {
    if (
        objective_matrix == NULL || out_distances == NULL || n_rows <= 0 ||
        n_objectives <= 0
    ) {
        return -1;
    }

    if (n_rows <= 2) {
        for (int32_t i = 0; i < n_rows; ++i) {
            out_distances[i] = INFINITY;
        }
        return 0;
    }

    for (int32_t i = 0; i < n_rows; ++i) {
        out_distances[i] = 0.0f;
    }

    aria_nsga_value_index_t* ordered =
        (aria_nsga_value_index_t*)malloc((size_t)n_rows * sizeof(aria_nsga_value_index_t));
    if (ordered == NULL) {
        return -2;
    }

    for (int32_t objective = 0; objective < n_objectives; ++objective) {
        for (int32_t row = 0; row < n_rows; ++row) {
            ordered[row].value =
                objective_matrix[(size_t)row * (size_t)n_objectives + (size_t)objective];
            ordered[row].index = row;
        }

        qsort(ordered, (size_t)n_rows, sizeof(aria_nsga_value_index_t), aria_compare_value_index);

        const float obj_min = ordered[0].value;
        const float obj_max = ordered[n_rows - 1].value;
        const float span = obj_max - obj_min;

        out_distances[ordered[0].index] = INFINITY;
        out_distances[ordered[n_rows - 1].index] = INFINITY;

        if (!(span > 0.0f)) {
            continue;
        }

        const float inv_span = 1.0f / span;
        for (int32_t i = 1; i < n_rows - 1; ++i) {
            const int32_t out_index = ordered[i].index;
            if (isinf(out_distances[out_index])) {
                continue;
            }
            out_distances[out_index] +=
                (ordered[i + 1].value - ordered[i - 1].value) * inv_span;
        }
    }

    free(ordered);
    return 0;
}

int aria_nsga_pareto_ranks(
    const float* objective_matrix,
    int32_t n_rows,
    int32_t n_objectives,
    int32_t* out_ranks
) {
    if (
        objective_matrix == NULL || out_ranks == NULL || n_rows <= 0 ||
        n_objectives <= 0
    ) {
        return -1;
    }

    unsigned char* dominates =
        (unsigned char*)calloc((size_t)n_rows * (size_t)n_rows, sizeof(unsigned char));
    int32_t* domination_count =
        (int32_t*)calloc((size_t)n_rows, sizeof(int32_t));
    unsigned char* remaining =
        (unsigned char*)calloc((size_t)n_rows, sizeof(unsigned char));
    int32_t* front_indices =
        (int32_t*)malloc((size_t)n_rows * sizeof(int32_t));
    if (
        dominates == NULL || domination_count == NULL || remaining == NULL ||
        front_indices == NULL
    ) {
        free(dominates);
        free(domination_count);
        free(remaining);
        free(front_indices);
        return -2;
    }

    for (int32_t i = 0; i < n_rows; ++i) {
        remaining[i] = 1u;
        out_ranks[i] = 0;
    }

    for (int32_t i = 0; i < n_rows; ++i) {
        const float* left =
            objective_matrix + ((size_t)i * (size_t)n_objectives);
        for (int32_t j = 0; j < n_rows; ++j) {
            if (i == j) {
                continue;
            }
            const float* right =
                objective_matrix + ((size_t)j * (size_t)n_objectives);
            int dominates_all = 1;
            int dominates_any = 0;
            for (int32_t k = 0; k < n_objectives; ++k) {
                if (left[k] < right[k]) {
                    dominates_all = 0;
                    break;
                }
                if (left[k] > right[k]) {
                    dominates_any = 1;
                }
            }
            if (dominates_all && dominates_any) {
                dominates[(size_t)i * (size_t)n_rows + (size_t)j] = 1u;
                domination_count[j] += 1;
            }
        }
    }

    int32_t remaining_count = n_rows;
    int32_t rank = 1;
    while (remaining_count > 0) {
        int32_t front_size = 0;
        for (int32_t i = 0; i < n_rows; ++i) {
            if (remaining[i] != 0u && domination_count[i] == 0) {
                front_indices[front_size++] = i;
            }
        }
        if (front_size == 0) {
            break;
        }

        for (int32_t idx = 0; idx < front_size; ++idx) {
            const int32_t i = front_indices[idx];
            remaining[i] = 0u;
            out_ranks[i] = rank;
            remaining_count -= 1;
        }

        for (int32_t idx = 0; idx < front_size; ++idx) {
            const int32_t i = front_indices[idx];
            const size_t row_offset = (size_t)i * (size_t)n_rows;
            for (int32_t j = 0; j < n_rows; ++j) {
                if (remaining[j] != 0u && dominates[row_offset + (size_t)j] != 0u) {
                    domination_count[j] -= 1;
                }
            }
        }

        rank += 1;
    }

    free(dominates);
    free(domination_count);
    free(remaining);
    free(front_indices);
    return 0;
}
