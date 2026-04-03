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
