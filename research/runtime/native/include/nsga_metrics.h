#ifndef ARIA_NSGA_METRICS_H
#define ARIA_NSGA_METRICS_H

#include <stdint.h>

int aria_nsga_crowding_distance(
    const float* objective_matrix,
    int32_t n_rows,
    int32_t n_objectives,
    float* out_distances
);

int aria_nsga_pareto_ranks(
    const float* objective_matrix,
    int32_t n_rows,
    int32_t n_objectives,
    int32_t* out_ranks
);

int aria_nsga_pareto_frontier_mask(
    const float* objective_matrix,
    int32_t n_rows,
    int32_t n_objectives,
    uint8_t* out_mask
);

#endif
