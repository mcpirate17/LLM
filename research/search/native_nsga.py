from research.frontier_kernel import (
    load_native_frontier_lib,
    native_crowding_distances,
    native_pareto_frontier_mask,
    native_pareto_ranks,
    reset_native_frontier_lib,
)


def load_native_nsga_lib():
    return load_native_frontier_lib()


def reset_native_nsga_lib() -> None:
    reset_native_frontier_lib()


def compute_crowding_distances(objective_matrix):
    return native_crowding_distances(objective_matrix)


def compute_pareto_ranks(objective_matrix):
    return native_pareto_ranks(objective_matrix)


def compute_pareto_frontier_mask(objective_matrix):
    return native_pareto_frontier_mask(objective_matrix)
