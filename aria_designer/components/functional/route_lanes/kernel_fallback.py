"""Fallback kernel for route_lanes."""

from aria_designer.runtime.fallback_templates import make_route_argmax_handler

ComponentHandler = make_route_argmax_handler(
    "route_lanes",
    "lane_indices",
    "n_lanes",
    2,
)
