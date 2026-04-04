"""Native-first fallback kernel for route_recursion."""

from aria_designer.runtime.fallback_templates import make_route_argmax_handler

ComponentHandler = make_route_argmax_handler(
    "route_recursion",
    "depth",
    "max_depth",
    1,
)
