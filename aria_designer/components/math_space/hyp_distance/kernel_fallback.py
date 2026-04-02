"""Kernel handler for hyp_distance — delegates to research.mathspaces.hyperbolic."""

from runtime.fallback_templates import make_mathspace_binary_handler

ComponentHandler = make_mathspace_binary_handler(
    "hyp_distance",
    "research.mathspaces.hyperbolic.execute_hyp_distance",
)
