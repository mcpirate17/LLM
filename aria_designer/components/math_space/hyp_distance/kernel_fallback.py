"""Kernel handler for hyp_distance — delegates to research.mathspaces.hyperbolic."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "hyp_distance",
    "research.mathspaces.hyperbolic.execute_hyp_distance",
    arity=2,
)
