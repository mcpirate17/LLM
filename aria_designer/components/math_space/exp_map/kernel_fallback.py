"""Kernel handler for exp_map — delegates to research.mathspaces.hyperbolic."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "exp_map",
    "research.mathspaces.hyperbolic.execute_exp_map",
)
