"""Kernel handler for hyp_tangent_nonlinear — delegates to research.mathspaces.hyperbolic."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "hyp_tangent_nonlinear",
    "research.mathspaces.hyperbolic.execute_hyp_tangent_nonlinear",
)
