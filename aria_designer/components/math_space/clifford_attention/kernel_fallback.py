"""Kernel handler for clifford_attention — delegates to research.mathspaces.clifford."""

from aria_designer.runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "clifford_attention",
    "research.mathspaces.clifford.execute_clifford_attention",
)
