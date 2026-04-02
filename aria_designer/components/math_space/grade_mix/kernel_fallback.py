"""Kernel handler for grade_mix — delegates to research.mathspaces.clifford."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "grade_mix",
    "research.mathspaces.clifford.execute_grade_mix",
)
