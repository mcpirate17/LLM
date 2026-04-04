"""Kernel handler for grade_select — delegates to research.mathspaces.clifford."""

from aria_designer.runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "grade_select",
    "research.mathspaces.clifford.execute_grade_select",
)
