"""Kernel handler for tropical_matmul — delegates to research.mathspaces.tropical."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "tropical_matmul",
    "research.mathspaces.tropical.execute_tropical_matmul",
    arity=2,
)
