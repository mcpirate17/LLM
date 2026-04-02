"""Kernel handler for tropical_matmul — delegates to research.mathspaces.tropical."""

from runtime.fallback_templates import make_mathspace_binary_handler

ComponentHandler = make_mathspace_binary_handler(
    "tropical_matmul",
    "research.mathspaces.tropical.execute_tropical_matmul",
)
