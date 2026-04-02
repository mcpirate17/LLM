"""Kernel handler for tropical_gate — delegates to research.mathspaces.tropical."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "tropical_gate",
    "research.mathspaces.tropical.execute_tropical_gate",
)
