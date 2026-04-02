"""Kernel handler for tropical_attention — delegates to research.mathspaces.tropical."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "tropical_attention",
    "research.mathspaces.tropical.execute_tropical_attention",
)
