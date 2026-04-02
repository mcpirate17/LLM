"""Kernel handler for tropical_add — delegates to research.mathspaces.tropical."""

from runtime.fallback_templates import make_mathspace_binary_handler

ComponentHandler = make_mathspace_binary_handler(
    "tropical_add",
    "research.mathspaces.tropical.execute_tropical_add",
)
