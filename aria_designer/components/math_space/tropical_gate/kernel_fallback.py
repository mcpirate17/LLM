"""Kernel handler for tropical_gate — delegates to research.mathspaces.tropical."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "tropical_gate",
    "research.mathspaces.tropical.execute_tropical_gate",
)
