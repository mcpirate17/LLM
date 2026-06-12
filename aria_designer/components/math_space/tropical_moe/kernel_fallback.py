"""Kernel handler for tropical_moe — delegates to research.mathspaces.tropical_routing."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "tropical_moe",
    "research.mathspaces.tropical_routing.execute_tropical_moe",
)
