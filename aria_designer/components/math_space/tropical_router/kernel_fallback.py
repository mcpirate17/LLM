"""Kernel handler for tropical_router — delegates to research.mathspaces.tropical_routing."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "tropical_router",
    "research.mathspaces.tropical_routing.execute_tropical_router",
)
