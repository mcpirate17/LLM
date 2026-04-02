"""Kernel handler for tied_proj — delegates to research.mathspaces.compression."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "tied_proj",
    "research.mathspaces.compression.execute_tied_proj",
)
