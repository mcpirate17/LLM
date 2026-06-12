"""Kernel handler for tied_proj — delegates to research.mathspaces.compression."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "tied_proj",
    "research.mathspaces.compression.execute_tied_proj",
)
