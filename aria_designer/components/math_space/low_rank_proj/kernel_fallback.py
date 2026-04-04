"""Kernel handler for low_rank_proj — delegates to research.mathspaces.compression."""

from aria_designer.runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "low_rank_proj",
    "research.mathspaces.compression.execute_low_rank_proj",
)
