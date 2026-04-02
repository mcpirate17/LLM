"""Kernel handler for bottleneck_proj — delegates to research.mathspaces.compression."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "bottleneck_proj",
    "research.mathspaces.compression.execute_bottleneck_proj",
)
