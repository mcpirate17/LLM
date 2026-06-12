"""Kernel handler for bottleneck_proj — delegates to research.mathspaces.compression."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "bottleneck_proj",
    "research.mathspaces.compression.execute_bottleneck_proj",
)
