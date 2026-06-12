"""Kernel handler for sparse_threshold — delegates to research.mathspaces.spiking."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "sparse_threshold",
    "research.mathspaces.spiking.execute_sparse_threshold",
)
