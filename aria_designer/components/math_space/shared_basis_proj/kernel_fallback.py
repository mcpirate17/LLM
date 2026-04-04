"""Kernel handler for shared_basis_proj — delegates to research.mathspaces.compression."""

from aria_designer.runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "shared_basis_proj",
    "research.mathspaces.compression.execute_shared_basis_proj",
)
