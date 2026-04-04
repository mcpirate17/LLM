"""Kernel handler for geometric_product — delegates to research.mathspaces.clifford."""

from aria_designer.runtime.fallback_templates import make_mathspace_binary_handler

ComponentHandler = make_mathspace_binary_handler(
    "clifford_geometric_product_cl30",
    "research.mathspaces.clifford.execute_geometric_product",
)
