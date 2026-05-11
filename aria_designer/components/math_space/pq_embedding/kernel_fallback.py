"""Kernel handler for pq_embedding — delegates to research.mathspaces.pq_embedding."""

from aria_designer.runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "pq_embedding",
    "research.mathspaces.pq_embedding.execute_pq_embedding",
)
