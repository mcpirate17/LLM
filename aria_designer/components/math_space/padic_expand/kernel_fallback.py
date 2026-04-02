"""Kernel handler for padic_expand — delegates to research.mathspaces.padic."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "padic_expand",
    "research.mathspaces.padic.execute_padic_expand",
)
