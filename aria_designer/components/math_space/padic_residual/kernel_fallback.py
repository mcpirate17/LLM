"""Kernel handler for padic_residual — delegates to research.mathspaces.padic."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "padic_residual",
    "research.mathspaces.padic.execute_padic_residual",
)
