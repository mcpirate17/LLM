"""Kernel handler for padic_residual — delegates to research.mathspaces.padic."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "padic_residual",
    "research.mathspaces.padic.execute_padic_residual",
)
