"""Kernel handler for padic_expand — delegates to research.mathspaces.padic."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "padic_expand",
    "research.mathspaces.padic.execute_padic_expand",
)
