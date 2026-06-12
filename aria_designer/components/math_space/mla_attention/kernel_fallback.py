"""Kernel handler for mla_attention — delegates to research.mathspaces.mla."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "mla_attention",
    "research.mathspaces.mla.execute_mla_attention",
    arity=2,
)
