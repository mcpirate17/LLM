"""Kernel handler for tropical_softmax — delegates to research.mathspaces.tropical."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "tropical_softmax",
    "research.mathspaces.tropical.execute_tropical_softmax",
)
