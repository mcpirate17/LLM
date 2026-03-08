"""Fallback kernel shim for structural/conditional_dispatch."""
from runtime.fallback_templates import make_identity_handler

ComponentHandler = make_identity_handler("structural/conditional_dispatch")
