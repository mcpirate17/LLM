"""Fallback kernel shim for representation/embedding_lookup."""
from runtime.fallback_templates import make_identity_handler

ComponentHandler = make_identity_handler("representation/embedding_lookup")
