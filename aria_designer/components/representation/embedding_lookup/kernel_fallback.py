"""Fallback kernel shim for representation/embedding_lookup."""

from runtime.fallback_templates import make_embedding_lookup_handler

ComponentHandler = make_embedding_lookup_handler("representation/embedding_lookup")
