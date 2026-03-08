"""Fallback kernel shim for routing/lane_router."""
from runtime.fallback_templates import make_identity_handler

ComponentHandler = make_identity_handler("routing/lane_router")
