"""Fallback kernel shim for functional/load_balance_loss."""
from runtime.fallback_templates import make_identity_handler

ComponentHandler = make_identity_handler("functional/load_balance_loss")
