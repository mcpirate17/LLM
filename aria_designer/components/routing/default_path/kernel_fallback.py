"""Python fallback kernel for default_path."""

from aria_designer.runtime.fallback_templates import make_identity_handler


ComponentHandler = make_identity_handler("default_path")
