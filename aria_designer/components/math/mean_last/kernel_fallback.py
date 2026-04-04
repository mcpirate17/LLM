"""Python fallback kernel for mean_last (mean along last dimension)."""

from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: x.mean(dim=-1, keepdim=True))
