"""Python fallback kernel for max_last (max along last dimension)."""

from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: x.max(dim=-1, keepdim=True).values)
