"""Python fallback kernel for sum_last (sum along last dimension)."""

from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: x.sum(dim=-1, keepdim=True))
