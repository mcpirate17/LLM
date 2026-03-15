"""Python fallback kernel for max_last (identity stub)."""
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: x)
