"""Python fallback kernel for sub."""

from components.base import make_binary_handler

ComponentHandler = make_binary_handler(lambda a, b: a - b)
