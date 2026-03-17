"""Python fallback kernel for add."""

from components.base import make_binary_handler

ComponentHandler = make_binary_handler(lambda a, b: a + b)
