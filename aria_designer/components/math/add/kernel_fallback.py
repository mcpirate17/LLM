"""Python fallback kernel for add."""

from aria_designer.components.base import make_binary_handler

ComponentHandler = make_binary_handler(lambda a, b: a + b, native_op_name="add")
