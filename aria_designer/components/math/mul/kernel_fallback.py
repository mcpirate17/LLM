"""Python fallback kernel for mul."""

from aria_designer.components.base import make_binary_handler

ComponentHandler = make_binary_handler(lambda a, b: a * b, native_op_name="mul")
