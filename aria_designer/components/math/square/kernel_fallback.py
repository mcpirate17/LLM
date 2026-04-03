"""Python fallback kernel for square."""

from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: x * x, native_op_name="square")
