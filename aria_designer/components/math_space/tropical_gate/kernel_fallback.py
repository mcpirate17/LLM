"""Fallback kernel shim for math_space/tropical_gate."""
from runtime.fallback_templates import make_native_temperature_handler

ComponentHandler = make_native_temperature_handler(
    "math_space/tropical_gate",
    "tropical_gate",
)
