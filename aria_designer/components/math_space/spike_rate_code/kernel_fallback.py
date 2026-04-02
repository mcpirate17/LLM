"""Kernel handler for spike_rate_code — delegates to research.mathspaces.spiking."""

from runtime.fallback_templates import make_mathspace_unary_handler

ComponentHandler = make_mathspace_unary_handler(
    "spike_rate_code",
    "research.mathspaces.spiking.execute_spike_rate_code",
)
