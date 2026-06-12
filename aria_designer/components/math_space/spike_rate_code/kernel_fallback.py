"""Kernel handler for spike_rate_code — delegates to research.mathspaces.spiking."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "spike_rate_code",
    "research.mathspaces.spiking.execute_spike_rate_code",
)
