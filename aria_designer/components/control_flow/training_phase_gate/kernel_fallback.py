"""Fallback kernel shim for control_flow/training_phase_gate."""
from runtime.fallback_templates import make_identity_handler

ComponentHandler = make_identity_handler("control_flow/training_phase_gate")
