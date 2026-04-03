"""Python fallback kernel for early_exit (confidence-gated output).

Identical gating logic to learned_token_gate — delegates to avoid duplication.
"""

from components.routing.learned_token_gate.kernel_fallback import ComponentHandler

__all__ = ["ComponentHandler"]
