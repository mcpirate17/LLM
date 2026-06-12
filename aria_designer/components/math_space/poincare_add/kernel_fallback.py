"""Kernel handler for poincare_add — delegates to research.mathspaces.hyperbolic."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler


def _native_args(inputs, config):
    x = inputs.get("x", inputs.get("a")).detach().contiguous().float()
    y = inputs.get("y", inputs.get("b", x)).detach().contiguous().float()
    c = config.get("curvature", 1.0)
    return (x, y, c)


ComponentHandler = make_mathspace_handler(
    "poincare_add",
    "research.mathspaces.hyperbolic.execute_poincare_add",
    native_args_fn=_native_args,
)
