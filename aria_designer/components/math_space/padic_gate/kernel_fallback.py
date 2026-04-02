"""Kernel handler for padic_gate — delegates to research.mathspaces.padic."""

from runtime.fallback_templates import make_mathspace_unary_handler


def _native_args(inputs, config):
    x = inputs["x"].detach().contiguous().float()
    p = config.get("p", 2.0)
    return (x, p)


ComponentHandler = make_mathspace_unary_handler(
    "padic_gate",
    "research.mathspaces.padic.execute_padic_gate",
    native_args_fn=_native_args,
)
