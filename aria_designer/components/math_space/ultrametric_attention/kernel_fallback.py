"""Kernel handler for ultrametric_attention — delegates to research.mathspaces.padic."""

from runtime.fallback_templates import make_mathspace_unary_handler


def _native_args(inputs, config):
    x = inputs["x"].detach().contiguous().float()
    p = config.get("p", 2)
    return (x, float(p))


ComponentHandler = make_mathspace_unary_handler(
    "ultrametric_attention",
    "research.mathspaces.padic.execute_ultrametric_attn",
    native_args_fn=_native_args,
)
