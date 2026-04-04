"""Kernel handler for stdp_attention — delegates to research.mathspaces.spiking."""

from aria_designer.runtime.fallback_templates import make_mathspace_unary_handler


def _native_args(inputs, config):
    x = inputs["x"].detach().contiguous().float()
    tau_plus = config.get("tau_plus", 20.0)
    tau_minus = config.get("tau_minus", 20.0)
    return (x, tau_plus, tau_minus)


ComponentHandler = make_mathspace_unary_handler(
    "stdp_attention",
    "research.mathspaces.spiking.execute_stdp_attention",
    native_args_fn=_native_args,
)
