"""Kernel handler for lif_neuron — delegates to research.mathspaces.spiking."""

from runtime.fallback_templates import make_mathspace_unary_handler


def _native_args(inputs, config):
    x = inputs["x"].detach().contiguous().float()
    tau = config.get("tau", 20.0)
    threshold = config.get("threshold", 1.0)
    return (x, tau, threshold)


ComponentHandler = make_mathspace_unary_handler(
    "lif_neuron",
    "research.mathspaces.spiking.execute_lif",
    native_args_fn=_native_args,
)
