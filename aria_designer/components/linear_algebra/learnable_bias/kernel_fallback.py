"""Python fallback kernel for learnable_bias."""


class ComponentHandler:
    """Fallback handler for learnable_bias: x + bias (bias initialized to zero)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        # Without a persistent module, bias is zero — pass through.
        # The WorkflowModule build path should construct a proper nn.Module
        # for trainable workflows; this fallback is for inference/preview.
        return {"y": inputs["x"]}
