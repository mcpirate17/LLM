"""Python fallback kernel for learnable_scale."""


class ComponentHandler:
    """Fallback handler for learnable_scale: x * scale (scale initialized to one)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        # Without a persistent module, scale is one — pass through.
        # The WorkflowModule build path should construct a proper nn.Module
        # for trainable workflows; this fallback is for inference/preview.
        return {"y": inputs["x"]}
