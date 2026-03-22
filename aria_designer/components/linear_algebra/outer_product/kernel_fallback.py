"""Python fallback kernel for outer_product."""


class ComponentHandler:
    """Fallback handler for outer_product: element-wise a * b."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        return {"y": a * b}
