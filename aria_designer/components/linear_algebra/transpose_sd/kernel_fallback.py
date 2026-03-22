"""Python fallback kernel for transpose_sd."""


class ComponentHandler:
    """Fallback handler for transpose_sd: swap S and D dims."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        return {"y": x.transpose(-1, -2).contiguous()}
