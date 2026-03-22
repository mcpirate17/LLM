"""Python fallback kernel for multi_head_mix."""


class ComponentHandler:
    """Fallback handler for multi_head_mix: reshape to heads, mix, reshape back."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        n_heads = config.get("n_heads", 4)
        B, S, D = x.shape
        if D % n_heads != 0:
            return {"y": x}
        hd = D // n_heads
        # Reshape to (B, n_heads, S, hd), transpose heads for mixing, reshape back
        heads = x.view(B, S, n_heads, hd).permute(0, 2, 1, 3)  # (B, nh, S, hd)
        # Simple mean-pool across heads as a mixing baseline
        mixed = heads.mean(dim=1, keepdim=True).expand_as(heads)
        y = mixed.permute(0, 2, 1, 3).reshape(B, S, D)
        return {"y": y}
