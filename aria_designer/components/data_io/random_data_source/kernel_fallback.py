"""Python fallback for random_data_source."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        batch_size = int(config.get("batch_size", 2))
        seq_len = int(config.get("seq_len", 16))
        feature_dim = int(config.get("feature_dim", 128))
        distribution = str(config.get("distribution", "gaussian"))
        seed = int(config.get("seed", 42))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        shape = (batch_size, seq_len, feature_dim)

        if distribution == "uniform":
            low = float(config.get("low", -1.0))
            high = float(config.get("high", 1.0))
            data = torch.rand(shape, generator=gen) * (high - low) + low
        elif distribution == "bernoulli":
            p = float(config.get("high", 0.5))
            p = max(0.0, min(1.0, p))
            data = torch.bernoulli(torch.full(shape, p), generator=gen)
        else:
            mean = float(config.get("mean", 0.0))
            std = float(config.get("std", 1.0))
            data = torch.randn(shape, generator=gen) * std + mean

        return {"data": data}
