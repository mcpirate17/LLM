"""Python fallback kernel for sparse_span_builder."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        span_width = max(1, min(int(config.get("span_width", 3)), x.shape[1]))
        y = torch.zeros_like(x)
        keep_mask = x.abs().sum(dim=-1) > 1e-8
        min_kept = 1 if span_width <= 1 else 2
        for b in range(x.shape[0]):
            packed = 0
            for start in range(0, x.shape[1] - span_width + 1):
                window = keep_mask[b, start : start + span_width]
                if int(window.sum().item()) < min_kept or packed >= x.shape[1]:
                    continue
                y[b, packed] = x[b, start : start + span_width].mean(dim=0)
                packed += 1
        return {"y": y}
