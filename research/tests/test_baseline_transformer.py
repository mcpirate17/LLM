from __future__ import annotations

import torch

from research.eval.baseline import _SimpleTransformerLayer


def test_simple_transformer_layer_reuses_cached_causal_mask():
    layer = _SimpleTransformerLayer(d_model=32, n_heads=4)
    x = torch.randn(2, 16, 32)

    out1 = layer(x)
    out2 = layer(x)

    assert out1.shape == x.shape
    assert out2.shape == x.shape
    assert len(layer._causal_mask_cache) == 1
