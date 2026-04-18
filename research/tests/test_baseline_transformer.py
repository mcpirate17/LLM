from __future__ import annotations

import pytest
import torch

# `_SimpleTransformerLayer` was renamed to `SimpleTransformerLayer` (no
# leading underscore) and moved to `research.eval.reference_training` during
# the baseline-vs-reference split.
from research.eval.reference_training import SimpleTransformerLayer

pytestmark = pytest.mark.unit


def test_simple_transformer_layer_preserves_shape():
    layer = SimpleTransformerLayer(d_model=32, n_heads=4)
    x = torch.randn(2, 16, 32)

    out1 = layer(x)
    out2 = layer(x)

    assert out1.shape == x.shape
    assert out2.shape == x.shape
