from __future__ import annotations

import pytest
import torch

from aria_designer.components.routing.hybrid_sparse_router.kernel_fallback import (
    ComponentHandler,
    NativeSparseHybridRouter,
)


def test_hybrid_sparse_router_handler_runs_with_native_backend_when_available():
    if NativeSparseHybridRouter is None:
        pytest.skip("native intelligent router bridge unavailable")
    handler = ComponentHandler()
    x = torch.randn(2, 8, 16)
    out = handler.forward(
        {"x": x},
        {"lane_count": 3, "confidence_threshold": 0.1},
    )
    y = out["y"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert handler._native_router is not None
