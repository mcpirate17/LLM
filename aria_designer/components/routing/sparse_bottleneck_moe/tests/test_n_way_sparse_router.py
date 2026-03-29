"""Contract tests for n_way_sparse_router component."""

import torch


def test_n_way_sparse_router_shape():
    from aria_designer.components.routing.n_way_sparse_router.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 8, 64)
    result = handler.forward({"x": x}, {"n_ways": 4, "top_k": 2})
    assert result["y"].shape == (2, 8, 64)


def test_n_way_sparse_router_gradient():
    from aria_designer.components.routing.n_way_sparse_router.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 4, 32, requires_grad=True)
    result = handler.forward({"x": x}, {"n_ways": 4, "top_k": 2})
    loss = result["y"].sum()
    loss.backward()
    assert x.grad is not None


def test_n_way_sparse_router_default_config():
    from aria_designer.components.routing.n_way_sparse_router.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(1, 4, 64)
    result = handler.forward({"x": x}, {})
    assert result["y"].shape == (1, 4, 64)


def test_n_way_sparse_router_validate():
    from aria_designer.components.routing.n_way_sparse_router.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    assert handler.validate_config({"n_ways": 4, "top_k": 2}) == []
    assert len(handler.validate_config({"n_ways": 1})) > 0
    assert len(handler.validate_config({"n_ways": 4, "top_k": 5})) > 0
