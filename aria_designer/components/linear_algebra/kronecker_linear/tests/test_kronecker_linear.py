"""Contract tests for kronecker_linear component."""

import torch


def test_kronecker_linear_shape():
    from aria_designer.components.linear_algebra.kronecker_linear.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 8, 256)
    result = handler.forward({"x": x}, {})
    assert result["y"].shape == (2, 8, 256)


def test_kronecker_linear_gradient():
    from aria_designer.components.linear_algebra.kronecker_linear.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 4, 64, requires_grad=True)
    result = handler.forward({"x": x}, {})
    loss = result["y"].sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_kronecker_linear_non_square_dim():
    """Test with D that isn't a perfect square."""
    from aria_designer.components.linear_algebra.kronecker_linear.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 4, 128)  # 128 = 8 * 16
    result = handler.forward({"x": x}, {})
    assert result["y"].shape == (2, 4, 128)


def test_kronecker_linear_validate():
    from aria_designer.components.linear_algebra.kronecker_linear.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    assert handler.validate_config({}) == []
