"""Contract tests for chebyshev_spectral_mix component."""

import torch


def test_chebyshev_spectral_mix_shape():
    from aria_designer.components.mixing.chebyshev_spectral_mix.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 8, 256)
    result = handler.forward({"x": x}, {"chebyshev_order": 6})
    assert result["y"].shape == (2, 8, 256)


def test_chebyshev_spectral_mix_gradient():
    from aria_designer.components.mixing.chebyshev_spectral_mix.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(2, 4, 64, requires_grad=True)
    result = handler.forward({"x": x}, {"chebyshev_order": 4})
    loss = result["y"].sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_chebyshev_spectral_mix_default_config():
    from aria_designer.components.mixing.chebyshev_spectral_mix.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(1, 4, 128)
    result = handler.forward({"x": x}, {})
    assert result["y"].shape == (1, 4, 128)


def test_chebyshev_spectral_mix_validate():
    from aria_designer.components.mixing.chebyshev_spectral_mix.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    assert handler.validate_config({"chebyshev_order": 6}) == []
    assert len(handler.validate_config({"chebyshev_order": 1})) > 0
    assert len(handler.validate_config({"chebyshev_order": 20})) > 0


def test_chebyshev_spectral_mix_min_order():
    """Order 2 should still work (T_0 + T_1 only)."""
    from aria_designer.components.mixing.chebyshev_spectral_mix.kernel_fallback import (
        ComponentHandler,
    )

    handler = ComponentHandler()
    x = torch.randn(1, 4, 32)
    result = handler.forward({"x": x}, {"chebyshev_order": 2})
    assert result["y"].shape == (1, 4, 32)
