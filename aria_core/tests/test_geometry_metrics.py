import math

import pytest
import torch

import aria_core


def _geometry_metrics_python(reps: torch.Tensor, max_rows: int = 500) -> torch.Tensor:
    flat = reps.reshape(-1, reps.shape[-1]).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    subset = flat[torch.randperm(flat.shape[0])[: min(flat.shape[0], max_rows)]]
    singular_values = torch.linalg.svdvals(subset).clamp(min=1e-10)
    normalized = singular_values / singular_values.sum()
    entropy = (-(normalized * torch.log(normalized))).sum()
    return torch.tensor(
        [
            (1.0 / (normalized.square().sum())).item(),
            (singular_values.min() / singular_values.max()).item(),
            (math.exp(float(entropy.item())) / len(singular_values)),
        ],
        dtype=torch.float32,
    )


def test_geometry_metrics_f32_matches_python_reference():
    torch.manual_seed(0)
    reps = torch.randn(3, 12, 8, dtype=torch.float32)

    torch.manual_seed(123)
    native = aria_core.geometry_metrics_f32(reps.contiguous(), 10)
    torch.manual_seed(123)
    expected = _geometry_metrics_python(reps, max_rows=10)

    assert native.shape == (3,)
    assert torch.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_geometry_metrics_f32_is_deterministic_for_fixed_seed():
    torch.manual_seed(7)
    reps = torch.randn(4, 16, 8, dtype=torch.float32)

    torch.manual_seed(99)
    a = aria_core.geometry_metrics_f32(reps.contiguous(), 10)
    torch.manual_seed(99)
    b = aria_core.geometry_metrics_f32(reps.contiguous(), 10)

    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("shape", [(16, 8), (2, 9, 8)])
def test_geometry_metrics_f32_returns_finite_values(shape):
    torch.manual_seed(1)
    reps = torch.randn(*shape, dtype=torch.float32)

    result = aria_core.geometry_metrics_f32(reps.contiguous(), 500)

    assert result.shape == (3,)
    assert torch.isfinite(result).all()
    assert result[0].item() >= 1.0
    assert 0.0 <= result[1].item() <= 1.0
    assert 0.0 <= result[2].item() <= 1.0
