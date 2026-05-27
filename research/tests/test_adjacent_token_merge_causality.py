"""Causality regression test for the adjacent_token_merge primitive.

2026-05-23: `_op_adjacent_token_merge` was anti-causal — it merged token p
INTO p-1, so output[p-1] depended on x[p] (a one-step next-token leak). Because
the binding_range/screening/curriculum probes are causal next-token tasks
(scored at position i against input[i+1]), the leak handed them the label and
inflated the binding scores of every model containing the op.

These tests pin the invariant that fixed the leak:
    output[i] depends ONLY on inputs at positions <= i.

If anyone re-introduces a backward merge (or any anti-causal off-by-one), the
``test_op_*`` cases below fail loudly.
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis.compiler_ops_routing import _op_adjacent_token_merge
from research.tools.scaling_blimp_study import AdjacentTokenMergeLane


class _Mod:
    """Minimal stand-in for the compiled-op module (telemetry sink)."""


def _max_earlier_delta(fn, S: int, D: int = 8, seed: int = 0) -> float:
    """Max change at output positions STRICTLY BEFORE a perturbed input pos.

    Sweeps every position t in [1, S): blow up input[t], measure how much any
    output position < t moves. For a causal map this is exactly 0 everywhere.
    """
    torch.manual_seed(seed)
    x = torch.randn(1, S, D)
    base = fn(x)
    worst = 0.0
    for t in range(1, S):
        xp = x.clone()
        xp[:, t, :] = 99.0  # large, unambiguous perturbation
        out = fn(xp)
        worst = max(worst, (out[:, :t] - base[:, :t]).abs().max().item())
    return worst


def test_op_causal_no_future_leak() -> None:
    """The core invariant: perturbing input[t] never changes output[<t]."""
    for S in (8, 16, 31):  # include an odd length to catch boundary off-by-ones
        for n_keep in (S // 4, S // 2, S - 1):
            n_keep = max(1, n_keep)
            delta = _max_earlier_delta(
                lambda x: _op_adjacent_token_merge(_Mod(), [x], {"n_keep": n_keep}),
                S,
            )
            assert delta < 1e-6, (
                f"FUTURE LEAK: S={S} n_keep={n_keep} moved an earlier output by {delta:.4e}"
            )


def test_op_causal_under_grad() -> None:
    """The training path uses the torch branch (requires_grad / non-CPU-f32).
    Causality must hold there too (autograd-tracked tensors)."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        xg = x.clone().requires_grad_(True)
        with torch.no_grad():
            pass
        return _op_adjacent_token_merge(
            _Mod(), [xg], {"n_keep": x.shape[1] // 2}
        ).detach()

    assert _max_earlier_delta(fn, 16) < 1e-6


def test_lane_causal_no_future_leak() -> None:
    """AdjacentTokenMergeLane wraps the primitive; pointwise proj/gate cannot
    move information across positions, so lane causality == op causality."""
    lane = AdjacentTokenMergeLane(8).eval()
    with torch.no_grad():
        delta = _max_earlier_delta(lambda x: lane(x), 16, D=8)
    assert delta < 1e-6, f"lane leaks future: {delta:.4e}"


def test_op_identity_when_n_keep_equals_seq_len() -> None:
    """No merge requested -> output is the input untouched."""
    x = torch.randn(2, 12, 8)
    out = _op_adjacent_token_merge(_Mod(), [x], {"n_keep": 12})
    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_op_shape_restored() -> None:
    """Output is always restored to the original sequence length."""
    x = torch.randn(2, 12, 8)
    out = _op_adjacent_token_merge(_Mod(), [x], {"n_keep": 4})
    assert out.shape == (2, 12, 8)
    assert torch.isfinite(out).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
