"""Tests for the measured algebraic-property / softmax-twin detector (NM-11).

The detector must flag the convex-token-averaging (softmax) family and leave
genuinely novel geometry — pointwise gates/MLPs, identity, and *signed* token
mixers — on the novel side. It also reports idempotence and additivity as
orthogonal novelty features.
"""

from __future__ import annotations

import pytest
import torch

from component_fab.proposer.algebraic_properties import (
    ALGEBRAIC_PROPERTY_NAMES,
    AlgebraicProperties,
    measure_algebraic_properties,
)

_DIM = 24
_SEQ = 16


def _softmax_qk_mixer(x: torch.Tensor) -> torch.Tensor:
    """Content-dependent softmax averaging with separate Q/K: the canonical twin."""
    d = x.shape[-1]
    gen = torch.Generator().manual_seed(7)
    wq = torch.randn(d, d, generator=gen) * d**-0.5
    wk = torch.randn(d, d, generator=gen) * d**-0.5
    scores = (x @ wq) @ (x @ wk).transpose(1, 2) / (d**0.5)
    return torch.softmax(scores, dim=-1) @ x  # rows sum to 1, non-negative


def _mean_pool(x: torch.Tensor) -> torch.Tensor:
    """Uniform token average broadcast to every position: the purest twin."""
    return x.mean(dim=1, keepdim=True).expand_as(x)


def _token_difference(x: torch.Tensor) -> torch.Tensor:
    """Signed high-pass token mixer: mixes tokens but breaks partition-of-unity."""
    return x - torch.roll(x, 1, dims=1)


def _pointwise_gate(x: torch.Tensor) -> torch.Tensor:
    """Channel-wise nonlinear gate: no token mixing."""
    return torch.nn.functional.gelu(x)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _linear_channel_map(x: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(x.shape[-1], x.shape[-1], generator=gen) * (x.shape[-1] ** -0.5)
    return x @ w


def _measure(f) -> AlgebraicProperties:
    return measure_algebraic_properties(f, dim=_DIM, seq_len=_SEQ, n_seeds=3)


def test_softmax_mixer_flagged_as_twin() -> None:
    props = _measure(_softmax_qk_mixer)
    assert props.is_softmax_twin(), props.to_dict()
    # All three convex-averaging tells fire for a real softmax average.
    assert props.constant_token_preservation > 0.9
    assert props.convex_range_fraction > 0.95
    assert props.cross_token_mixing > 0.5


def test_mean_pool_flagged_as_twin() -> None:
    props = _measure(_mean_pool)
    assert props.is_softmax_twin(), props.to_dict()
    assert props.softmax_twin_score > 0.9


def test_pointwise_gate_not_twin() -> None:
    """A pointwise nonlinearity does not mix tokens -> never a twin."""
    props = _measure(_pointwise_gate)
    assert not props.is_softmax_twin(), props.to_dict()
    assert props.cross_token_mixing < 0.05


def test_identity_not_twin() -> None:
    props = _measure(_identity)
    assert not props.is_softmax_twin(), props.to_dict()
    assert props.cross_token_mixing < 0.05
    # Identity is idempotent and additive.
    assert props.idempotence > 0.95
    assert props.additivity > 0.95


def test_signed_token_mixer_not_twin() -> None:
    """A signed mixer blends tokens but breaks partition-of-unity: novel, not twin."""
    props = _measure(_token_difference)
    assert not props.is_softmax_twin(), props.to_dict()
    assert props.cross_token_mixing > 0.5  # it DOES mix tokens...
    assert props.constant_token_preservation < 0.7  # ...but not convexly.


def test_idempotence_detects_projection() -> None:
    def mean_subtract(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)

    props = _measure(mean_subtract)
    assert props.idempotence > 0.95, props.to_dict()


def test_additivity_separates_linear_from_nonlinear() -> None:
    linear = _measure(_linear_channel_map)
    nonlinear = _measure(_pointwise_gate)
    assert linear.additivity > 0.95, linear.to_dict()
    assert nonlinear.additivity < linear.additivity


def test_reports_at_least_three_properties() -> None:
    props = _measure(_identity)
    meta = props.as_metadata()
    assert len(meta) >= 3
    assert set(ALGEBRAIC_PROPERTY_NAMES) <= set(meta)
    assert all(isinstance(v, float) for v in meta.values())


def test_probe_fails_loud_on_shape_change() -> None:
    def bad(x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1)  # drops a rank -> [B, L]

    with pytest.raises(ValueError):
        _measure(bad)


def test_tells_with_precomputed_fx_match_direct_calls() -> None:
    """The fx=/fy= dedup must be a pure forward-count saving, never a value change."""
    from component_fab.proposer.algebraic_properties import (
        additivity,
        convex_range_fraction,
        cross_token_mixing,
        idempotence,
    )

    gen = torch.Generator().manual_seed(11)
    x = torch.randn(2, _SEQ, _DIM, generator=gen)
    y = torch.randn(2, _SEQ, _DIM, generator=gen)
    for f in (_softmax_qk_mixer, _mean_pool, _token_difference):
        fx, fy = f(x), f(y)
        assert convex_range_fraction(f, x) == convex_range_fraction(f, x, fx=fx)
        assert idempotence(f, x) == idempotence(f, x, fx=fx)
        assert additivity(f, x, y) == additivity(f, x, y, fx=fx, fy=fy)
        g1 = torch.Generator().manual_seed(3)
        g2 = torch.Generator().manual_seed(3)
        assert cross_token_mixing(f, x, generator=g1) == cross_token_mixing(
            f, x, generator=g2, fx=fx
        )
