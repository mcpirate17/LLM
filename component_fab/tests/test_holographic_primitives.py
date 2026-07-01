"""Tests for the holographic (VSA/HRR) binding lane (non-QKV binding, W1 wall).

The decisive tests are the binding algebra: unbind(k, bind(k, v)) recovers v far
above chance, and a superposition of pairs retrieves the value for the queried
key while rejecting the others — the compositional-binding capability that
non-QKV mechanisms usually miss. Plus the usual shape/finiteness, causality, and
anti-softmax-twin structural checks, and end-to-end dispatch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.holographic_primitives import (
    HolographicBindingLane,
    circular_bind,
    circular_unbind,
)
from component_fab.inventor.mechanism_catalog import enumerate_invention_specs
from component_fab.proposer.algebraic_properties import measure_algebraic_properties


def test_bind_unbind_round_trip_recovers_value() -> None:
    torch.manual_seed(0)
    dim = 256
    k = F.normalize(torch.randn(500, dim), dim=-1)
    v = torch.randn(500, dim)
    recovered = circular_unbind(k, circular_bind(k, v))
    cos_true = F.cosine_similarity(recovered, v, dim=-1).mean()
    cos_rand = F.cosine_similarity(recovered, v[torch.randperm(500)], dim=-1).mean()
    assert cos_true > 0.4  # recovers the value...
    assert cos_true > cos_rand + 0.3  # ...far above chance


def test_superposition_retrieves_queried_pair() -> None:
    torch.manual_seed(1)
    dim = 256
    keys = F.normalize(torch.randn(3, dim), dim=-1)
    vals = torch.randn(3, dim)
    memory = sum(circular_bind(keys[i : i + 1], vals[i : i + 1]) for i in range(3))
    read0 = circular_unbind(keys[0:1], memory)
    # The queried key's value is retrieved above the other stored values.
    assert F.cosine_similarity(read0, vals[0:1]).item() > 0.3
    assert (
        F.cosine_similarity(read0, vals[0:1]).item()
        > F.cosine_similarity(read0, vals[1:2]).item() + 0.2
    )


def test_lane_shape_finite_and_grad() -> None:
    torch.manual_seed(0)
    lane = HolographicBindingLane(64)
    x = torch.randn(4, 16, 64, requires_grad=True)
    y = lane(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_lane_is_causal() -> None:
    torch.manual_seed(2)
    lane = HolographicBindingLane(32)
    x_a = torch.randn(1, 16, 32)
    x_b = x_a.clone()
    x_b[:, 9:] += torch.randn(1, 7, 32)
    with torch.no_grad():
        assert torch.allclose(lane(x_a)[:, :9], lane(x_b)[:, :9], atol=1e-5)


def test_lane_requires_min_dim() -> None:
    with pytest.raises(ValueError):
        HolographicBindingLane(1)


def test_lane_is_not_a_softmax_twin() -> None:
    torch.manual_seed(0)
    lane = HolographicBindingLane(64)
    props = measure_algebraic_properties(lane, dim=64, n_seeds=3)
    assert not props.is_softmax_twin(), props.to_dict()


def test_codegen_dispatches_holographic_lane() -> None:
    specs = {
        s.math_axes["op_invention_mechanism"]: s for s in enumerate_invention_specs()
    }
    if "holographic_binding" not in specs:
        pytest.skip("holographic_binding not yet registered in the shared catalog")
    module = generate_module_from_spec(specs["holographic_binding"], dim=16)
    assert isinstance(module, HolographicBindingLane)
    x = torch.randn(2, 8, 16)
    assert module(x).shape == x.shape
