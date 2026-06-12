"""Regression guards for the slotted-memory lanes.

These lanes were repeatedly regressed in one session (acausal pooling leak, a frozen
argmax router with zero gradient, a content-blind slot-sum read, and several times
gutted to a bare ``nn.Linear`` stub). These tests lock in the properties every
legitimate slotted-memory lane must have, across all three real implementations, so a
silent revert fails CI instead of a training run.
"""

import pytest
import torch

from component_fab.generator.memory_primitives import (
    MultiHeadSlotTableMemoryLane,
    SlotTableMemoryLane,
    UniversalMasterLane,
)

LANES = [SlotTableMemoryLane, MultiHeadSlotTableMemoryLane, UniversalMasterLane]


def _legacy_multi_head_forward(
    lane: MultiHeadSlotTableMemoryLane, x: torch.Tensor
) -> torch.Tensor:
    b, seq_len, _ = x.shape
    h, s, hd = lane.n_heads, lane.n_slots, lane.head_dim
    q = torch.tanh(lane.q(x)).view(b, seq_len, h, hd)
    k = torch.tanh(lane.k(x)).view(b, seq_len, h, hd)
    v = torch.tanh(lane.v(x)).view(b, seq_len, h, hd)
    route = torch.softmax(
        lane.write_route(k.reshape(b, seq_len, -1)).view(b, seq_len, h, s),
        dim=-1,
    )
    weighted_key = route.unsqueeze(-1) * k.unsqueeze(3)
    weighted_val = route.unsqueeze(-1) * v.unsqueeze(3)
    denom = route.cumsum(dim=1).clamp_min(1e-6).unsqueeze(-1)
    slot_key = weighted_key.cumsum(dim=1) / denom
    slot_val = weighted_val.cumsum(dim=1) / denom
    slot_key = torch.cat(
        [slot_key.new_zeros(slot_key[:, :1].shape), slot_key[:, :-1]], dim=1
    )
    slot_val = torch.cat(
        [slot_val.new_zeros(slot_val[:, :1].shape), slot_val[:, :-1]], dim=1
    )
    scores = torch.einsum("blhd,blhsd->blhs", q, slot_key) * (hd**-0.5)
    read = torch.einsum("blhs,blhsd->blhd", torch.softmax(scores, dim=-1), slot_val)
    return lane.out(read.reshape(b, seq_len, lane.memory_dim))


@pytest.mark.parametrize("cls", LANES)
def test_is_a_real_mechanism_not_a_stub(cls: type) -> None:
    lane = cls(64)
    children = dict(lane.named_children())
    for name in ("q", "k", "v", "write_route", "out"):
        assert name in children, f"{cls.__name__} missing {name!r} — stubbed/gutted"
    assert "proj" not in children, f"{cls.__name__} is a bare nn.Linear stub"


@pytest.mark.parametrize("cls", LANES)
def test_strictly_causal_no_future_leak(cls: type) -> None:
    torch.manual_seed(0)
    lane = cls(64).eval()
    seq = 13
    x = torch.randn(1, seq, 64, requires_grad=True)
    y = lane(x)
    future_edges = []
    for i in range(seq):
        (g,) = torch.autograd.grad(
            y[0, i].sum(), x, retain_graph=True, allow_unused=True
        )
        if g is None:
            continue
        gnorm = g[0].norm(dim=-1)
        future_edges += [(i, j) for j in range(seq) if j > i and gnorm[j].item() > 1e-9]
    assert not future_edges, (
        f"{cls.__name__} acausal: out[i]<-x[j>i] {future_edges[:5]}"
    )


@pytest.mark.parametrize("cls", LANES)
def test_write_router_is_differentiable(cls: type) -> None:
    torch.manual_seed(0)
    lane = cls(64)
    lane(torch.randn(2, 16, 64)).pow(2).mean().backward()
    g = lane.write_route.weight.grad
    assert g is not None, f"{cls.__name__} write_route has no gradient path (argmax?)"
    assert g.abs().sum().item() > 0.0, f"{cls.__name__} write_route gradient is zero"


@pytest.mark.parametrize("cls", LANES)
def test_read_is_content_addressed(cls: type) -> None:
    """Output must consult slot keys (softmax(q·slot_key)·slot_val), not a slot-sum."""
    torch.manual_seed(0)
    lane = cls(64).eval()
    x = torch.randn(1, 48, 64)
    with torch.no_grad():
        y_full = lane(x)
        saved = lane.k.weight.clone()
        lane.k.weight.zero_()
        y_nokey = lane(x)
        lane.k.weight.copy_(saved)
        rel = (
            (y_full - y_nokey).abs().mean() / y_full.abs().mean().clamp_min(1e-9)
        ).item()
    assert rel > 1e-5, f"{cls.__name__} read ignores slot keys (rel={rel:.2e})"


def test_multi_head_baseline_flags_match_original_equations() -> None:
    torch.manual_seed(4)
    lane = MultiHeadSlotTableMemoryLane(
        16,
        memory_dim=12,
        n_slots=3,
        n_heads=3,
        use_null_write=False,
        use_composer=False,
        use_delta_update=False,
        normalize_read=False,
    ).eval()
    x = torch.randn(2, 9, 16)
    assert torch.allclose(lane(x), _legacy_multi_head_forward(lane, x), atol=1e-6)


def test_null_write_gate_can_suppress_all_memory_updates() -> None:
    torch.manual_seed(5)
    lane = MultiHeadSlotTableMemoryLane(
        16,
        memory_dim=16,
        n_slots=4,
        n_heads=4,
        use_composer=False,
        use_delta_update=False,
        normalize_read=False,
    ).eval()
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        lane.write_gate.weight.zero_()
        lane.write_gate.bias.fill_(-40.0)
        suppressed = lane(x)
        lane.write_gate.bias.fill_(40.0)
        enabled = lane(x)
    assert suppressed.abs().max().item() < 1e-8
    assert enabled.abs().mean().item() > 1e-4


def test_causal_composer_binds_current_and_two_previous_tokens() -> None:
    lane = MultiHeadSlotTableMemoryLane(
        4,
        memory_dim=4,
        n_slots=1,
        n_heads=1,
        composer_width=3,
    )
    x = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    with torch.no_grad():
        lane.composer.weight[:, 0, 0].fill_(0.2)
        lane.composer.weight[:, 0, 1].fill_(0.3)
        lane.composer.weight[:, 0, 2].fill_(0.5)
    composed = lane._compose(x)
    assert torch.allclose(composed[:, 0], 0.5 * x[:, 0])
    assert torch.allclose(composed[:, 2], 0.2 * x[:, 0] + 0.3 * x[:, 1] + 0.5 * x[:, 2])


def test_delta_update_replaces_slot_when_write_weight_is_one() -> None:
    lane = MultiHeadSlotTableMemoryLane(
        2,
        memory_dim=2,
        n_slots=1,
        n_heads=1,
        use_composer=False,
        use_delta_update=True,
    )
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[2.0, 2.0]]]])
    value = key + 10.0
    write_weight = torch.ones(1, 3, 1, 1)
    slot_key, slot_value = lane._prewrite_slot_states(key, value, write_weight)
    assert torch.allclose(slot_key[:, 0], torch.zeros_like(slot_key[:, 0]))
    assert torch.allclose(slot_key[:, 1], key[:, 0], atol=2e-6)
    assert torch.allclose(slot_key[:, 2], key[:, 1], atol=2e-6)
    assert torch.allclose(slot_value[:, 2], value[:, 1], atol=2e-6)


def test_normalized_read_is_invariant_to_positive_query_key_scaling() -> None:
    torch.manual_seed(6)
    lane = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        normalize_read=True,
    )
    q = torch.randn(2, 5, 2, 4)
    slot_key = torch.randn(2, 5, 2, 3, 4)
    slot_value = torch.randn(2, 5, 2, 3, 4)
    base = lane._read_slots(q, slot_key, slot_value)
    scaled = lane._read_slots(7.0 * q, 0.25 * slot_key, slot_value)
    assert torch.allclose(base, scaled, atol=1e-6)


def test_slot_value_rmsnorm_matches_manual_normalization() -> None:
    torch.manual_seed(7)
    plain = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        normalize_slot_values=False,
    )
    normalized = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        normalize_slot_values=True,
    )
    normalized.load_state_dict(plain.state_dict())
    q = torch.randn(2, 5, 2, 4)
    slot_key = torch.randn(2, 5, 2, 3, 4)
    slot_value = torch.randn(2, 5, 2, 3, 4)
    rms = torch.rsqrt(slot_value.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = plain._read_slots(q, slot_key, slot_value * rms)
    actual = normalized._read_slots(q, slot_key, slot_value)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_grouped_router_uses_per_head_route_modules() -> None:
    torch.manual_seed(7)
    lane = MultiHeadSlotTableMemoryLane(
        16,
        memory_dim=16,
        n_slots=4,
        n_heads=4,
        grouped_router=True,
    )
    assert isinstance(lane.write_route, torch.nn.ModuleList)
    assert len(lane.write_route) == 4
    x = torch.randn(2, 10, 16, requires_grad=True)
    y = lane(x)
    assert y.shape == x.shape
    y.pow(2).mean().backward()
    assert all(router.weight.grad is not None for router in lane.write_route)


def test_query_lift_is_zero_init_residual_and_causal() -> None:
    torch.manual_seed(8)
    lane = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=2,
        n_heads=2,
        use_query_lift=True,
    )
    x = torch.randn(1, 7, 8)
    memory_input = lane._compose(x)
    assert torch.allclose(lane._lift_query(x, memory_input), memory_input)
    with torch.no_grad():
        lane.query_lift.weight.fill_(0.25)
    lifted = lane._lift_query(x, memory_input)
    changed = x.clone()
    changed[:, 5:] += 100.0
    changed_memory = lane._compose(changed)
    changed_lifted = lane._lift_query(changed, changed_memory)
    assert torch.allclose(lifted[:, :5], changed_lifted[:, :5], atol=1e-6)


def test_route_from_input_uses_composed_lane_representation() -> None:
    lane = MultiHeadSlotTableMemoryLane(
        16,
        memory_dim=12,
        n_slots=3,
        n_heads=3,
        route_from_input=True,
    )
    assert lane.write_route.in_features == 16
    x = torch.randn(2, 9, 16, requires_grad=True)
    lane(x).pow(2).mean().backward()
    assert lane.write_route.weight.grad is not None


def test_router_prior_is_zero_init_residual_before_null_write_gate() -> None:
    torch.manual_seed(12)
    common = dict(
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        use_null_write=True,
        use_composer=True,
        use_delta_update=False,
        normalize_read=True,
        route_from_input=True,
        normalize_slot_values=True,
    )
    base = MultiHeadSlotTableMemoryLane(8, **common)
    prior = MultiHeadSlotTableMemoryLane(8, **common, use_router_prior=True)
    prior.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 11, 8)
    assert torch.allclose(prior(x), base(x), atol=1e-6)

    with torch.no_grad():
        prior.route_proto_beta.fill_(1.0)
        prior.write_gate.weight.zero_()
        prior.write_gate.bias.fill_(-40.0)
    assert prior(x).abs().max().item() < 1e-8


def test_router_prior_preserves_same_seed_base_initialization() -> None:
    common = dict(
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        use_delta_update=False,
        route_from_input=True,
        normalize_slot_values=True,
    )
    torch.manual_seed(13)
    base = MultiHeadSlotTableMemoryLane(8, **common)
    torch.manual_seed(13)
    prior = MultiHeadSlotTableMemoryLane(8, **common, use_router_prior=True)
    x = torch.randn(2, 9, 8)
    assert torch.allclose(prior(x), base(x), atol=1e-6)


def test_grouped_router_rejects_route_from_input() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        MultiHeadSlotTableMemoryLane(
            16,
            memory_dim=16,
            n_slots=4,
            n_heads=4,
            grouped_router=True,
            route_from_input=True,
        )


@pytest.mark.parametrize("normalize_read", [False, True])
def test_identity_bilinear_read_matches_dot_product(normalize_read: bool) -> None:
    torch.manual_seed(9)
    plain = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        normalize_read=normalize_read,
        bilinear_read=False,
    )
    bilinear = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        normalize_read=normalize_read,
        bilinear_read=True,
    )
    if normalize_read:
        bilinear.log_read_scale.data.copy_(plain.log_read_scale.data)
    q = torch.randn(2, 5, 2, 4)
    slot_key = torch.randn(2, 5, 2, 3, 4)
    slot_value = torch.randn(2, 5, 2, 3, 4)
    expected = plain._read_slots(q, slot_key, slot_value)
    actual = bilinear._read_slots(q, slot_key, slot_value)
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "option",
    [
        {"refine_write_route": True},
        {"consolidate_slots": True},
    ],
)
def test_zero_initialized_structural_refinement_matches_base(
    option: dict[str, bool],
) -> None:
    torch.manual_seed(10)
    common = dict(
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        use_null_write=True,
        use_composer=True,
        use_delta_update=False,
        normalize_read=True,
        route_from_input=True,
    )
    base = MultiHeadSlotTableMemoryLane(8, **common)
    refined = MultiHeadSlotTableMemoryLane(8, **common, **option)
    refined.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 11, 8)
    assert torch.allclose(refined(x), base(x), atol=1e-6)


@pytest.mark.parametrize(
    "option",
    [
        {"refine_write_route": True},
        {"consolidate_slots": True},
    ],
)
def test_structural_refinement_remains_causal(option: dict[str, bool]) -> None:
    torch.manual_seed(11)
    lane = MultiHeadSlotTableMemoryLane(
        8,
        memory_dim=8,
        n_slots=3,
        n_heads=2,
        use_null_write=True,
        use_composer=True,
        use_delta_update=False,
        normalize_read=True,
        route_from_input=True,
        **option,
    ).eval()
    with torch.no_grad():
        if lane.refine_write_route:
            lane.content_route_scale.fill_(1.0)
        if lane.consolidate_slots:
            lane.consolidation_gate.fill_(0.5)
    x = torch.randn(1, 12, 8)
    changed = x.clone()
    changed[:, 7:] += torch.randn_like(changed[:, 7:])
    assert torch.allclose(lane(x)[:, :7], lane(changed)[:, :7], atol=1e-6)
