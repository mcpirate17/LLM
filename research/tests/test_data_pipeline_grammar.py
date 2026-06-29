"""Unit tests for the data-pipeline search grammar + routed batcher."""

from __future__ import annotations

import pytest
import torch

from research.synthesis.data_pipeline_grammar import (
    CARRIER_SEGMENT,
    DATA_ORDERS,
    LOCAL_SEGMENT,
    SEQ_FOLDS,
    DataRouteSpec,
    apply_data_route,
    data_route_from_axes,
    data_route_to_axes,
    gate_bias_from_segments,
    route_permutation,
    route_segment_ids,
    route_segments_from_surprisal,
    sample_data_route_spec,
)
from research.training.data_routed_batcher import (
    DataRoutedBatcher,
    maybe_route_batcher,
)


def test_spec_validates_enum_values() -> None:
    DataRouteSpec()  # all-identity is valid
    for kwargs in (
        {"pack": "nope"},
        {"order": "sideways"},
        {"fold": 7},
        {"route": "everywhere"},
    ):
        with pytest.raises(ValueError):
            DataRouteSpec(**kwargs)


def test_axes_round_trip() -> None:
    spec = DataRouteSpec(
        order="bidirectional", fold=16, route="surprisal_split", carrier_fraction=0.25
    )
    axes = data_route_to_axes(spec)
    assert axes == {
        "op_data_pack": "contiguous",
        "op_data_order": "bidirectional",
        "op_seq_fold": 16,
        "op_data_route": "surprisal_split",
        "op_data_carrier_fraction": 0.25,
    }
    assert data_route_from_axes(axes) == spec


def test_identity_route_returns_input_unchanged() -> None:
    tokens = torch.arange(2 * 12).reshape(2, 12)
    out = apply_data_route(tokens, DataRouteSpec())
    assert out is tokens  # identity short-circuits


def test_reverse_flips_sequence() -> None:
    tokens = torch.arange(12).reshape(1, 12)
    out = apply_data_route(tokens, DataRouteSpec(order="reverse"))
    assert torch.equal(out, tokens.flip(-1))


def test_bidirectional_first_half_forward_second_half_reversed() -> None:
    tokens = torch.arange(8).reshape(1, 8)
    out = apply_data_route(tokens, DataRouteSpec(order="bidirectional"))
    # [0,1,2,3] kept; [4,5,6,7] reversed -> [7,6,5,4]
    assert out.flatten().tolist() == [0, 1, 2, 3, 7, 6, 5, 4]


def test_fold_is_serpentine_permutation() -> None:
    # fold=8 over length 16 -> 8 segments of 2, every other one reversed.
    tokens = torch.arange(16).reshape(1, 16)
    out = apply_data_route(tokens, DataRouteSpec(fold=8))
    assert out.flatten().tolist() == [
        0,
        1,
        3,
        2,
        4,
        5,
        7,
        6,
        8,
        9,
        11,
        10,
        12,
        13,
        15,
        14,
    ]


def test_route_is_always_a_token_preserving_permutation() -> None:
    tokens = torch.randint(0, 50, (3, 32))
    for order in DATA_ORDERS:
        for fold in SEQ_FOLDS:
            spec = DataRouteSpec(order=order, fold=fold)
            out = apply_data_route(tokens, spec)
            assert out.shape == tokens.shape
            # a permutation preserves the per-row multiset of tokens
            assert torch.equal(out.sort(-1).values, tokens.sort(-1).values)


def test_route_permutation_is_a_bijection() -> None:
    for order in DATA_ORDERS:
        for fold in SEQ_FOLDS:
            perm = route_permutation(32, DataRouteSpec(order=order, fold=fold))
            assert torch.equal(perm.sort().values, torch.arange(32))


def test_apply_is_deterministic() -> None:
    tokens = torch.randint(0, 99, (4, 16))
    spec = DataRouteSpec(order="bidirectional", fold=8)
    assert torch.equal(apply_data_route(tokens, spec), apply_data_route(tokens, spec))


def test_fold_exceeding_length_fails_loud() -> None:
    with pytest.raises(ValueError, match="exceeds sequence length"):
        apply_data_route(torch.arange(4).reshape(1, 4), DataRouteSpec(fold=8))


def test_unwired_pack_fails_loud() -> None:
    with pytest.raises(NotImplementedError):
        apply_data_route(torch.arange(8).reshape(1, 8), DataRouteSpec(pack="best_fit"))


def test_route_does_not_permute_tokens() -> None:
    # route is a submodule-assignment, orthogonal to the token permutation.
    tokens = torch.arange(8).reshape(1, 8)
    out = apply_data_route(tokens, DataRouteSpec(route="surprisal_split"))
    assert torch.equal(out, tokens)
    assert DataRouteSpec(route="surprisal_split").is_token_identity
    assert not DataRouteSpec(route="surprisal_split").is_identity


def test_sampler_emits_only_wired_specs() -> None:
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        spec = sample_data_route_spec(gen)
        assert spec.pack == "contiguous" and spec.route == "none"
        assert spec.order in DATA_ORDERS and spec.fold in SEQ_FOLDS
        # every sampled spec must be appliable (no fail-loud path)
        apply_data_route(torch.arange(64).reshape(2, 32), spec)


def test_sampler_is_deterministic_for_seed() -> None:
    a = [sample_data_route_spec(torch.Generator().manual_seed(7)) for _ in range(3)]
    b = [sample_data_route_spec(torch.Generator().manual_seed(7)) for _ in range(3)]
    assert a == b


class _FakeBatcher:
    def __init__(self) -> None:
        self.ready = True
        self.calls = 0

    def sample_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        self.calls += 1
        return torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)


def test_routed_batcher_transforms_and_delegates() -> None:
    inner = _FakeBatcher()
    wrapped = DataRoutedBatcher(inner, DataRouteSpec(order="reverse"))
    out = wrapped.sample_batch(2, 8)
    expected = apply_data_route(
        inner.sample_batch(2, 8), DataRouteSpec(order="reverse")
    )
    assert torch.equal(out, expected)
    assert wrapped.ready is True  # delegated to inner via __getattr__


def test_routed_batcher_passes_through_none_batch() -> None:
    class _NoneBatcher:
        def sample_batch(self, *_: object) -> None:
            return None

    wrapped = DataRoutedBatcher(_NoneBatcher(), DataRouteSpec(order="reverse"))
    assert wrapped.sample_batch(2, 8) is None


def test_maybe_route_batcher_skips_identity() -> None:
    inner = _FakeBatcher()
    assert maybe_route_batcher(inner, None) is inner
    assert maybe_route_batcher(inner, DataRouteSpec()) is inner
    assert isinstance(
        maybe_route_batcher(inner, DataRouteSpec(order="reverse")), DataRoutedBatcher
    )


def test_routed_batcher_rejects_unwired_pack() -> None:
    with pytest.raises(NotImplementedError):
        DataRoutedBatcher(_FakeBatcher(), DataRouteSpec(pack="doc_boundary"))


# ---------------- surprisal-driven span routing (signal <-> D) ----------------


def test_surprisal_segments_send_hardest_fraction_to_carrier() -> None:
    # surprisal ascending 0..9; carrier_fraction 0.3 -> top 3 (positions 7,8,9).
    surprisal = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    seg = route_segments_from_surprisal(surprisal, 0.3)
    assert seg.flatten().tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert int((seg == CARRIER_SEGMENT).sum()) == 3


def test_surprisal_segments_extremes() -> None:
    surprisal = torch.rand(2, 16)
    assert (
        int((route_segments_from_surprisal(surprisal, 0.0) == CARRIER_SEGMENT).sum())
        == 0
    )
    assert (
        int((route_segments_from_surprisal(surprisal, 1.0) == LOCAL_SEGMENT).sum()) == 0
    )


def test_route_segment_ids_dispatch() -> None:
    # none -> all local
    assert torch.equal(
        route_segment_ids(DataRouteSpec(), length=6),
        torch.zeros(6, dtype=torch.long),
    )
    # local_global_split -> tail (carrier_fraction) is carrier
    seg = route_segment_ids(DataRouteSpec(route="local_global_split"), length=10)
    assert seg.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    # surprisal_split needs the signal
    with pytest.raises(ValueError, match="surprisal"):
        route_segment_ids(DataRouteSpec(route="surprisal_split"), length=8)
    surprisal = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    seg = route_segment_ids(
        DataRouteSpec(route="surprisal_split", carrier_fraction=0.25),
        surprisal=surprisal,
    )
    assert seg.flatten().tolist() == [0, 0, 0, 0, 0, 0, 1, 1]


def test_gate_bias_signs_match_segments() -> None:
    seg = torch.tensor([[0, 1, 0, 1]])
    bias = gate_bias_from_segments(seg, strength=4.0)
    assert bias.shape == (1, 4, 1)
    assert bias.flatten().tolist() == [-4.0, 4.0, -4.0, 4.0]


def test_surprisal_bias_drives_paired_block_to_carrier() -> None:
    """End-to-end: the surprisal signal routes hard tokens onto B's carrier lane."""
    from component_fab.generator.block_templates import LossMonsterPairedBlock

    torch.manual_seed(0)
    dim = 8
    block = LossMonsterPairedBlock(
        lambda d: torch.nn.Linear(d, d),  # carrier (partner)
        lambda d: torch.nn.Linear(d, d),  # local loss specialist
        dim,
        partner_floor=0.0,  # let routing span the full [0, 1] range
    )
    x = torch.randn(1, 6, dim)

    # High surprisal on the first 3 positions -> those route to the carrier.
    surprisal = torch.tensor([[9.0, 9.0, 9.0, 0.1, 0.1, 0.1]])
    spec = DataRouteSpec(route="surprisal_split", carrier_fraction=0.5)
    seg = route_segment_ids(spec, surprisal=surprisal)
    block.route_prior = gate_bias_from_segments(seg, strength=12.0)
    block(x)
    # gate logit = learned + bias; with strong bias, carrier-routed positions
    # should pull the mean partner (carrier) weight well above the others.
    assert block.last_partner_frac is not None and block.last_partner_frac > 0.5


def test_paired_block_route_prior_shape_checked() -> None:
    from component_fab.generator.block_templates import LossMonsterPairedBlock

    block = LossMonsterPairedBlock(
        lambda d: torch.nn.Linear(d, d), lambda d: torch.nn.Linear(d, d), 8
    )
    block.route_prior = torch.zeros(1, 6, 3)  # last dim must be 1
    with pytest.raises(ValueError, match="route_prior"):
        block(torch.randn(1, 6, 8))
