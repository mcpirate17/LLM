"""Tests for NM-F9 CDMA slot binding.

Pins the spec: fixed ±1 spreading codes (zero learned params in the binding law),
Gold preferred-pair cross-correlation bounded by ``t(n)``, Hadamard exactly
orthogonal, identity-at-init, strictly-causal exclusive superposition (no
self-retrieval), content-addressed retrieval that survives the randomized-query
control (positions carry no information), and a finite NM-10 physics fingerprint.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.cdma_slot_binding import (
    CDMASlotBinding,
    _m_sequence,
    cdma_param_count,
    gold_cross_correlation_bound,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = CDMASlotBinding(dim=64, n_slots=8, chips=32, code_family="gold")
    x = torch.randn(2, 10, 64)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    ("dim", "chips", "family"),
    [(32, 8, "hadamard"), (62, 31, "gold"), (64, 32, "gold")],
)
def test_identity_at_init(dim: int, chips: int, family: str) -> None:
    """Zero-init output lift ⟹ the mixer is an exact no-op drop-in at init."""
    mix = CDMASlotBinding(dim=dim, n_slots=8, chips=chips, code_family=family)
    x = torch.randn(3, 7, dim)
    assert torch.allclose(mix(x), x, atol=1e-6)


def test_codes_are_fixed_sign_buffers() -> None:
    """The binding law carries zero learned parameters: codes are ±1 buffers,
    identical across instances (deterministic LFSR, no seed dependence)."""
    a = CDMASlotBinding(dim=62, n_slots=16, chips=31, code_family="gold")
    b = CDMASlotBinding(dim=62, n_slots=16, chips=31, code_family="gold")
    assert ((a.codes == 1) | (a.codes == -1)).all()
    assert torch.equal(a.codes, b.codes)
    param_names = {name for name, _ in a.named_parameters()}
    assert "codes" not in param_names
    assert not a.codes.requires_grad


def test_m_sequence_has_maximal_period_and_balance() -> None:
    """Primitive polynomial ⟹ period ``2^n − 1`` with ``2^{n−1}`` ones (the
    m-sequence balance property) for every tabulated preferred-pair polynomial."""
    from research.synthesis.cdma_slot_binding import _PREFERRED_PAIRS

    for degree, (poly_u, poly_v) in _PREFERRED_PAIRS.items():
        for poly in (poly_u, poly_v):
            seq = _m_sequence(poly, degree)
            assert len(seq) == (1 << degree) - 1
            assert sum(seq) == 1 << (degree - 1), (
                f"degree={degree} poly={poly:#x} not balanced -> not primitive"
            )


def test_gold_cross_correlation_within_preferred_pair_bound() -> None:
    """Full-period Gold family (degree 5, 31 chips, 33 codes): every pairwise
    synchronous cross-correlation is bounded by ``t(5) = 9`` — the near-Welch
    property the capacity scaling law rests on. Also exercises MORE slots than
    chips, which orthogonal designs cannot do."""
    mix = CDMASlotBinding(dim=62, n_slots=33, chips=31, code_family="gold")
    assert mix.degree == 5
    gram = mix.codes @ mix.codes.T  # (33, 33)
    diag = torch.diagonal(gram)
    assert torch.equal(diag, torch.full_like(diag, 31.0))
    off = gram - torch.diag_embed(diag)
    bound = gold_cross_correlation_bound(5)
    assert bound == 9
    assert off.abs().max().item() <= bound
    assert mix.interference_bound() == pytest.approx(9 / 31)


def test_hadamard_codes_exactly_orthogonal() -> None:
    mix = CDMASlotBinding(dim=32, n_slots=8, chips=8, code_family="hadamard")
    gram = mix.codes @ mix.codes.T
    assert torch.equal(gram, 8.0 * torch.eye(8))
    assert mix.interference_bound() == 0.0


def _surgery(mix: CDMASlotBinding) -> None:
    """Wire the lifts so tokens are ``[code | payload | 0…]``: key/query = first
    ``chips`` dims, payload = next ``d_v`` dims, gate forced open, selection
    pinned hard (the deploy behavior the exactness tests specify), and the
    write-address taps pinned to the CURRENT position (these tests bind key and
    payload on one token; the default previous-tap init is the two-token
    header-then-payload prior)."""
    mix.selection_hardness = 1.0
    with torch.no_grad():
        mix.write_addr_taps.zero_()
        mix.write_addr_taps[:, 2] = 1.0  # current-position tap
        mix.key_lift.weight.zero_()
        mix.key_lift.weight[:, : mix.chips] = torch.eye(mix.chips)
        mix.query_lift.weight.zero_()
        mix.query_lift.weight[:, : mix.chips] = torch.eye(mix.chips)
        mix.value_compress.weight.zero_()
        mix.value_compress.weight[:, mix.chips : mix.chips + mix.d_v] = torch.eye(
            mix.d_v
        )
        mix.gate_weight.zero_()
        mix.gate_bias.fill_(20.0)  # sigmoid(20) ≈ 1: gate open for every token


def test_content_addressed_retrieval_randomized_positions() -> None:
    """The randomized-query control: bind key→payload at a RANDOM position, query
    at a RANDOM later position, distractors (other slots, junk payloads)
    everywhere else. Retrieval must be exact (Hadamard: zero synchronous
    interference) regardless of where anything sits — content addressing, not a
    positional/recency shortcut."""
    dim, chips, d_v, n_slots, seq = 32, 8, 4, 8, 12
    mix = CDMASlotBinding(dim=dim, n_slots=n_slots, chips=chips, code_family="hadamard")
    _surgery(mix)
    for trial in range(5):
        gen = torch.Generator().manual_seed(trial)
        target = int(torch.randint(0, n_slots, (1,), generator=gen))
        write_pos = int(torch.randint(0, seq - 1, (1,), generator=gen))
        query_pos = int(torch.randint(write_pos + 1, seq, (1,), generator=gen))
        payload = torch.randn(d_v, generator=gen)
        x = torch.zeros(1, seq, dim)
        for pos in range(seq):
            if pos == write_pos:
                x[0, pos, :chips] = mix.codes[target]
                x[0, pos, chips : chips + d_v] = payload
            elif pos == query_pos:
                x[0, pos, :chips] = mix.codes[target]
            else:  # distractor: another slot, junk payload
                other = int(torch.randint(0, n_slots - 1, (1,), generator=gen))
                other += other >= target
                x[0, pos, :chips] = mix.codes[other]
                x[0, pos, chips : chips + d_v] = torch.randn(d_v, generator=gen)
        v_hat, write_idx, read_idx = mix.read_raw(x)
        assert int(write_idx[0, write_pos]) == target
        assert int(read_idx[0, query_pos]) == target
        assert torch.allclose(v_hat[0, query_pos], payload, atol=1e-4), (
            f"trial={trial}: retrieval failed at write={write_pos}, query={query_pos}"
        )


def test_no_self_retrieval_exclusive_prefix() -> None:
    """A token that writes and queries the same slot reads the state of strictly
    EARLIER tokens only — retrieving your own value is the shortcut the exclusive
    prefix sum forbids."""
    mix = CDMASlotBinding(dim=32, n_slots=8, chips=8, code_family="hadamard")
    _surgery(mix)
    x = torch.zeros(1, 1, 32)
    x[0, 0, :8] = mix.codes[3]
    x[0, 0, 8:12] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    v_hat, _, _ = mix.read_raw(x)
    assert torch.equal(v_hat[0, 0], torch.zeros(4))


def test_gold_despreading_error_within_interference_bound() -> None:
    """The Welch-curve claim, analytically: superpose S payloads, despread one —
    the per-element recovery error is exactly the code cross-talk and must sit
    inside ``Σ_{j≠i} |v_j| · t(n)/chips``. Interference is nonzero (Gold ≠
    orthogonal): the capacity trade is real, bounded, and predictable."""
    mix = CDMASlotBinding(dim=62, n_slots=8, chips=31, code_family="gold")
    gen = torch.Generator().manual_seed(0)
    payloads = torch.randn(8, 2, generator=gen)  # (S, d_v)
    memory = (payloads.unsqueeze(-1) * mix.codes.unsqueeze(1)).sum(dim=0)  # (d_v, 31)
    rho = mix.interference_bound()
    max_err = 0.0
    for i in range(8):
        recovered = (memory * mix.codes[i]).sum(dim=-1) / mix.chips
        err = (recovered - payloads[i]).abs()
        others = torch.cat([payloads[:i], payloads[i + 1 :]])
        bound = others.abs().sum(dim=0) * rho
        assert (err <= bound + 1e-5).all(), f"slot {i} breaches interference bound"
        max_err = max(max_err, float(err.max()))
    assert max_err > 0.0  # bounded, not magically zero


def test_backward_flows_through_hard_selection() -> None:
    """STE delivers gradient through the hard top-1 slot select to every lift."""
    mix = CDMASlotBinding(dim=32, n_slots=8, chips=8, code_family="hadamard")
    with torch.no_grad():  # move off identity so gradients are non-trivial
        mix.out_lift.weight.add_(0.3 * torch.randn_like(mix.out_lift.weight))
    x = torch.randn(2, 6, 32, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in mix.named_parameters():
        if not param.requires_grad:  # write taps are frozen structure by default
            continue
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert mix.key_lift.weight.grad.abs().sum() > 0
    assert mix.value_compress.weight.grad.abs().sum() > 0


def test_param_count_excludes_codes_and_frozen_taps() -> None:
    dim, chips = 64, 32
    d_v = dim // chips
    tied = CDMASlotBinding(dim=dim, n_slots=16, chips=chips, code_family="gold")
    expected_tied = chips * dim + 2 * d_v * dim + dim + 1  # no taps: frozen default
    assert cdma_param_count(dim, chips) == expected_tied
    assert tied.num_parameters == expected_tied
    assert expected_tied == sum(p.numel() for p in tied.parameters() if p.requires_grad)
    untied = CDMASlotBinding(
        dim=dim, n_slots=16, chips=chips, code_family="gold", tie_addressing=False
    )
    expected_untied = expected_tied + chips * dim
    assert cdma_param_count(dim, chips, tie_addressing=False) == expected_untied
    assert expected_untied == sum(
        p.numel() for p in untied.parameters() if p.requires_grad
    )


def test_write_taps_frozen_by_default_learnable_opt_in() -> None:
    """F9.2: the previous-tap write shift is STRUCTURE — frozen at the
    header-then-payload prior (ablation: training it away costs up to 0.23 at 16
    pairs). ``learn_write_taps=True`` opts back in and counts the 3·D params."""
    frozen = CDMASlotBinding(dim=64, n_slots=8, chips=32)
    assert not frozen.write_addr_taps.requires_grad
    learn = CDMASlotBinding(dim=64, n_slots=8, chips=32, learn_write_taps=True)
    assert learn.write_addr_taps.requires_grad
    assert learn.num_parameters == frozen.num_parameters + 3 * 64


def test_tied_addressing_shares_the_lift() -> None:
    """F9.1 fix 1: the query lift IS the key lift (one module, one gradient) by
    default; untied mode keeps two."""
    tied = CDMASlotBinding(dim=64, n_slots=8, chips=32)
    assert tied.query_lift is tied.key_lift
    untied = CDMASlotBinding(dim=64, n_slots=8, chips=32, tie_addressing=False)
    assert untied.query_lift is not untied.key_lift


def test_annealed_selection_blends_to_hard() -> None:
    """F9.1 fix 2: at hardness 0 the selection is the normalized Lorentzian
    bounded-reciprocal weighting (sums to 1, argmax preserved, NOT one-hot); at
    hardness 1 it is exactly the v1 hard top-1."""
    torch.manual_seed(0)
    mix = CDMASlotBinding(dim=64, n_slots=8, chips=32)
    assert mix.selection == "annealed" and mix.selection_hardness == 0.0
    logits = torch.randn(3, 5, 8)
    soft = mix._select(logits)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(3, 5), atol=1e-6)
    assert torch.equal(soft.argmax(dim=-1), logits.argmax(dim=-1))
    assert soft.max() < 1.0  # genuinely soft, not one-hot
    mix.selection_hardness = 1.0
    hard = mix._select(logits)
    idx = logits.argmax(dim=-1)
    assert torch.equal(hard.detach().argmax(dim=-1), idx)
    assert torch.allclose(hard.detach().max(dim=-1).values, torch.ones(3, 5), atol=1e-6)
    pinned = CDMASlotBinding(dim=64, n_slots=8, chips=32, selection="hard")
    assert pinned.selection_hardness == 1.0
    with pytest.raises(ValueError):
        CDMASlotBinding(dim=64, n_slots=8, chips=32, selection="warm")


def test_code_alignment_aux_loss() -> None:
    """F9.1 fix 3: ``aux_loss`` is stashed per forward, finite, in [0, 2], and
    exactly 0 when the key lift outputs a code (perfect constellation
    alignment)."""
    mix = CDMASlotBinding(dim=32, n_slots=8, chips=8, code_family="hadamard")
    assert mix.aux_loss is None  # not computed before any forward
    x = torch.randn(2, 6, 32)
    mix(x)
    assert mix.aux_loss is not None
    assert 0.0 <= float(mix.aux_loss.detach()) <= 2.0
    _surgery(mix)  # key lift = identity on the code prefix
    x_aligned = torch.zeros(1, 4, 32)
    x_aligned[0, :, :8] = mix.codes[:4]  # tokens ARE codes ⟹ cos = 1 ⟹ loss = 0
    mix(x_aligned)
    assert float(mix.aux_loss.detach()) == pytest.approx(0.0, abs=1e-5)


def test_invalid_configs_fail_fast() -> None:
    with pytest.raises(ValueError):
        CDMASlotBinding(dim=60, chips=32)  # chips must divide dim
    with pytest.raises(ValueError):
        CDMASlotBinding(dim=64, chips=32, code_family="walsh")  # unknown family
    with pytest.raises(ValueError):
        CDMASlotBinding(dim=32, n_slots=16, chips=8, code_family="hadamard")
    with pytest.raises(ValueError):
        CDMASlotBinding(dim=64, n_slots=1, chips=32)


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: finite physics fingerprint so the mixer is scorable on the
    geometric-novelty axis alongside Monarch/Butterfly/TernarySignMix."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = CDMASlotBinding(dim=16, n_slots=8, chips=8, code_family="hadamard")
    with torch.no_grad():  # nudge off identity for a non-trivial fingerprint
        mix.out_lift.weight.add_(0.4 * torch.randn_like(mix.out_lift.weight))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
