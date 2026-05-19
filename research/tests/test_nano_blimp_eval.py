from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrizations import weight_norm

from research.eval import nano_blimp_eval as nbe
from research.eval import synthetic_association_eval as sae


# ── Minimal model fixtures ──────────────────────────────────────────────


class _ZeroLogitModel(torch.nn.Module):
    """Returns uniform logits; pair accuracies hover near 0.5 (chance)."""

    vocab_size = 64

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        return torch.zeros(b, s, self.vocab_size, device=input_ids.device)


class _PerfectAssociationOracle(torch.nn.Module):
    """Returns logits that perfectly favor the *associated* target token at
    position 1 — used to exercise the eval-only path with held-out aware
    splitting on a known-good model."""

    vocab_size = 64

    def __init__(self, layout: sae.AssociationLayout) -> None:
        super().__init__()
        self.layout = layout

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        logits = torch.full(
            (b, s, self.vocab_size), -10.0, dtype=torch.float32, device=input_ids.device
        )
        for idx in range(b):
            noun = int(input_ids[idx, 0].item())
            relation = int(input_ids[idx, 1].item())
            if self.layout.noun_lo <= noun < self.layout.noun_hi and relation in (
                sae._VERB_QUERY,
                sae._ADJ_QUERY,
            ):
                target = sae._association_target_int(noun, relation, self.layout)
                logits[idx, 1, target] = 10.0
        return logits


class _NounRelationLearner(torch.nn.Module):
    """Tiny learnable model that can in principle learn the offset rule."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.relation_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        noun = input_ids[:, 0]
        relation = input_ids[:, 1]
        hidden = torch.tanh(self.token_emb(noun) + self.relation_emb(relation))
        pred = self.out(hidden)
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[:, 1, :] = pred
        return logits


class _WeightNormLearner(_NounRelationLearner):
    """Same model but with a ``weight_norm`` parametrization on ``out`` —
    forces the probe through the state_dict snapshot/restore code path."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 64) -> None:
        super().__init__(vocab_size=vocab_size, hidden_dim=hidden_dim)
        weight_norm(self.out, "weight")


class _TinyVocabModel(torch.nn.Module):
    vocab_size = 12

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        return torch.zeros(b, s, self.vocab_size, device=input_ids.device)


# ── Held-out selection ──────────────────────────────────────────────────


def test_select_held_out_nouns_is_deterministic_and_in_range():
    layout = sae._make_layout(40)
    a = nbe._select_held_out_nouns(layout, 3, seed=42)
    b = nbe._select_held_out_nouns(layout, 3, seed=42)

    assert a == b
    assert len(a) == 3
    assert len(set(a)) == 3
    assert all(layout.noun_lo <= n < layout.noun_hi for n in a)


def test_select_held_out_nouns_changes_with_seed():
    layout = sae._make_layout(80)
    a = nbe._select_held_out_nouns(layout, 3, seed=1)
    b = nbe._select_held_out_nouns(layout, 3, seed=2)

    assert a != b


def test_select_held_out_nouns_zero_returns_empty():
    layout = sae._make_layout(40)
    assert nbe._select_held_out_nouns(layout, 0, seed=42) == ()


def test_select_held_out_nouns_clamps_to_leave_in_dist():
    layout = sae._make_layout(20)
    held = nbe._select_held_out_nouns(layout, layout.n_per_type + 5, seed=42)

    # At least 2 nouns must remain in-distribution for binding eval.
    assert len(held) <= layout.n_per_type - 2


# ── Pair construction ───────────────────────────────────────────────────


def test_class_coherence_pairs_filter_to_supplied_nouns():
    layout = sae._make_layout(32)
    held = (layout.noun_lo, layout.noun_lo + 1)

    good, bad = nbe._build_class_coherence_pairs(layout, "cpu", held)

    assert good.shape == (4, 3)  # 2 nouns × 2 queries
    assert bad.shape == (4, 3)
    nouns_in_good = set(int(x) for x in good[:, 0].tolist())
    assert nouns_in_good <= set(held)


def test_binding_pairs_use_distractor_pool_only():
    layout = sae._make_layout(32)
    held = (layout.noun_hi - 1,)  # one held-out noun
    in_dist = tuple(n for n in range(layout.noun_lo, layout.noun_hi) if n not in held)

    good, bad = nbe._build_binding_fidelity_pairs(
        layout, "cpu", held, distractor_pool=in_dist
    )

    # 1 held-out noun × 2 queries = 2 pairs (assuming targets differ).
    assert good.shape[0] in (1, 2)
    assert good.shape[0] == bad.shape[0]
    # The bad sequence's target must come from an in-dist noun's mapping
    # — i.e. its third token must equal _association_target_int(some
    # in_dist noun, query, layout).
    valid_bad_targets = set()
    for noun in in_dist:
        for query in (sae._VERB_QUERY, sae._ADJ_QUERY):
            valid_bad_targets.add(sae._association_target_int(noun, query, layout))
    for row in bad.tolist():
        assert int(row[2]) in valid_bad_targets


def test_order_pairs_swap_first_two_tokens():
    layout = sae._make_layout(32)
    nouns = (layout.noun_lo, layout.noun_lo + 1)

    good, bad = nbe._build_order_pairs(layout, "cpu", nouns)

    assert good.shape == bad.shape
    assert good.shape[0] == 4  # 2 nouns × 2 queries
    # bad has the noun and query swapped: bad[:, :2] == good[:, 1::-1]
    flipped = good[:, [1, 0]]
    assert torch.equal(bad[:, :2], flipped)
    # Target column unchanged.
    assert torch.equal(good[:, 2], bad[:, 2])


def test_pair_accuracy_zero_logits_is_chance():
    layout = sae._make_layout(32)
    good, bad = nbe._build_class_coherence_pairs(
        layout, "cpu", tuple(range(layout.noun_lo, layout.noun_hi))
    )
    model = _ZeroLogitModel()

    acc = nbe._pair_accuracy(model, good, bad)

    # Strict tie ⇒ no row counts as "good > bad" ⇒ 0.0.
    assert acc == 0.0


def test_pair_accuracy_handles_empty_tensors():
    empty = torch.empty((0, 3), dtype=torch.long)
    acc = nbe._pair_accuracy(_ZeroLogitModel(), empty, empty)
    assert acc == 0.0


def test_evaluate_pair_split_metrics_weights_split_counts(monkeypatch):
    pairs = {
        "class_in": (
            torch.zeros((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long),
        ),
        "class_held_out": (
            torch.zeros((1, 3), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
        ),
        "binding_in": (
            torch.zeros((1, 3), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
        ),
        "binding_held_out": (
            torch.zeros((3, 3), dtype=torch.long),
            torch.ones((3, 3), dtype=torch.long),
        ),
        "order": (
            torch.zeros((4, 3), dtype=torch.long),
            torch.ones((4, 3), dtype=torch.long),
        ),
    }
    accuracy_by_tensor = {
        id(pairs["class_in"][0]): 0.25,
        id(pairs["class_held_out"][0]): 1.0,
        id(pairs["binding_in"][0]): 0.1,
        id(pairs["binding_held_out"][0]): 0.7,
        id(pairs["order"][0]): 0.9,
    }

    def fake_pair_accuracy(_model, good, _bad):
        return accuracy_by_tensor[id(good)]

    monkeypatch.setattr(nbe, "_pair_accuracy", fake_pair_accuracy)

    metrics = nbe._evaluate_pair_split_metrics(_ZeroLogitModel(), **pairs)

    assert metrics["class_coherence_acc"] == 0.5
    assert metrics["binding_fidelity_acc"] == pytest.approx(0.55)
    assert metrics["order_grammaticality_acc"] == 0.9
    assert metrics["n_in_dist_pairs"] == 2
    assert metrics["n_held_out_pairs"] == 1
    assert metrics["n_pairs_per_test"] == 4


# ── Training-time held-out invariant ────────────────────────────────────


def test_in_dist_train_batch_never_samples_held_out():
    layout = sae._make_layout(40)
    in_dist = tuple(n for n in range(layout.noun_lo, layout.noun_hi - 2))
    pool = torch.tensor(in_dist, dtype=torch.long)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(99)

    held = set(range(layout.noun_lo, layout.noun_hi)) - set(in_dist)
    for _ in range(8):
        ids, _ = nbe._make_in_dist_train_batch(layout, pool, 64, "cpu", rng)
        assert set(int(x) for x in ids[:, 0].tolist()).isdisjoint(held)


# ── Public probe ────────────────────────────────────────────────────────


def test_public_probe_reports_vocab_too_small():
    result = nbe.nano_blimp_score(
        _TinyVocabModel(),
        active_vocab_size=32,
        n_train_steps=1,
        device="cpu",
    )

    payload = result.to_dict()
    assert payload["nano_blimp_status"] == "model_vocab_too_small"
    assert payload["nano_blimp_metric_version"] == "nano_blimp_v2"


def test_public_probe_returns_held_out_fields_with_default_count():
    result = nbe.nano_blimp_score(
        _NounRelationLearner(),
        active_vocab_size=32,
        n_train_steps=2,
        batch_size=8,
        lr=1e-3,
        device="cpu",
        seed=7,
    )

    assert result.held_out_count == nbe._DEFAULT_HELD_OUT_COUNT
    assert len(result.held_out_noun_ids) == nbe._DEFAULT_HELD_OUT_COUNT
    assert result.n_held_out_pairs > 0
    assert result.n_in_dist_pairs > 0


def test_public_probe_zero_held_out_returns_only_in_dist_metrics():
    result = nbe.nano_blimp_score(
        _NounRelationLearner(),
        active_vocab_size=32,
        n_train_steps=2,
        batch_size=8,
        lr=1e-3,
        device="cpu",
        seed=7,
        held_out_count=0,
    )

    assert result.held_out_count == 0
    assert result.held_out_noun_ids == ()
    assert result.held_out_score == 0.0
    assert result.n_held_out_pairs == 0
    assert result.class_coherence_held_out_acc == 0.0
    assert result.binding_fidelity_held_out_acc == 0.0


def test_eval_only_with_held_out_split_separates_in_dist_and_held_out():
    layout = sae._make_layout(40)
    held = (layout.noun_hi - 2, layout.noun_hi - 1)
    model = _PerfectAssociationOracle(layout)

    result = nbe.nano_blimp_eval_only(model, layout, device="cpu", held_out_nouns=held)

    # Oracle perfectly knows every association → both splits should be 1.0.
    assert result.class_coherence_in_dist_acc == 1.0
    assert result.class_coherence_held_out_acc == 1.0
    assert result.binding_fidelity_in_dist_acc == 1.0
    assert result.binding_fidelity_held_out_acc == 1.0
    assert result.held_out_count == 2
    assert tuple(result.held_out_noun_ids) == held


def test_eval_only_without_held_out_is_v1_compatible():
    layout = sae._make_layout(32)
    model = _PerfectAssociationOracle(layout)

    result = nbe.nano_blimp_eval_only(model, layout, device="cpu")

    assert result.held_out_count == 0
    assert result.held_out_noun_ids == ()
    # Overall accuracies populated; held-out fields zero.
    assert result.class_coherence_acc == 1.0
    assert result.binding_fidelity_acc == 1.0
    assert result.class_coherence_held_out_acc == 0.0
    assert result.binding_fidelity_held_out_acc == 0.0


def test_held_out_nouns_never_appear_in_in_dist_eval():
    """The in-dist class/binding eval must not contain any held-out noun
    in the prefix position. Verifies the wiring inside the public probe."""
    layout = sae._make_layout(40)
    held = nbe._select_held_out_nouns(layout, 3, seed=123)
    in_dist = tuple(n for n in range(layout.noun_lo, layout.noun_hi) if n not in held)

    good_c, _ = nbe._build_class_coherence_pairs(layout, "cpu", in_dist)
    good_b, _ = nbe._build_binding_fidelity_pairs(
        layout, "cpu", in_dist, distractor_pool=in_dist
    )

    held_set = set(held)
    assert held_set.isdisjoint(set(int(x) for x in good_c[:, 0].tolist()))
    assert held_set.isdisjoint(set(int(x) for x in good_b[:, 0].tolist()))


def test_weight_norm_model_runs_and_restores_state():
    """The state_dict snapshot/restore path must survive ``weight_norm``.
    Regression for the silent 0.0/0.0 bug that copy.deepcopy caused."""
    model = _WeightNormLearner()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = nbe.nano_blimp_score(
        model,
        active_vocab_size=32,
        n_train_steps=2,
        batch_size=8,
        device="cpu",
        seed=123,
    )

    assert result.status == "ok"
    assert not model.training  # original eval state restored
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


def test_timeout_returns_partial_result():
    result = nbe.nano_blimp_score(
        _NounRelationLearner(),
        active_vocab_size=32,
        n_train_steps=10_000,
        batch_size=8,
        device="cpu",
        seed=1,
        timeout_s=0.05,
    )

    # Either the loop hit the deadline mid-train (status="timeout") or
    # finished fast enough on a tiny model — both are valid; the contract
    # is that the result is well-formed.
    assert result.status in ("timeout", "ok")
    assert result.metric_version == "nano_blimp_v2"
    assert 0.0 <= result.score <= 1.0


def test_dict_payload_contains_v2_keys():
    result = nbe.nano_blimp_score(
        _NounRelationLearner(),
        active_vocab_size=32,
        n_train_steps=2,
        batch_size=8,
        device="cpu",
        seed=42,
    )

    payload = result.to_dict()
    for key in (
        "nano_blimp_score",
        "nano_blimp_class_coherence_acc",
        "nano_blimp_binding_fidelity_acc",
        "nano_blimp_order_grammaticality_acc",
        "nano_blimp_class_coherence_in_dist_acc",
        "nano_blimp_class_coherence_held_out_acc",
        "nano_blimp_binding_fidelity_in_dist_acc",
        "nano_blimp_binding_fidelity_held_out_acc",
        "nano_blimp_held_out_score",
        "nano_blimp_held_out_count",
        "nano_blimp_held_out_noun_ids",
        "nano_blimp_metric_version",
    ):
        assert key in payload, key


def test_metric_version_is_v2():
    assert nbe.NANO_BLIMP_METRIC_VERSION == "nano_blimp_v2"
