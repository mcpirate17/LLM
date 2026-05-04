"""Focused tests for the nano_blimp v3 real-word held-out probe.

The v3 probe trains a model to predict the offset-rule-associated verb
after ``[the, noun]`` for an in-distribution real-word noun subset, then
evaluates class coherence, binding fidelity, and order grammaticality on
both training-noun and held-out-noun splits."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from research.eval import nano_blimp_eval as nbe


# ── Test fixtures ───────────────────────────────────────────────────────


class _PerfectV3Oracle(nn.Module):
    """Returns +20 dB on the offset-rule-associated verb whenever the
    second-to-last token is a known noun. Used to verify pair-builder
    semantics on a known-good model.

    Carries one dummy parameter so AdamW has something to optimize during
    the brief training phase — the parameter doesn't influence logits."""

    def __init__(self, layout: nbe.RealWordLayout, vocab_size: int = 50257) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.layout = layout
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        out = torch.full((b, s, self.vocab_size), -10.0, device=x.device)
        noun_to_idx = {nid: i for i, nid in enumerate(self.layout.noun_ids)}
        for i in range(b):
            for pos in range(s - 1):
                tok = int(x[i, pos].item())
                if tok in noun_to_idx:
                    n_idx = noun_to_idx[tok]
                    v_idx = nbe._associated_verb_idx(n_idx, self.layout.n_per_type)
                    out[i, pos, self.layout.verb_ids[v_idx]] = 10.0
        # zero-coupled dummy keeps the parameter in the autograd graph.
        return out + 0 * self.dummy.sum()


class _ZeroLogitV3Model(nn.Module):
    vocab_size = 50257

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        return torch.zeros(b, s, self.vocab_size, device=x.device) + 0 * self.dummy


class _TinyVocabV3Model(nn.Module):
    vocab_size = 64  # smaller than any real-word token id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        return torch.zeros(b, s, self.vocab_size, device=x.device)


class _RealWordLearner(nn.Module):
    """Tiny token-embedding learner over the gpt2 50257 vocab. Has enough
    capacity to learn the offset rule on the in-dist split."""

    def __init__(self, vocab_size: int = 50257, hidden: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, hidden)
        self.h2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        # condition position-1 logits on tokens 0+1 (determiner+noun).
        ctx = torch.cat([e[:, 0, :], e[:, 1, :]], dim=-1)
        h = torch.tanh(self.h2(ctx))
        logits = torch.zeros(x.shape[0], x.shape[1], self.vocab_size, device=x.device)
        logits[:, 1, :] = self.out(h)
        # last-position logits zero — pair_accuracy uses sum-log-prob
        # over the full sequence, so position 0 and 2 contribute equally
        # to good and bad pairs.
        return logits


class _WeightNormV3Learner(_RealWordLearner):
    def __init__(self, vocab_size: int = 50257, hidden: int = 32) -> None:
        super().__init__(vocab_size=vocab_size, hidden=hidden)
        weight_norm(self.out, "weight")


# ── Layout / single-token filtering ─────────────────────────────────────


def test_filter_single_token_words_keeps_only_one_token_words():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    pairs = nbe._filter_single_token_words(("dog", "carpenter", "cat"), enc)

    # 'carpenter' tokenizes to multiple gpt2 tokens — must be excluded.
    words = [w for w, _ in pairs]
    assert "dog" in words
    assert "cat" in words
    assert "carpenter" not in words


def test_build_real_word_layout_uses_codex_word_lists():
    layout = nbe.build_real_word_layout(n_per_type=8)

    assert layout.n_per_type == 8
    assert len(layout.nouns) == 8
    assert len(layout.verbs) == 8
    assert len(layout.noun_ids) == 8
    assert len(layout.verb_ids) == 8
    # All ids must be distinct between nouns and verbs (first 8 of each list).
    assert set(layout.noun_ids).isdisjoint(set(layout.verb_ids))
    # 'the' is the determiner for the curated grammar.
    assert layout.determiner == "the"


def test_build_real_word_layout_raises_when_too_many_words_requested():
    # Codex's curated lists are <100 words each.
    try:
        nbe.build_real_word_layout(n_per_type=10_000)
    except ValueError:
        return
    raise AssertionError("expected ValueError for absurd n_per_type")


# ── Held-out selection ──────────────────────────────────────────────────


def test_select_held_out_v3_is_deterministic():
    layout = nbe.build_real_word_layout(n_per_type=16)
    a = nbe._select_held_out_v3(layout, 4, seed=7)
    b = nbe._select_held_out_v3(layout, 4, seed=7)

    assert a == b
    assert len(a) == 4
    assert len(set(a)) == 4
    assert all(0 <= idx < 16 for idx in a)


def test_select_held_out_v3_changes_with_seed():
    layout = nbe.build_real_word_layout(n_per_type=16)
    assert nbe._select_held_out_v3(layout, 4, seed=1) != nbe._select_held_out_v3(
        layout, 4, seed=2
    )


def test_select_held_out_v3_clamps_to_leave_in_dist_pool():
    layout = nbe.build_real_word_layout(n_per_type=8)
    # Asking for too many — must leave at least 2 in-dist nouns.
    held = nbe._select_held_out_v3(layout, 100, seed=42)
    assert len(held) <= layout.n_per_type - 2


# ── Pair builders ───────────────────────────────────────────────────────


def test_class_pairs_use_determiner_as_position_zero():
    layout = nbe.build_real_word_layout(n_per_type=8)
    good, bad = nbe._v3_class_pairs(layout, "cpu", (0, 1, 2))

    assert good.shape == (3, 3)
    assert bad.shape == (3, 3)
    assert torch.all(good[:, 0] == layout.determiner_id)
    assert torch.all(bad[:, 0] == layout.determiner_id)
    # Good ends in a verb id; bad ends in a noun id.
    for row in good.tolist():
        assert row[2] in layout.verb_ids
    for row in bad.tolist():
        assert row[2] in layout.noun_ids


def test_binding_distractor_targets_are_real_in_dist_verbs():
    layout = nbe.build_real_word_layout(n_per_type=8)
    in_dist = (0, 1, 2, 3, 4, 5)  # held-out: indices 6 and 7
    held = (6, 7)

    good, bad = nbe._v3_binding_pairs(layout, "cpu", held, distractor_pool=in_dist)

    valid_distractor_verb_ids = {
        layout.verb_ids[nbe._associated_verb_idx(idx, layout.n_per_type)]
        for idx in in_dist
    }
    for row in bad.tolist():
        assert row[2] in valid_distractor_verb_ids
    # Good binding is always the held-out noun's own associated verb.
    for k, ni in enumerate(held):
        if k < good.shape[0]:
            expected_v = layout.verb_ids[
                nbe._associated_verb_idx(ni, layout.n_per_type)
            ]
            assert int(good[k, 2]) == expected_v


def test_order_pairs_swap_noun_and_verb_positions():
    layout = nbe.build_real_word_layout(n_per_type=8)
    good, bad = nbe._v3_order_pairs(layout, "cpu", (0, 1, 2))

    assert good.shape == bad.shape == (3, 3)
    # determiner stays at position 0; noun/verb swap.
    assert torch.equal(good[:, 0], bad[:, 0])
    assert torch.equal(good[:, 1], bad[:, 2])
    assert torch.equal(good[:, 2], bad[:, 1])


def test_train_batch_excludes_held_out_nouns():
    layout = nbe.build_real_word_layout(n_per_type=12)
    in_dist = (0, 1, 2, 3, 4, 5, 6, 7)
    held = (8, 9, 10, 11)
    held_token_ids = {layout.noun_ids[i] for i in held}
    pool = torch.tensor(in_dist, dtype=torch.long)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(7)

    for _ in range(8):
        ids, _ = nbe._v3_make_train_batch(layout, pool, 64, "cpu", rng)
        assert set(int(t) for t in ids[:, 1].tolist()).isdisjoint(held_token_ids)


# ── Public probe ────────────────────────────────────────────────────────


def test_public_probe_reports_vocab_too_small():
    result = nbe.nano_blimp_v3_score(
        _TinyVocabV3Model(), n_per_type=8, n_train_steps=1, device="cpu"
    )

    payload = result.to_dict()
    assert payload["nano_blimp_v3_status"] == "model_vocab_too_small"
    assert payload["nano_blimp_v3_metric_version"] == "nano_blimp_v3_real_word"


def test_oracle_perfect_score_in_dist_and_held_out():
    layout = nbe.build_real_word_layout(n_per_type=12)
    model = _PerfectV3Oracle(layout)

    result = nbe.nano_blimp_v3_score(
        model,
        n_per_type=12,
        n_train_steps=2,
        batch_size=8,
        device="cpu",
        seed=42,
        held_out_count=3,
    )

    assert result.status == "ok"
    assert result.class_coherence_in_dist_acc == 1.0
    assert result.class_coherence_held_out_acc == 1.0
    assert result.binding_fidelity_in_dist_acc == 1.0
    assert result.binding_fidelity_held_out_acc == 1.0
    assert result.order_grammaticality_acc == 1.0
    assert result.held_out_count == 3
    assert len(result.held_out_noun_words) == 3
    assert all(w in layout.nouns for w in result.held_out_noun_words)


def test_zero_logit_model_is_at_chance_floor():
    """All ties → ``g > b`` is False → 0.0 across the board."""
    result = nbe.nano_blimp_v3_score(
        _ZeroLogitV3Model(),
        n_per_type=8,
        n_train_steps=0,
        device="cpu",
        seed=1,
        held_out_count=2,
    )

    assert result.status in ("ok", "timeout")
    assert result.class_coherence_acc == 0.0
    assert result.binding_fidelity_acc == 0.0
    assert result.order_grammaticality_acc == 0.0


def test_held_out_count_zero_disables_held_out_metrics():
    result = nbe.nano_blimp_v3_score(
        _RealWordLearner(),
        n_per_type=8,
        n_train_steps=2,
        batch_size=8,
        device="cpu",
        seed=1,
        held_out_count=0,
    )

    assert result.held_out_count == 0
    assert result.held_out_noun_words == ()
    assert result.held_out_score == 0.0
    assert result.class_coherence_held_out_acc == 0.0
    assert result.binding_fidelity_held_out_acc == 0.0
    assert result.n_held_out_pairs == 0


def test_weight_norm_state_restored_after_probe():
    model = _WeightNormV3Learner()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = nbe.nano_blimp_v3_score(
        model,
        n_per_type=8,
        n_train_steps=2,
        batch_size=8,
        device="cpu",
        seed=1,
    )

    assert result.status == "ok"
    assert not model.training
    after = model.state_dict()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


def test_dict_payload_contains_v3_keys():
    result = nbe.nano_blimp_v3_score(
        _RealWordLearner(),
        n_per_type=8,
        n_train_steps=2,
        batch_size=8,
        device="cpu",
        seed=42,
    )

    payload = result.to_dict()
    for key in (
        "nano_blimp_v3_score",
        "nano_blimp_v3_class_coherence_in_dist_acc",
        "nano_blimp_v3_class_coherence_held_out_acc",
        "nano_blimp_v3_binding_fidelity_in_dist_acc",
        "nano_blimp_v3_binding_fidelity_held_out_acc",
        "nano_blimp_v3_held_out_score",
        "nano_blimp_v3_held_out_noun_words",
        "nano_blimp_v3_metric_version",
        "nano_blimp_v3_encoding",
    ):
        assert key in payload, key


def test_metric_version_namespace_is_distinct_from_v2():
    # v2 and v3 must coexist with separate namespaces so a single result
    # row can carry both probe outputs without key collisions.
    assert nbe.NANO_BLIMP_METRIC_VERSION == "nano_blimp_v2"
    assert nbe.NANO_BLIMP_V3_METRIC_VERSION == "nano_blimp_v3_real_word"


def test_associated_verb_idx_is_offset_by_one():
    assert nbe._associated_verb_idx(0, 8) == 1
    assert nbe._associated_verb_idx(7, 8) == 0  # wraparound
    assert nbe._associated_verb_idx(3, 12) == 4
