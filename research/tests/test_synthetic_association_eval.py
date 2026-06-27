from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrizations import weight_norm

from research.eval import synthetic_association_eval as sae


class _AssociationOracle(torch.nn.Module):
    vocab_size = 64

    def __init__(self, layout: sae.AssociationLayout) -> None:
        super().__init__()
        self.layout = layout

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for idx in range(batch):
            noun = int(input_ids[idx, 0].item())
            relation = int(input_ids[idx, 1].item())
            target = sae._association_target_int(noun, relation, self.layout)
            logits[idx, 1, target] = 10.0
        return logits


class _TinyVocabModel(torch.nn.Module):
    vocab_size = 12

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)


class _ZeroLogitModel(torch.nn.Module):
    vocab_size = 64

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)


class _NounRelationLearner(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 48) -> None:
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


class _WeightNormNounRelationLearner(_NounRelationLearner):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 48) -> None:
        super().__init__(vocab_size=vocab_size, hidden_dim=hidden_dim)
        weight_norm(self.out, "weight")


class _RelationOnlyLearner(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.relation_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.relation_emb(input_ids[:, 1]))
        pred = self.out(hidden)
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[:, 1, :] = pred
        return logits


def test_layout_keeps_active_vocab_in_20_to_40_word_range():
    layout = sae._make_layout(32)

    assert layout.active_vocab_size == 32
    assert layout.n_per_type == 9
    assert layout.noun_lo == 4
    assert layout.adjective_hi <= 32
    assert layout.chance == 0.25


def test_association_mappings_are_one_to_one_for_supported_vocab_sizes():
    for active_vocab_size in (20, 32, 40):
        layout = sae._make_layout(active_vocab_size)
        nouns = range(layout.noun_lo, layout.noun_hi)
        verb_targets = {
            sae._association_target_int(noun, sae._VERB_QUERY, layout) for noun in nouns
        }
        adjective_targets = {
            sae._association_target_int(noun, sae._ADJ_QUERY, layout) for noun in nouns
        }

        assert len(verb_targets) == layout.n_per_type
        assert len(adjective_targets) == layout.n_per_type
        assert all(layout.verb_lo <= target < layout.verb_hi for target in verb_targets)
        assert all(
            layout.adjective_lo <= target < layout.adjective_hi
            for target in adjective_targets
        )


def test_train_batch_does_not_include_answer_token():
    layout = sae._make_layout(32)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)

    input_ids, targets = sae._make_train_batch(layout, 128, "cpu", rng)

    assert torch.all(input_ids[:, 2] == sae._PAD)
    assert not torch.any(targets == sae._PAD)
    assert not torch.any(input_ids[:, 2] == targets)


def test_forced_choice_oracle_gets_verb_and_adjective_associations():
    layout = sae._make_layout(32)
    model = _AssociationOracle(layout)

    verb_acc = sae._eval_forced_choice_accuracy(
        model,
        layout,
        relation_id=sae._VERB_QUERY,
        eval_repeats=2,
        batch_size=8,
        device="cpu",
    )
    adjective_acc = sae._eval_forced_choice_accuracy(
        model,
        layout,
        relation_id=sae._ADJ_QUERY,
        eval_repeats=2,
        batch_size=8,
        device="cpu",
    )

    assert verb_acc == 1.0
    assert adjective_acc == 1.0


def test_zero_logits_are_at_chance_on_balanced_forced_choice():
    layout = sae._make_layout(32)
    model = _ZeroLogitModel()

    verb_acc = sae._eval_forced_choice_accuracy(
        model,
        layout,
        relation_id=sae._VERB_QUERY,
        eval_repeats=4,
        batch_size=16,
        device="cpu",
    )
    adjective_acc = sae._eval_forced_choice_accuracy(
        model,
        layout,
        relation_id=sae._ADJ_QUERY,
        eval_repeats=4,
        batch_size=16,
        device="cpu",
    )

    assert verb_acc == 0.25
    assert adjective_acc == 0.25


def test_public_probe_reports_vocab_too_small_without_training():
    result = sae.synthetic_association_score(
        _TinyVocabModel(),
        active_vocab_size=32,
        n_train_steps=1,
        device="cpu",
    )

    payload = result.to_dict()
    assert payload["synthetic_association_status"] == "model_vocab_too_small"
    assert payload["synthetic_association_metric_version"] == "synthetic_association_v1"


# Capability experiment (300-step training); hits the probe's internal
# timeout under CPU-only budgets — run via the slow lane.
@pytest.mark.slow
def test_trainable_noun_relation_model_learns_probe():
    result = sae.synthetic_association_score(
        _NounRelationLearner(),
        active_vocab_size=32,
        n_train_steps=300,
        eval_repeats=4,
        batch_size=32,
        lr=2e-3,
        device="cpu",
        seed=123,
    )

    assert result.status == "ok"
    assert result.verb_accuracy >= 0.80
    assert result.adjective_accuracy >= 0.80
    assert result.score >= 0.70


def test_weight_norm_model_runs_and_restores_state():
    model = _WeightNormNounRelationLearner()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = sae.synthetic_association_score(
        model,
        active_vocab_size=32,
        n_train_steps=2,
        eval_repeats=1,
        batch_size=8,
        device="cpu",
        seed=123,
    )

    assert result.status == "ok"
    assert not model.training
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


# Negative-control experiment (100-step training) — run via the slow lane.
@pytest.mark.slow
def test_relation_only_model_does_not_pass_probe():
    result = sae.synthetic_association_score(
        _RelationOnlyLearner(),
        active_vocab_size=32,
        n_train_steps=100,
        eval_repeats=4,
        batch_size=32,
        lr=2e-3,
        device="cpu",
        seed=123,
    )

    assert result.score <= 0.20
