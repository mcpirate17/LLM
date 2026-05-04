from __future__ import annotations

import pytest
import torch

from research.eval import nano_controlled_holdout_probe as nhp


class _NanoOracle(torch.nn.Module):
    """Returns log-probs that perfectly score the correct continuation
    of every eval item.  Used to confirm the harness can hit 1.0."""

    def __init__(
        self, corpus: nhp.NanoControlledHoldoutCorpus, vocab_size: int
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.next_by_prefix: dict[tuple[int, ...], int] = {}
        for item in corpus.eval_items:
            full = item.prefix + item.correct
            tokens = nhp._sentence_tokens(
                full, vocab_size=self.vocab_size, corpus=corpus
            )
            for pos in range(len(tokens) - 1):
                self.next_by_prefix[tuple(tokens[: pos + 1])] = tokens[pos + 1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row in range(batch):
            for pos in range(seq_len):
                prefix = tuple(int(x) for x in input_ids[row, : pos + 1].tolist())
                target = self.next_by_prefix.get(prefix)
                if target is not None:
                    logits[row, pos, target] = 10.0
        return logits


class _TinyLearner(torch.nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.rnn = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.emb(input_ids))
        return self.out(hidden)


def test_corpus_holds_out_pair_combinations_strictly():
    corpus = nhp.build_nano_controlled_holdout_corpus(
        active_vocab_size=80,
        vocab_size=128,
        tokenizer="byte",
        n_eval_per_bucket=8,
        n_classes=4,
        seed=7,
    )
    train = set(corpus.train_sentences)
    held_out = [item for item in corpus.eval_items if item.bucket == "held_out_pair"]
    assert held_out, "expected non-empty held_out_pair bucket"
    for item in held_out:
        # The full surface form must never appear in training.
        full = (item.prefix + item.correct).strip()
        assert full not in train
        # The bare two-word "the {noun} {verb}" form must also be unseen.
        head = " ".join(full.split()[:3])
        assert head not in train, head


def test_compositional_bucket_uses_unseen_adj_noun_pair():
    corpus = nhp.build_nano_controlled_holdout_corpus(
        active_vocab_size=80,
        vocab_size=128,
        tokenizer="byte",
        n_eval_per_bucket=8,
        n_classes=4,
        seed=11,
    )
    train = set(corpus.train_sentences)
    composed = [item for item in corpus.eval_items if item.bucket == "compositional"]
    assert composed, "expected non-empty compositional bucket"
    for item in composed:
        # `the {adj} {noun}` prefix must not appear as a train substring.
        prefix = item.prefix.strip()
        assert all(not s.startswith(prefix + " ") for s in train), prefix


def test_oracle_scores_perfect_on_every_bucket():
    corpus = nhp.build_nano_controlled_holdout_corpus(
        active_vocab_size=80,
        vocab_size=128,
        tokenizer="byte",
        n_eval_per_bucket=6,
        n_classes=4,
        seed=3,
    )
    model = _NanoOracle(corpus, vocab_size=128)
    bucket_acc = nhp.evaluate_nano_controlled_holdout(
        model, corpus, vocab_size=128, device="cpu"
    )
    assert bucket_acc["seen"] == 1.0
    assert bucket_acc["held_out_pair"] == 1.0
    assert bucket_acc["compositional"] == 1.0


def test_random_model_is_close_to_chance_score():
    torch.manual_seed(0)
    model = _TinyLearner()
    result = nhp.nano_controlled_holdout_probe(
        model,
        active_vocab_size=80,
        n_train_steps=0,
        n_eval_per_bucket=24,
        n_classes=4,
        batch_size=8,
        device="cpu",
        tokenizer="byte",
        seed=0,
    )
    # No training => model is random => score ≈ chance => normalized ≈ 0.
    # Allow a generous tolerance because the eval set is small.
    assert result.status == "ok"
    assert result.score < 0.5
    assert 0.0 <= result.seen_acc <= 0.6
    assert 0.0 <= result.held_out_pair_acc <= 0.6
    assert 0.0 <= result.compositional_acc <= 0.6


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_probe_runs_and_restores_across_step_budgets(steps: int):
    model = _TinyLearner()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = nhp.nano_controlled_holdout_probe(
        model,
        active_vocab_size=80,
        n_train_steps=steps,
        n_eval_per_bucket=4,
        n_classes=4,
        batch_size=4,
        device="cpu",
        tokenizer="byte",
        seed=123,
    )
    assert result.status == "ok"
    assert result.n_train_steps == steps
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.seen_acc <= 1.0
    assert 0.0 <= result.held_out_pair_acc <= 1.0
    assert 0.0 <= result.compositional_acc <= 1.0
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


def test_probe_restores_state_after_real_training():
    model = _TinyLearner()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    nhp.nano_controlled_holdout_probe(
        model,
        active_vocab_size=80,
        n_train_steps=8,
        n_eval_per_bucket=4,
        n_classes=4,
        batch_size=4,
        device="cpu",
        tokenizer="byte",
        seed=5,
    )
    after = model.state_dict()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), f"weight {key} not restored"
    # Probe must restore the caller's pre-probe training mode (eval here).
    assert not model.training


def test_probe_reports_timeout_with_zero_completed_steps():
    model = _TinyLearner()
    result = nhp.nano_controlled_holdout_probe(
        model,
        active_vocab_size=80,
        n_train_steps=10,
        n_eval_per_bucket=4,
        n_classes=4,
        batch_size=4,
        device="cpu",
        tokenizer="byte",
        timeout_s=-1.0,
        seed=5,
    )
    assert result.status == "timeout"
    assert result.n_train_steps == 0


def test_probe_handles_missing_vocab_size():
    class _NoVocab(torch.nn.Module):
        def forward(self, x):  # pragma: no cover — never called
            return x

    result = nhp.nano_controlled_holdout_probe(
        _NoVocab(),
        active_vocab_size=80,
        n_train_steps=1,
        n_eval_per_bucket=4,
        device="cpu",
        tokenizer="byte",
    )
    assert result.status == "missing_vocab_size"
    assert result.score == 0.0


def test_probe_aggregate_score_weights_holdouts_more_than_seen():
    bucket_acc = {"seen": 1.0, "held_out_pair": 0.25, "compositional": 0.25}
    seen_dom_score, *_ = nhp._aggregate_score(bucket_acc)
    bucket_acc = {"seen": 0.25, "held_out_pair": 1.0, "compositional": 1.0}
    held_dom_score, *_ = nhp._aggregate_score(bucket_acc)
    # A model that aces the held-out buckets but is at chance on seen should
    # outscore a model that aces seen but is at chance on held-out.
    assert held_dom_score > seen_dom_score


def test_corpus_is_deterministic_under_seed():
    a = nhp.build_nano_controlled_holdout_corpus(
        active_vocab_size=80, vocab_size=128, tokenizer="byte", seed=42, n_classes=4
    )
    b = nhp.build_nano_controlled_holdout_corpus(
        active_vocab_size=80, vocab_size=128, tokenizer="byte", seed=42, n_classes=4
    )
    assert a.train_sentences == b.train_sentences
    assert a.eval_items == b.eval_items
