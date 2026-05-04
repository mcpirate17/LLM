from __future__ import annotations

import pytest
import torch

from research.eval import _real_word_vocab as rwv
from research.eval import controlled_sentence_probe as csp
from research.tools import controlled_sentence_tune as cst


class _SentenceOracle(torch.nn.Module):
    def __init__(self, corpus: csp.SentenceProbeCorpus, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.next_by_prefix: dict[tuple[int, ...], int] = {}
        for item in corpus.eval_items:
            for sentence in (item.good_sentence,):
                tokens = csp._sentence_tokens(
                    sentence,
                    vocab_size=self.vocab_size,
                    tokenizer=corpus.tokenizer,
                    tiktoken_encoding=corpus.tiktoken_encoding,
                )
                for pos in range(len(tokens) - 1):
                    self.next_by_prefix[tuple(tokens[: pos + 1])] = tokens[pos + 1]
        for good, _bad in corpus.blimp_pairs:
            tokens = csp._sentence_tokens(
                good,
                vocab_size=self.vocab_size,
                tokenizer=corpus.tokenizer,
                tiktoken_encoding=corpus.tiktoken_encoding,
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


class _TinySentenceLearner(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.rnn = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.emb(input_ids))
        return self.out(hidden)


def test_shared_real_word_vocab_preserves_v3_anchor_order():
    expected_first_24_nouns = (
        "dog",
        "cat",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "bird",
        "horse",
        "cook",
        "teacher",
        "doctor",
        "artist",
        "driver",
        "runner",
        "farmer",
        "guard",
        "singer",
        "student",
        "worker",
        "baby",
        "friend",
        "king",
        "queen",
    )
    expected_first_24_verbs = (
        "ran",
        "sat",
        "jumped",
        "walked",
        "slept",
        "ate",
        "sang",
        "danced",
        "laughed",
        "cried",
        "swam",
        "read",
        "wrote",
        "cooked",
        "drove",
        "worked",
        "played",
        "climbed",
        "looked",
        "waited",
        "stood",
        "fell",
        "came",
        "went",
    )

    assert rwv.REAL_WORD_VOCAB_V1_NOUNS == expected_first_24_nouns
    assert rwv.REAL_WORD_VOCAB_V1_VERBS == expected_first_24_verbs
    assert rwv._REAL_WORD_VOCAB_V1 == (  # noqa: SLF001 - explicit reproducibility pin
        expected_first_24_nouns,
        expected_first_24_verbs,
    )
    assert csp._NOUNS[:24] == expected_first_24_nouns  # noqa: SLF001
    assert csp._VERBS[:24] == expected_first_24_verbs  # noqa: SLF001
    assert rwv.REAL_WORD_VERBS[24:27] == ("ran", "hid", "dug")
    assert rwv.REAL_WORD_NOUNS[47:50] == ("gardener", "mary", "john")


def test_build_sentence_probe_corpus_uses_real_tokenizer_vocab_at_1000_words():
    pytest.importorskip("tiktoken")

    corpus = csp.build_sentence_probe_corpus(
        active_vocab_size=1000,
        vocab_size=50257,
        tokenizer="tiktoken",
        tiktoken_encoding="gpt2",
        n_eval_items=32,
    )

    assert corpus.active_vocab_size == 1000
    assert len(corpus.vocabulary) == 1000
    assert len(corpus.train_sentences) >= 32
    assert len(corpus.eval_items) == 32
    assert corpus.source_counts["babi_eval"] == 32
    assert corpus.source_counts["hellaswag_eval"] == 0
    assert corpus.source_counts["train_real"] >= 32
    assert corpus.source_counts["train_babi"] >= 32
    assert corpus.source_counts["curated_train"] == 0
    assert corpus.eval_items[0].correct.startswith(" ")
    assert corpus.eval_items[0].correct.strip().isalpha()
    assert "title" not in set(corpus.eval_items[0].good_sentence.split())
    assert "substeps" not in set(corpus.eval_items[0].good_sentence.split())
    assert corpus.eval_items[0].source == "babi_qa"


def test_sentence_probe_corpus_keeps_eval_forms_out_of_training():
    corpus = csp.build_sentence_probe_corpus(
        active_vocab_size=80,
        vocab_size=128,
        tokenizer="byte",
        n_eval_items=8,
    )

    train = set(corpus.train_sentences)
    assert corpus.eval_items
    for item in corpus.eval_items:
        assert item.good_sentence not in train
        assert item.bad_order_sentence not in train
        assert item.bad_binding_sentence not in train
        assert len((item.correct, *item.distractors)) == 4
    for good, bad in corpus.blimp_pairs:
        assert good not in train
        assert bad not in train


def test_real_corpus_filters_noisy_artifacts_from_samples():
    corpus = csp.build_sentence_probe_corpus(
        active_vocab_size=1000,
        vocab_size=50257,
        tokenizer="tiktoken",
        tiktoken_encoding="gpt2",
        n_eval_items=32,
    )

    forbidden = {"title", "substep", "substeps"}
    for item in corpus.eval_items:
        words = set(item.good_sentence.split())
        assert words.isdisjoint(forbidden)
    for good, bad in corpus.blimp_pairs:
        assert set(good.split()).isdisjoint(forbidden)
        assert set(bad.split()).isdisjoint(forbidden)


def test_curated_fallback_uses_grammatical_modal_base_forms(monkeypatch):
    monkeypatch.setattr(csp, "_cached_hellaswag_rows", lambda: ())
    monkeypatch.setattr(csp, "_cached_blimp_pairs", lambda: ())
    monkeypatch.setattr(csp, "_benchmark_word_counts", lambda: ())

    corpus = csp.build_sentence_probe_corpus(
        active_vocab_size=80,
        vocab_size=128,
        tokenizer="byte",
        n_eval_items=8,
    )

    assert corpus.source_counts["curated_train"] > 0
    assert "dog can sat" not in corpus.train_sentences
    assert "the dog can sit" in corpus.train_sentences


def test_oracle_scores_perfect_on_choice_order_and_binding(monkeypatch):
    monkeypatch.setattr(csp, "_cached_hellaswag_rows", lambda: ())
    monkeypatch.setattr(csp, "_cached_blimp_pairs", lambda: ())
    monkeypatch.setattr(csp, "_benchmark_word_counts", lambda: ())
    corpus = csp.build_sentence_probe_corpus(
        active_vocab_size=80,
        vocab_size=128,
        tokenizer="byte",
        n_eval_items=8,
    )
    model = _SentenceOracle(corpus, vocab_size=128)

    hella, order, binding = csp.evaluate_controlled_sentence_probe(
        model,
        corpus,
        vocab_size=128,
        device="cpu",
    )

    assert hella == 1.0
    assert order == 1.0
    assert binding == 1.0


def test_public_probe_restores_model_state_and_reports_fields():
    model = _TinySentenceLearner()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = csp.controlled_sentence_probe(
        model,
        active_vocab_size=80,
        n_train_steps=1,
        n_eval_items=4,
        batch_size=4,
        device="cpu",
        tokenizer="byte",
        seed=123,
    )

    payload = result.to_dict()
    assert payload["controlled_sentence_metric_version"] == "controlled_sentence_v2"
    assert payload["controlled_sentence_probe_role"] == "language_shape_diagnostic"
    assert payload["controlled_sentence_status"] == "ok"
    assert payload["controlled_sentence_n_eval_items"] == 4
    assert not model.training
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_public_probe_runs_and_restores_across_step_budgets(steps: int):
    model = _TinySentenceLearner()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = csp.controlled_sentence_probe(
        model,
        active_vocab_size=80,
        n_train_steps=steps,
        n_eval_items=4,
        batch_size=4,
        device="cpu",
        tokenizer="byte",
        seed=123,
    )

    assert result.status == "ok"
    assert result.n_train_steps == steps
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.nano_hellaswag_acc <= 1.0
    assert 0.0 <= result.nano_blimp_order_acc <= 1.0
    assert 0.0 <= result.nano_blimp_binding_acc <= 1.0
    after = model.state_dict()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


def test_public_probe_timeout_reports_partial_steps_and_bounded_metrics():
    model = _TinySentenceLearner()

    result = csp.controlled_sentence_probe(
        model,
        active_vocab_size=80,
        n_train_steps=5,
        n_eval_items=4,
        batch_size=4,
        device="cpu",
        tokenizer="byte",
        seed=123,
        timeout_s=-1.0,
    )

    assert result.status == "timeout"
    assert result.n_train_steps == 0
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.nano_hellaswag_acc <= 1.0
    assert 0.0 <= result.nano_blimp_order_acc <= 1.0
    assert 0.0 <= result.nano_blimp_binding_acc <= 1.0


def test_tuning_harness_parses_step_grid_configs():
    assert cst._parse_config("1000:300") == (1000, 300)
    with pytest.raises(SystemExit):
        parser = cst.argparse.ArgumentParser()
        parser.add_argument("--config", type=cst._parse_config)
        parser.parse_args(["--config", "1000"])


def test_tuning_harness_corpus_summary_includes_auditable_samples():
    summary = cst._corpus_summary(active_vocab_size=1000, n_eval_items=8, seed=42)

    assert summary["probe_role"] == "language_shape_diagnostic"
    assert summary["source_counts"]["babi_eval"] == 8
    assert summary["source_counts"]["train_babi"] > 0
    assert len(summary["vocab_sample"]) > 0
    assert len(summary["train_sample"]) > 0
    assert summary["sentence_shape_sample"] == summary["hellaswag_sample"]
    assert len(summary["hellaswag_sample"]) > 0
    assert len(summary["blimp_pair_sample"]) > 0


def test_tuning_harness_step_curve_summary_reports_deltas_and_auc():
    rows = [
        {
            "controlled_sentence_status": "ok",
            "config_vocab": 256,
            "config_seed": 1,
            "config_steps": 0,
            "controlled_sentence_score": 0.1,
            "controlled_sentence_nano_hellaswag_acc": 0.25,
            "controlled_sentence_nano_blimp_order_acc": 0.4,
            "controlled_sentence_nano_blimp_binding_acc": 0.5,
        },
        {
            "controlled_sentence_status": "ok",
            "config_vocab": 256,
            "config_seed": 1,
            "config_steps": 40,
            "controlled_sentence_score": 0.4,
            "controlled_sentence_nano_hellaswag_acc": 0.5,
            "controlled_sentence_nano_blimp_order_acc": 0.6,
            "controlled_sentence_nano_blimp_binding_acc": 0.7,
        },
    ]

    summary = cst._step_curve_summary(rows)

    assert len(summary) == 1
    item = summary[0]
    assert item["baseline_step"] == 0
    assert item["final_step"] == 40
    assert item["controlled_sentence_score_delta"] == 0.3
    assert item["controlled_sentence_score_auc"] == 0.25
