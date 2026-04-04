"""Focused tests for screening-time WikiText evaluation.

Tests: non-invasive eval, deterministic caching, metadata/version fields,
graceful failure, and score formula correctness.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from research.eval.wikitext_eval import (
    screening_wikitext_eval,
    screening_wikitext_payload,
    wikitext_score_from_ppl,
    _batch_cache,
    _get_cached_batches,
    _put_cached_batches,
)


# ── Tiny model fixture ──────────────────────────────────────────────────


class _TinyLM(nn.Module):
    """Minimal language model for testing."""

    __slots__ = ()

    def __init__(self, vocab_size: int = 256, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return _TinyLM(vocab_size=256, dim=32)


# ── wikitext_score_from_ppl ─────────────────────────────────────────────


class TestWikitextScoreFromPpl:
    def test_perfect_ppl_returns_1(self):
        # ppl=1 → log(32000/1)/log(32000) = 1.0
        assert wikitext_score_from_ppl(1.0) == 1.0

    def test_random_ppl_returns_0(self):
        # ppl=vocab_size → log(1)/log(vocab) = 0.0
        assert wikitext_score_from_ppl(32000) == 0.0

    def test_worse_than_random_clamped(self):
        assert wikitext_score_from_ppl(100_000) == 0.0

    def test_none_input(self):
        assert wikitext_score_from_ppl(None) is None

    def test_zero_ppl(self):
        assert wikitext_score_from_ppl(0.0) is None

    def test_negative_ppl(self):
        assert wikitext_score_from_ppl(-5.0) is None

    def test_gpt2_range(self):
        # GPT-2 ppl ~29.4 → score ~0.67
        score = wikitext_score_from_ppl(29.4)
        assert 0.6 < score < 0.75


# ── Non-invasive eval ────────────────────────────────────────────────────


class TestNonInvasiveEval:
    """Live model parameters must be unchanged after screening WikiText eval."""

    def test_model_params_unchanged(self, tiny_model):
        """Core invariant: screening eval must not mutate the live model."""
        params_before = {name: p.clone() for name, p in tiny_model.named_parameters()}
        training_before = tiny_model.training

        with patch("research.eval.wikitext_eval._download_wikitext") as mock_dl:
            # Create fake text files
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as td:
                train_p = os.path.join(td, "train.txt")
                val_p = os.path.join(td, "val.txt")
                # Write enough text to tokenize
                with open(train_p, "w") as f:
                    f.write("hello world test data " * 500)
                with open(val_p, "w") as f:
                    f.write("validation text here " * 200)
                from pathlib import Path

                mock_dl.return_value = (Path(train_p), Path(val_p))

                screening_wikitext_eval(
                    tiny_model,
                    vocab_size=256,
                    device="cpu",
                    seq_len=16,
                    n_train_steps=5,
                    n_train_batches=2,
                    n_eval_batches=2,
                    batch_size=2,
                )

        # Verify parameters unchanged
        for name, p in tiny_model.named_parameters():
            assert torch.equal(p, params_before[name]), (
                f"Parameter {name} was mutated by screening eval"
            )
        # Verify training mode restored
        assert tiny_model.training == training_before

    def test_clone_failure_graceful(self, tiny_model):
        """If stateless state capture fails, return error status without crashing."""
        with patch(
            "research.eval.wikitext_eval.clone_module_state",
            side_effect=RuntimeError("clone boom"),
        ):
            with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
                mock_pb.return_value = (
                    [torch.zeros(2, 16)],
                    [torch.zeros(2, 16)],
                    100,
                    50,
                )
                result = screening_wikitext_eval(
                    tiny_model,
                    vocab_size=256,
                    device="cpu",
                )

        assert result["screening_wikitext_status"] == "clone_failed"
        assert "clone boom" in result.get("error", "")


# ── Metadata and version fields ──────────────────────────────────────────


class TestMetadataFields:
    def test_required_fields_present(self, tiny_model):
        """All metadata fields must be present regardless of eval outcome."""
        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            # Return empty batches to trigger insufficient_tokens
            mock_pb.return_value = (None, None, 10, 5)
            result = screening_wikitext_eval(
                tiny_model,
                vocab_size=256,
                device="cpu",
            )

        assert "screening_wikitext_metric_version" in result
        assert "screening_wikitext_status" in result
        assert "screening_wikitext_budget" in result
        assert "wikitext_perplexity" in result
        assert "wikitext_score" in result
        assert "elapsed_ms" in result
        assert result["screening_wikitext_metric_version"] == "screening_wikitext_v1"

    def test_budget_fields(self, tiny_model):
        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            mock_pb.return_value = (None, None, 10, 5)
            result = screening_wikitext_eval(
                tiny_model,
                vocab_size=256,
                device="cpu",
            )

        budget = result["screening_wikitext_budget"]
        for key in (
            "n_train_steps",
            "n_train_batches",
            "n_eval_batches",
            "batch_size",
            "seq_len",
            "max_chars_train",
            "max_chars_val",
        ):
            assert key in budget, f"Missing budget field: {key}"

    def test_payload_builder(self):
        payload = screening_wikitext_payload(
            {
                "screening_wikitext_metric_version": "screening_wikitext_v1",
                "screening_wikitext_status": "ok",
                "variant": "wikitext-2-raw-v1",
                "elapsed_ms": 123.4,
                "screening_wikitext_budget": {"n_train_steps": 50},
                "wikitext_perplexity": 42.0,
                "wikitext_pre_perplexity": 210.0,
                "wikitext_ppl_improvement": 0.2,
                "wikitext_score": 0.61,
            }
        )

        assert payload is not None
        bench = payload["screening_wikitext"]
        assert bench["benchmark_family"] == "real_token_screening"
        assert bench["metric_version"] == "screening_wikitext_v1"
        assert bench["status"] == "ok"
        assert bench["budget"]["n_train_steps"] == 50
        assert bench["metrics"]["wikitext_perplexity"] == 42.0
        assert bench["metrics"]["wikitext_score"] == 0.61


# ── Graceful failure ─────────────────────────────────────────────────────


class TestGracefulFailure:
    def test_data_download_failure(self, tiny_model):
        with patch(
            "research.eval.wikitext_eval._prepare_batches",
            side_effect=OSError("no network"),
        ):
            result = screening_wikitext_eval(
                tiny_model,
                vocab_size=256,
                device="cpu",
            )
        assert result["screening_wikitext_status"] == "data_failed"
        assert result["wikitext_perplexity"] is None

    def test_insufficient_tokens(self, tiny_model):
        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            mock_pb.return_value = (None, None, 5, 3)
            result = screening_wikitext_eval(
                tiny_model,
                vocab_size=256,
                device="cpu",
            )
        assert result["screening_wikitext_status"] == "insufficient_tokens"
        assert result["wikitext_perplexity"] is None

    def test_empty_batches(self, tiny_model):
        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            mock_pb.return_value = ([], [], 100, 50)
            result = screening_wikitext_eval(
                tiny_model,
                vocab_size=256,
                device="cpu",
            )
        assert result["screening_wikitext_status"] == "insufficient_tokens"


# ── Batch cache ──────────────────────────────────────────────────────────


class TestBatchCache:
    def setup_method(self):
        _batch_cache.clear()

    def test_put_and_get(self):
        batches = [torch.randn(2, 16)]
        _put_cached_batches(
            "wikitext-2-raw-v1", 256, 16, 4, 8, 100_000, "train", 42, batches
        )
        got = _get_cached_batches(
            "wikitext-2-raw-v1", 256, 16, 4, 8, 100_000, "cpu", "train", 42
        )
        assert got is not None
        assert len(got) == 1
        assert torch.equal(got[0], batches[0])

    def test_cache_miss(self):
        got = _get_cached_batches(
            "wikitext-2-raw-v1", 256, 16, 4, 8, 100_000, "cpu", "train", 42
        )
        assert got is None

    def test_different_seed_different_key(self):
        batches = [torch.randn(2, 16)]
        _put_cached_batches(
            "wikitext-2-raw-v1", 256, 16, 4, 8, 100_000, "train", 42, batches
        )
        got = _get_cached_batches(
            "wikitext-2-raw-v1", 256, 16, 4, 8, 100_000, "cpu", "train", 99
        )
        assert got is None

    def teardown_method(self):
        _batch_cache.clear()
