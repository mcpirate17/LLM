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
)
from research.eval import corpus_pipeline


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

    def test_cuda_probe_bypasses_native_dispatch(self, tiny_model):
        calls = {"enter": 0, "exit": 0, "device": None}

        class _ProbeCtx:
            def __enter__(self):
                calls["enter"] += 1

            def __exit__(self, exc_type, exc, tb):
                calls["exit"] += 1
                return False

        def _fake_disable(model, *, device):
            del model
            calls["device"] = device
            return _ProbeCtx()

        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            mock_pb.return_value = (
                [torch.zeros(2, 16, dtype=torch.long)],
                [torch.zeros(2, 16, dtype=torch.long)],
                100,
                50,
            )
            with patch(
                "research.eval.wikitext_eval.disable_native_probe_dispatch",
                side_effect=_fake_disable,
            ):
                screening_wikitext_eval(
                    tiny_model,
                    vocab_size=256,
                    device="cuda",
                    seq_len=16,
                    n_train_steps=1,
                    n_train_batches=1,
                    n_eval_batches=1,
                    batch_size=2,
                )

        assert calls["device"] == "cuda"
        assert calls["enter"] == 1
        assert calls["exit"] == 1


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
                "screening_wikitext_degraded": True,
                "screening_wikitext_degraded_reasons": ["persistent_heavy_clipping"],
                "screening_wikitext_clipped_steps": 4,
                "screening_wikitext_clip_fraction": 0.8,
                "screening_wikitext_max_lr_delta": 0.0,
                "screening_wikitext_nonfinite_grad_steps": 0,
                "max_grad_norm": 12.0,
                "mean_grad_norm": 9.0,
                "grad_norm_std": 1.5,
                "final_lr": 0.0003,
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
        assert bench["diagnostics"]["screening_wikitext_degraded"] is True
        assert bench["diagnostics"]["max_grad_norm"] == 12.0

    def test_screening_eval_exposes_grad_and_lr_diagnostics(self, tiny_model):
        fake_telemetry = {
            "steps": [
                {
                    "step": 1,
                    "loss": 5.0,
                    "lr_expected": [1.5e-4],
                    "lr_actual_before_step": [1.5e-4],
                    "lr_actual_after_scheduler": [1.5e-4],
                    "pre_clip_total_grad_norm": 2.0,
                    "post_clip_total_grad_norm": 1.0,
                    "pre_clip_layer_norms": {"embed.weight": 2.0},
                    "post_clip_layer_norms": {"embed.weight": 1.0},
                    "pre_clip_max_layer": "embed.weight",
                    "post_clip_max_layer": "embed.weight",
                    "pre_clip_max_layer_norm": 2.0,
                    "post_clip_max_layer_norm": 1.0,
                    "clipped": False,
                    "has_nonfinite_grad": False,
                },
                {
                    "step": 2,
                    "loss": 4.0,
                    "lr_expected": [3e-4],
                    "lr_actual_before_step": [3e-4],
                    "lr_actual_after_scheduler": [3e-4],
                    "pre_clip_total_grad_norm": 20.0,
                    "post_clip_total_grad_norm": 1.0,
                    "pre_clip_layer_norms": {"embed.weight": 20.0},
                    "post_clip_layer_norms": {"embed.weight": 1.0},
                    "pre_clip_max_layer": "embed.weight",
                    "post_clip_max_layer": "embed.weight",
                    "pre_clip_max_layer_norm": 20.0,
                    "post_clip_max_layer_norm": 1.0,
                    "clipped": True,
                    "has_nonfinite_grad": False,
                },
            ]
        }

        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            mock_pb.return_value = (
                [torch.zeros(2, 16, dtype=torch.long)],
                [torch.zeros(2, 16, dtype=torch.long)],
                100,
                50,
            )
            with patch(
                "research.eval.wikitext_eval._screening_train_eval",
                return_value=(120.0, 60.0, 3.5, {1: 5.0, 2: 4.0}, fake_telemetry),
            ):
                result = screening_wikitext_eval(
                    tiny_model,
                    vocab_size=256,
                    device="cpu",
                    seq_len=16,
                    n_train_steps=2,
                    n_train_batches=1,
                    n_eval_batches=1,
                    batch_size=2,
                )

        assert result["screening_wikitext_status"] == "ok"
        assert result["max_grad_norm"] == 20.0
        assert result["mean_grad_norm"] == 11.0
        assert result["final_lr"] == 0.0003
        assert result["screening_wikitext_degraded"] is False
        assert result["training_curve"] == [
            {"step": 1, "loss": 5.0, "grad_norm": 2.0},
            {"step": 2, "loss": 4.0, "grad_norm": 20.0},
        ]

    def test_screening_eval_flags_persistent_heavy_clipping(self, tiny_model):
        clipped_steps = []
        for step in range(1, 5):
            clipped_steps.append(
                {
                    "step": step,
                    "loss": 5.0 - step,
                    "lr_expected": [3e-4],
                    "lr_actual_before_step": [3e-4],
                    "lr_actual_after_scheduler": [3e-4],
                    "pre_clip_total_grad_norm": 15.0,
                    "post_clip_total_grad_norm": 1.0,
                    "pre_clip_layer_norms": {"embed.weight": 15.0},
                    "post_clip_layer_norms": {"embed.weight": 1.0},
                    "pre_clip_max_layer": "embed.weight",
                    "post_clip_max_layer": "embed.weight",
                    "pre_clip_max_layer_norm": 15.0,
                    "post_clip_max_layer_norm": 1.0,
                    "clipped": True,
                    "has_nonfinite_grad": False,
                }
            )

        with patch("research.eval.wikitext_eval._prepare_batches") as mock_pb:
            mock_pb.return_value = (
                [torch.zeros(2, 16, dtype=torch.long)],
                [torch.zeros(2, 16, dtype=torch.long)],
                100,
                50,
            )
            with patch(
                "research.eval.wikitext_eval._screening_train_eval",
                return_value=(
                    120.0,
                    80.0,
                    4.0,
                    {1: 4.5, 4: 3.0},
                    {"steps": clipped_steps},
                ),
            ):
                result = screening_wikitext_eval(
                    tiny_model,
                    vocab_size=256,
                    device="cpu",
                    seq_len=16,
                    n_train_steps=4,
                    n_train_batches=1,
                    n_eval_batches=1,
                    batch_size=2,
                )

        assert result["screening_wikitext_degraded"] is True
        assert (
            "persistent_heavy_clipping" in result["screening_wikitext_degraded_reasons"]
        )


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
        corpus_pipeline._batch_cache.clear()

    def test_put_and_get(self):
        batches = [torch.randn(2, 16)]
        cache_key = (
            "wikitext:wikitext-2-raw-v1",
            "/tmp/train.txt",
            1,
            256,
            16,
            4,
            8,
            "train",
            42,
        )
        corpus_pipeline._put_cached_batches(cache_key, batches)
        got = corpus_pipeline._get_cached_batches(cache_key, "cpu")
        assert got is not None
        assert len(got) == 1
        assert torch.equal(got[0], batches[0])

    def test_cache_miss(self):
        got = corpus_pipeline._get_cached_batches(("missing",), "cpu")
        assert got is None

    def test_different_seed_different_key(self):
        batches = [torch.randn(2, 16)]
        key_seed_42 = (
            "wikitext:wikitext-2-raw-v1",
            "/tmp/train.txt",
            1,
            256,
            16,
            4,
            8,
            "train",
            42,
        )
        key_seed_99 = key_seed_42[:-1] + (99,)
        corpus_pipeline._put_cached_batches(key_seed_42, batches)
        got = corpus_pipeline._get_cached_batches(key_seed_99, "cpu")
        assert got is None

    def teardown_method(self):
        corpus_pipeline._batch_cache.clear()
