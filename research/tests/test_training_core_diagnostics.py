from __future__ import annotations

import torch

from research.eval.training_core import run_training_loop
from research.eval.corpus_pipeline import (
    _batch_cache,
    _token_cache,
    prepare_text_split_batches,
)


def test_run_training_loop_records_grad_and_lr_telemetry():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 4)
    telemetry: dict = {}

    def compute_loss(step: int) -> torch.Tensor:
        pred = model(inputs[step : step + 1])
        return torch.nn.functional.mse_loss(pred, targets[step : step + 1])

    result = run_training_loop(
        model.parameters(),
        compute_loss,
        n_steps=3,
        lr=1e-2,
        clip_grad=0.05,
        warmup_steps=2,
        train_telemetry=telemetry,
        parameter_names=[name for name, _ in model.named_parameters()],
    )

    assert result.diverged is False
    assert result.telemetry is telemetry
    assert telemetry["summary"]["steps_completed"] == 3
    assert telemetry["summary"]["max_pre_clip_grad_norm"] >= 0.0
    assert telemetry["summary"]["max_post_clip_grad_norm"] <= 0.050001
    assert len(telemetry["steps"]) == 3
    assert telemetry["steps"][0]["lr_expected"] == [0.005]
    assert telemetry["steps"][1]["lr_expected"] == [0.01]
    assert "weight" in telemetry["steps"][0]["pre_clip_layer_norms"]
    assert "bias" in telemetry["steps"][0]["post_clip_layer_norms"]


def test_run_training_loop_warmup_preserves_optimizer_group_ratios():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "lr": 0.1},
            {"params": [second], "lr": 0.01},
        ]
    )
    telemetry: dict = {}

    def compute_loss(_step: int) -> torch.Tensor:
        return (first.square() + second.square()).sum()

    run_training_loop(
        [first, second],
        compute_loss,
        n_steps=1,
        optimizer=optimizer,
        lr=1.0,
        warmup_steps=4,
        clip_grad=0.0,
        train_telemetry=telemetry,
        parameter_names=["first", "second"],
    )

    assert telemetry["steps"][0]["lr_expected"] == [0.025, 0.0025]


def test_prepare_text_split_batches_reports_token_counts_on_cache_hit(tmp_path):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    text = "hello world test data " * 100
    train_path.write_text(text, encoding="utf-8")
    val_path.write_text(text, encoding="utf-8")

    _batch_cache.clear()
    _token_cache.clear()

    kwargs = dict(
        namespace="cache-hit-counts",
        train_path=train_path,
        val_path=val_path,
        vocab_size=256,
        seq_len=8,
        train_batch_size=2,
        eval_batch_size=2,
        n_train_batches=4,
        n_eval_batches=2,
        device="cpu",
    )

    _, _, train_tokens_first, val_tokens_first = prepare_text_split_batches(**kwargs)
    _, _, train_tokens_second, val_tokens_second = prepare_text_split_batches(**kwargs)

    assert train_tokens_first > 0
    assert val_tokens_first > 0
    assert train_tokens_second == train_tokens_first
    assert val_tokens_second == val_tokens_first
