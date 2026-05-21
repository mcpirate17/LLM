from __future__ import annotations

import gc
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._eval_native import load_eval_native
from ._probe_utils import safe_deepcopy_module
from .utils import clip_grad_norm, make_adamw


def _build_probe_model(model: nn.Module, device: str) -> nn.Module:
    probe_model = safe_deepcopy_module(model)
    probe_model.to(device)
    probe_model.train()
    return probe_model


def _run_retrieval_train_step(
    probe_model: nn.Module,
    opt: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: str,
    make_train_batch: Callable[[int, str], tuple[torch.Tensor, torch.Tensor]],
    query_pos: int,
    vocab_lo: int,
    vocab_hi: int,
) -> bool:
    input_ids, targets = make_train_batch(batch_size, device)
    opt.zero_grad(set_to_none=True)
    logits = probe_model(input_ids)
    pred_logits = logits[:, query_pos, vocab_lo:vocab_hi]
    loss = F.cross_entropy(pred_logits, targets - vocab_lo)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    clip_grad_norm(probe_model.parameters(), 1.0)
    opt.step()
    return True


def _cleanup_probe_model(probe_model: nn.Module, device: str) -> None:
    del probe_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def eval_restricted_last_token_accuracy(
    model: nn.Module,
    eval_ids: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    batch_size: int,
    vocab_lo: int,
    vocab_hi: int,
    query_pos: int | None = None,
) -> float:
    """Evaluate exact-match accuracy at one position over a restricted vocab."""

    model.eval()
    total = eval_ids.shape[0]
    answer_pos = eval_ids.shape[1] - 1 if query_pos is None else int(query_pos)
    correct = 0.0
    native = None
    native_attempted = False

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            inp = eval_ids[start:end]
            tgt = eval_targets[start:end]
            logits = model(inp)
            if not native_attempted:
                native_attempted = True
                try:
                    native = load_eval_native()
                except Exception:
                    native = None
            if native is not None:
                correct += float(
                    native.restricted_last_token_accuracy_native(
                        logits,
                        tgt,
                        int(vocab_lo),
                        int(vocab_hi),
                        int(answer_pos),
                    )
                ) * float(tgt.shape[0])
                continue
            pred_logits = logits[:, answer_pos, vocab_lo:vocab_hi]
            preds = pred_logits.argmax(dim=-1) + vocab_lo
            correct += float((preds == tgt).sum().item())

    return float(correct) / max(total, 1)


def run_retrieval_probe_config(
    model: nn.Module,
    *,
    n_train_steps: int,
    eval_ids: torch.Tensor,
    eval_targets: torch.Tensor,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    make_train_batch: Callable[[int, str], tuple[torch.Tensor, torch.Tensor]],
    query_pos: int,
    vocab_lo: int,
    vocab_hi: int,
) -> tuple[float, bool]:
    """Train a deepcopy on one retrieval config, then evaluate accuracy."""
    probe_model = _build_probe_model(model, device)
    opt = make_adamw(probe_model.parameters(), lr=lr)
    timed_out = False

    try:
        for _step in range(1, n_train_steps + 1):
            if time.perf_counter() > deadline:
                timed_out = True
                break

            if not _run_retrieval_train_step(
                probe_model,
                opt,
                batch_size=batch_size,
                device=device,
                make_train_batch=make_train_batch,
                query_pos=query_pos,
                vocab_lo=vocab_lo,
                vocab_hi=vocab_hi,
            ):
                break

        acc = eval_restricted_last_token_accuracy(
            probe_model,
            eval_ids,
            eval_targets,
            batch_size=batch_size,
            vocab_lo=vocab_lo,
            vocab_hi=vocab_hi,
            query_pos=query_pos,
        )
    finally:
        _cleanup_probe_model(probe_model, device)

    return acc, timed_out


def run_retrieval_probe_learning_curve(
    model: nn.Module,
    *,
    n_train_steps: int,
    eval_every: int,
    eval_ids: torch.Tensor,
    eval_targets: torch.Tensor,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    make_train_batch: Callable[[int, str], tuple[torch.Tensor, torch.Tensor]],
    query_pos: int,
    vocab_lo: int,
    vocab_hi: int,
) -> tuple[list[tuple[int, float]], int, bool, str]:
    """Train a deepcopy while collecting a periodic accuracy curve."""

    probe_model = _build_probe_model(model, device)
    opt = make_adamw(probe_model.parameters(), lr=lr)
    timed_out = False
    status = "ok"
    steps_trained = 0
    learning_curve = [
        (
            0,
            round(
                eval_restricted_last_token_accuracy(
                    probe_model,
                    eval_ids,
                    eval_targets,
                    batch_size=batch_size,
                    vocab_lo=vocab_lo,
                    vocab_hi=vocab_hi,
                    query_pos=query_pos,
                ),
                4,
            ),
        )
    ]

    try:
        probe_model.train()
        for step in range(1, n_train_steps + 1):
            if time.perf_counter() > deadline:
                timed_out = True
                status = "timeout"
                break
            if not _run_retrieval_train_step(
                probe_model,
                opt,
                batch_size=batch_size,
                device=device,
                make_train_batch=make_train_batch,
                query_pos=query_pos,
                vocab_lo=vocab_lo,
                vocab_hi=vocab_hi,
            ):
                status = "diverged"
                break
            steps_trained = step
            if step % eval_every == 0 or step == n_train_steps:
                acc = eval_restricted_last_token_accuracy(
                    probe_model,
                    eval_ids,
                    eval_targets,
                    batch_size=batch_size,
                    vocab_lo=vocab_lo,
                    vocab_hi=vocab_hi,
                    query_pos=query_pos,
                )
                learning_curve.append((step, round(acc, 4)))
                probe_model.train()
    finally:
        _cleanup_probe_model(probe_model, device)

    return learning_curve, steps_trained, timed_out, status
