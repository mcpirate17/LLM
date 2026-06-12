"""Wikitext + BLiMP evaluation for fab lanes wrapped in a TinyLM.

This is the Level-1 handoff: stay inside fab, reuse research/'s wikitext
data preparation and BLiMP scoring, but run the training loop locally so
we don't have to adopt research/'s functional-training assumptions.

The proven template is a pre-norm Transformer: ``Embedding -> (norm ->
mixer -> +x -> norm -> FFN -> +x) * N -> norm -> LMHead``. The mixer is
the only knob that differs between a fab candidate and the baselines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from research.defaults import VOCAB_SIZE
from research.eval.blimp_eval import BLiMPResult, evaluate_blimp
from research.eval.wikitext_eval import _prepare_batches

from .tiny_lm import TinyLM, count_trainable_params
from .training_probe import build_tiny_lm

logger = logging.getLogger(__name__)


# Wikitext-103 BPE prep — small budget so a CPU run completes in minutes.
_WIKITEXT_VARIANT = "wikitext-103-raw-v1"
_DEFAULT_TRAIN_BATCHES = 64
_DEFAULT_EVAL_BATCHES = 16
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_SEQ_LEN = 128
_DEFAULT_MAX_CHARS_TRAIN = 1_000_000
_DEFAULT_MAX_CHARS_VAL = 100_000


@dataclass(frozen=True, slots=True)
class WikitextTrainTrace:
    initial_loss: float
    final_loss: float
    pre_train_ppl: float
    post_train_ppl: float
    n_steps: int
    converged: bool


@dataclass(frozen=True, slots=True)
class LMEvalResult:
    mixer_label: str
    n_params: int
    wikitext: WikitextTrainTrace
    blimp_overall_accuracy: float
    blimp_by_subtask: dict[str, float]
    blimp_status: str


def _build_lm(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int,
    n_blocks: int,
    vocab_size: int,
    max_seq_len: int,
) -> TinyLM:
    return build_tiny_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=max_seq_len,
        use_position_embedding=True,
        use_ffn=True,  # Pre-norm Transformer requires FFN to be a real LM.
        ffn_mult=4,
    )


def _causal_lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a one-position causal shift."""
    return nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
    )


def _eval_ppl(model: TinyLM, batches: list[torch.Tensor]) -> float:
    """Mean cross-entropy over batches -> perplexity."""
    if not batches:
        return float("nan")
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in batches:
            logits = model(batch)
            loss = _causal_lm_loss(logits, batch)
            total_loss += float(loss.item())
            n += 1
    if n == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_loss / n)).item())


def train_on_wikitext(
    model: TinyLM,
    *,
    n_steps: int = 500,
    lr: float = 3e-4,
    seq_len: int = _DEFAULT_SEQ_LEN,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    n_train_batches: int = _DEFAULT_TRAIN_BATCHES,
    n_eval_batches: int = _DEFAULT_EVAL_BATCHES,
    max_chars_train: int = _DEFAULT_MAX_CHARS_TRAIN,
    max_chars_val: int = _DEFAULT_MAX_CHARS_VAL,
    device: str = "cpu",
) -> WikitextTrainTrace:
    """Train the LM on wikitext-103 BPE batches and return pre/post PPL.

    Uses research/'s ``_prepare_batches`` for the data path (cached BPE
    tokenization, same vocab_size as research/'s eval tier). The training
    loop itself is a plain Adam loop — no research/ functional-training
    machinery.
    """
    train_batches, val_batches, _, _ = _prepare_batches(
        _WIKITEXT_VARIANT,
        model.config.vocab_size,
        seq_len,
        batch_size,
        batch_size,
        n_train_batches,
        n_eval_batches,
        max_chars_train,
        max_chars_val,
        device,
    )
    if not train_batches or not val_batches:
        return WikitextTrainTrace(
            initial_loss=float("nan"),
            final_loss=float("nan"),
            pre_train_ppl=float("nan"),
            post_train_ppl=float("nan"),
            n_steps=0,
            converged=False,
        )
    pre_ppl = _eval_ppl(model, val_batches)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = len(train_batches)
    initial_loss = float("nan")
    final_loss = float("nan")
    converged = True
    try:
        model.train()
        for step in range(n_steps):
            batch = train_batches[step % n_train]
            logits = model(batch)
            loss = _causal_lm_loss(logits, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step == 0:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())
    except Exception:  # noqa: BLE001 - one bad lane must not abort the cohort
        logger.warning(
            "wikitext training failed; scoring as non-converged", exc_info=True
        )
        converged = False
    post_ppl = _eval_ppl(model, val_batches) if converged else float("nan")
    return WikitextTrainTrace(
        initial_loss=initial_loss,
        final_loss=final_loss,
        pre_train_ppl=pre_ppl,
        post_train_ppl=post_ppl,
        n_steps=n_steps,
        converged=converged,
    )


def evaluate_lm(
    lane_factory: Callable[[int], nn.Module],
    *,
    mixer_label: str,
    dim: int = 64,
    n_blocks: int = 2,
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: int = 128,
    n_train_steps: int = 500,
    learning_rate: float = 3e-4,
    blimp_n_per_subtask: int = 50,
    blimp_max_seq_len: int = 256,
    device: str = "cpu",
    seed: int = 0,
) -> LMEvalResult:
    """Build a TinyLM(lane), train on wikitext-103, then run BLiMP.

    Identical hyperparameters across candidate and baselines — the only
    moving part is ``lane_factory``.
    """
    torch.manual_seed(seed)
    model = _build_lm(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    model = model.to(device)
    wikitext = train_on_wikitext(
        model,
        n_steps=n_train_steps,
        lr=learning_rate,
        seq_len=max_seq_len,
        device=device,
    )
    blimp: BLiMPResult = evaluate_blimp(
        model,
        vocab_size=vocab_size,
        device=device,
        n_per_subtask=blimp_n_per_subtask,
        max_seq_len=blimp_max_seq_len,
    )
    return LMEvalResult(
        mixer_label=mixer_label,
        n_params=count_trainable_params(model),
        wikitext=wikitext,
        blimp_overall_accuracy=float(blimp.overall_accuracy or 0.0),
        blimp_by_subtask={
            k: float(v) for k, v in (blimp.subtask_accuracies or {}).items()
        },
        blimp_status=str(blimp.status or "ok"),
    )
