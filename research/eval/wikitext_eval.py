"""WikiText perplexity evaluation for architecture robustness.

Downloads WikiText-2 (or WikiText-103) from HuggingFace, micro-trains
the candidate model on the train split, then evaluates perplexity on the
validation split. This tests whether an architecture can learn real
linguistic patterns, not just synthetic data.

Uses the existing CorpusTokenBatcher infrastructure via a cached text file.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn

from .corpus_pipeline import (
    TextSplitSpec,
    cache_hf_text_splits,
    prepare_text_split_batches,
)
from .training_core import run_training_loop
from .utils import language_model_loss, make_adamw, micro_train_loop, compute_perplexity
from .stateless_training import (
    clone_module_state,
    functional_compute_perplexity,
    functional_micro_train_loop,
)

logger = logging.getLogger(__name__)

_WIKITEXT_CACHE_DIR = Path.home() / ".cache" / "aria" / "wikitext"

# Subset sizes (chars) — wiki103 for all non-screening evals
_DEFAULT_MAX_CHARS_TRAIN = 20_000_000  # 20MB of WikiText-103 for micro-training
_DEFAULT_MAX_CHARS_VAL = 200_000  # 200KB for validation perplexity

# Screening defaults — smaller budget for fast turnaround
_SCREENING_MAX_CHARS_TRAIN = 100_000
_SCREENING_MAX_CHARS_VAL = 20_000
_SCREENING_N_TRAIN_STEPS = 50
_SCREENING_N_TRAIN_BATCHES = 16
_SCREENING_N_EVAL_BATCHES = 4
_SCREENING_BATCH_SIZE = 4
_SCREENING_METRIC_VERSION = "screening_wikitext_v1"


def screening_wikitext_payload(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a normalized screening benchmark payload for persistence."""
    status = result.get("screening_wikitext_status")
    metric_version = result.get("screening_wikitext_metric_version")
    if not status and not metric_version:
        return None
    payload = {
        "screening_wikitext": {
            "benchmark_family": "real_token_screening",
            "metric_version": metric_version,
            "status": status,
            "variant": result.get("variant"),
            "elapsed_ms": result.get("elapsed_ms"),
            "budget": result.get("screening_wikitext_budget"),
            "metrics": {
                "wikitext_perplexity": result.get("wikitext_perplexity"),
                "wikitext_pre_perplexity": result.get("wikitext_pre_perplexity"),
                "wikitext_ppl_improvement": result.get("wikitext_ppl_improvement"),
                "wikitext_score": result.get("wikitext_score"),
            },
        }
    }
    return payload


def trajectory_wikitext_payload(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a normalized trajectory benchmark payload for persistence."""
    checkpoints = result.get("checkpoints")
    protocol = result.get("protocol")
    if not checkpoints and not protocol:
        return None
    return {
        "wikitext_trajectory": {
            "benchmark_family": "real_token_trajectory",
            "protocol": protocol,
            "variant": result.get("variant"),
            "elapsed_ms": result.get("elapsed_ms"),
            "checkpoint_steps": result.get("checkpoint_steps"),
            "total_steps": result.get("total_steps"),
            "improvement_ratio": result.get("improvement_ratio"),
            "peak_ppl": result.get("peak_ppl"),
            "peak_step": result.get("peak_step"),
            "steps_to_divergence": result.get("steps_to_divergence"),
            "checkpoints": checkpoints,
        }
    }


def _download_wikitext(
    variant: str = "wikitext-103-raw-v1",
    max_chars_train: int = _DEFAULT_MAX_CHARS_TRAIN,
    max_chars_val: int = _DEFAULT_MAX_CHARS_VAL,
) -> Tuple[Path, Path]:
    """Download and cache WikiText train/val splits as plain text files.

    Returns (train_path, val_path) pointing to cached .txt files.
    """
    cache_dir = _WIKITEXT_CACHE_DIR / variant
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train.txt"
    val_path = cache_dir / "validation.txt"

    if train_path.exists() and val_path.exists():
        return train_path, val_path

    logger.info("Downloading WikiText variant=%s ...", variant)
    paths = cache_hf_text_splits(
        cache_dir=cache_dir,
        dataset_name="wikitext",
        config_name=variant,
        split_specs=(
            TextSplitSpec("train", "train.txt", max_chars_train),
            TextSplitSpec("validation", "validation.txt", max_chars_val),
        ),
        trust_remote_code=True,
    )

    logger.info("WikiText cached at %s", cache_dir)
    return paths["train"], paths["validation"]


def _prepare_batches(
    variant: str,
    vocab_size: int,
    seq_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    n_train_batches: int,
    n_eval_batches: int,
    max_chars_train: int,
    max_chars_val: int,
    device: str,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], int, int]:
    """Prepare train/val batches with caching. Returns (train, val, n_train_tok, n_val_tok)."""
    train_path, val_path = _download_wikitext(variant, max_chars_train, max_chars_val)
    return prepare_text_split_batches(
        namespace=f"wikitext:{variant}",
        train_path=train_path,
        val_path=val_path,
        vocab_size=vocab_size,
        seq_len=seq_len,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        n_train_batches=n_train_batches,
        n_eval_batches=n_eval_batches,
        device=device,
    )


def _has_usable_batches(
    train_batches: Optional[List[torch.Tensor]],
    val_batches: Optional[List[torch.Tensor]],
) -> bool:
    return bool(train_batches) and bool(val_batches)


def _finalize_ppl_result(
    *,
    pre_ppl: Optional[float],
    post_ppl: Optional[float],
    train_final_loss: float,
    vocab_size: int,
    variant: str,
    n_train_steps: int,
    seq_len: int,
    elapsed_ms: float,
) -> Dict[str, Any]:
    ppl_improvement = None
    if pre_ppl is not None and post_ppl is not None and pre_ppl > 0:
        ppl_improvement = round(post_ppl / pre_ppl, 4)
    return {
        "wikitext_perplexity": round(post_ppl, 2) if post_ppl is not None else None,
        "wikitext_pre_perplexity": round(pre_ppl, 2) if pre_ppl is not None else None,
        "wikitext_score": wikitext_score_from_ppl(post_ppl, vocab_size),
        "wikitext_ppl_improvement": ppl_improvement,
        "train_final_loss": round(train_final_loss, 6),
        "variant": variant,
        "n_train_steps": n_train_steps,
        "seq_len": seq_len,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ── Score helper ─────────────────────────────────────────────────────────


def _make_scheduled_loss_fn(
    model: nn.Module,
    batches: List[torch.Tensor],
    vocab_size: int,
    start_step: int,
    n_steps: int,
):
    last_loss = torch.tensor(float("nan"))
    n_batches = len(batches)
    if start_step + n_steps <= n_batches:
        segment = batches[start_step : start_step + n_steps]

        def compute_loss(local_step: int) -> torch.Tensor:
            nonlocal last_loss
            batch = segment[local_step]
            last_loss = language_model_loss(model(batch), batch, vocab_size)
            return last_loss

    else:

        def compute_loss(local_step: int) -> torch.Tensor:
            nonlocal last_loss
            batch = batches[(start_step + local_step) % n_batches]
            last_loss = language_model_loss(model(batch), batch, vocab_size)
            return last_loss

    return compute_loss, lambda: last_loss


def wikitext_score_from_ppl(
    ppl: Optional[float], vocab_size: int = 32000
) -> Optional[float]:
    """log(vocab/ppl) / log(vocab) — 1.0 for perfect, 0.0 for random."""
    if ppl is None or ppl <= 0:
        return None
    return round(
        max(0.0, min(1.0, math.log(vocab_size / ppl) / math.log(vocab_size))),
        4,
    )


# ── Screening evaluation (non-invasive) ─────────────────────────────────


def screening_wikitext_eval(
    model: nn.Module,
    vocab_size: int,
    device: str,
    seq_len: int = 128,
    n_train_steps: int = _SCREENING_N_TRAIN_STEPS,
    n_train_batches: int = _SCREENING_N_TRAIN_BATCHES,
    n_eval_batches: int = _SCREENING_N_EVAL_BATCHES,
    batch_size: int = _SCREENING_BATCH_SIZE,
    lr: float = 3e-4,
    variant: str = "wikitext-103-raw-v1",
) -> Dict[str, Any]:
    """Non-invasive WikiText eval for screening — ~2-5s on GPU.

    Clones model weights before micro-training so the live model used by
    the screening/investigation handoff is never mutated.  Returns the same
    top-level keys as ``evaluate_wikitext_perplexity`` plus version and
    status metadata.
    """
    meta: Dict[str, Any] = {
        "screening_wikitext_metric_version": _SCREENING_METRIC_VERSION,
        "screening_wikitext_status": "skipped",
        "screening_wikitext_budget": {
            "n_train_steps": n_train_steps,
            "n_train_batches": n_train_batches,
            "n_eval_batches": n_eval_batches,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "max_chars_train": _SCREENING_MAX_CHARS_TRAIN,
            "max_chars_val": _SCREENING_MAX_CHARS_VAL,
        },
        "wikitext_perplexity": None,
        "wikitext_score": None,
    }
    t0 = time.perf_counter()

    # Prepare batches (cached across candidates within one process)
    try:
        train_batches, val_batches, n_train_tok, n_val_tok = _prepare_batches(
            variant,
            vocab_size,
            seq_len,
            batch_size,
            batch_size,
            n_train_batches,
            n_eval_batches,
            _SCREENING_MAX_CHARS_TRAIN,
            _SCREENING_MAX_CHARS_VAL,
            device,
        )
    except Exception as exc:
        meta["screening_wikitext_status"] = "data_failed"
        meta["error"] = str(exc)
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return meta

    if not _has_usable_batches(train_batches, val_batches):
        meta["screening_wikitext_status"] = "insufficient_tokens"
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return meta

    was_training = model.training
    try:
        params, buffers = clone_module_state(model)
    except Exception as exc:
        meta["screening_wikitext_status"] = "clone_failed"
        meta["error"] = str(exc)
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return meta

    try:
        model.eval()
        pre_ppl = functional_compute_perplexity(
            model, params, buffers, val_batches, vocab_size
        )

        loss_trajectory: dict = {}
        model.train()
        train_final_loss = functional_micro_train_loop(
            model,
            params,
            buffers,
            train_batches,
            vocab_size=vocab_size,
            n_steps=n_train_steps,
            lr=lr,
            loss_trajectory=loss_trajectory,
        )

        model.eval()
        post_ppl = functional_compute_perplexity(
            model, params, buffers, val_batches, vocab_size
        )

        meta.update(
            _finalize_ppl_result(
                pre_ppl=pre_ppl,
                post_ppl=post_ppl,
                train_final_loss=train_final_loss,
                vocab_size=vocab_size,
                variant=variant,
                n_train_steps=n_train_steps,
                seq_len=seq_len,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )
        meta["screening_wikitext_status"] = "ok"

        # Slope trajectory: sample at steps 10, 25, and final (50)
        sl_10 = loss_trajectory.get(10)
        sl_25 = loss_trajectory.get(25)
        sl_50 = loss_trajectory.get(n_train_steps)
        meta["screening_loss_10"] = round(sl_10, 6) if sl_10 is not None else None
        meta["screening_loss_25"] = round(sl_25, 6) if sl_25 is not None else None
        meta["screening_loss_50"] = round(sl_50, 6) if sl_50 is not None else None

        if sl_10 is not None and sl_50 is not None:
            # positive = improving, negative = diverging
            meta["screening_slope"] = round((sl_10 - sl_50) / 40.0, 6)
        else:
            meta["screening_slope"] = None

        if sl_10 is not None and sl_25 is not None and sl_50 is not None:
            interval_1 = (sl_10 - sl_25) / 15.0
            interval_2 = (sl_25 - sl_50) / 25.0
            meta["screening_slope_consistent"] = bool(interval_1 > 0 and interval_2 > 0)
        else:
            meta["screening_slope_consistent"] = None
    except Exception as exc:
        meta["screening_wikitext_status"] = "eval_failed"
        meta["error"] = str(exc)
    finally:
        model.train(was_training)

    meta["variant"] = variant
    meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return meta


# ── Full evaluation (investigation/validation) ──────────────────────────


def evaluate_wikitext_perplexity(
    model: nn.Module,
    vocab_size: int,
    device: str,
    variant: str = "wikitext-103-raw-v1",
    n_train_steps: int = 200,
    seq_len: int = 128,
    n_train_batches: int = 32,
    n_eval_batches: int = 8,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    lr: float = 3e-4,
    max_chars_train: int = _DEFAULT_MAX_CHARS_TRAIN,
    max_chars_val: int = _DEFAULT_MAX_CHARS_VAL,
) -> Dict[str, Any]:
    """Micro-train on WikiText and evaluate perplexity.

    Unlike ``screening_wikitext_eval``, this mutates the model in-place
    (appropriate for investigation/validation where the model is discarded
    after evaluation).
    """
    t0 = time.perf_counter()

    try:
        train_batches, val_batches, n_train_tok, n_val_tok = _prepare_batches(
            variant,
            vocab_size,
            seq_len,
            train_batch_size,
            eval_batch_size,
            n_train_batches,
            n_eval_batches,
            max_chars_train,
            max_chars_val,
            device,
        )
    except Exception as e:
        logger.warning("WikiText data preparation failed: %s", e)
        return {"wikitext_perplexity": None, "error": f"data_failed: {e}"}

    if not _has_usable_batches(train_batches, val_batches):
        return {
            "wikitext_perplexity": None,
            "error": "insufficient_tokens",
            "train_tokens": n_train_tok,
            "val_tokens": n_val_tok,
        }

    pre_ppl = compute_perplexity(model, val_batches, vocab_size)

    train_final_loss = micro_train_loop(
        model,
        train_batches,
        vocab_size,
        n_steps=n_train_steps,
        lr=lr,
    )

    post_ppl = compute_perplexity(model, val_batches, vocab_size)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return _finalize_ppl_result(
        pre_ppl=pre_ppl,
        post_ppl=post_ppl,
        train_final_loss=train_final_loss,
        vocab_size=vocab_size,
        variant=variant,
        n_train_steps=n_train_steps,
        seq_len=seq_len,
        elapsed_ms=elapsed_ms,
    )


def evaluate_wikitext_trajectory(
    model: nn.Module,
    vocab_size: int,
    device: str,
    checkpoints: tuple[int, ...] = (200, 500, 1000, 2000, 4000),
    variant: str = "wikitext-103-raw-v1",
    seq_len: int = 128,
    n_train_batches: int = 0,
    n_eval_batches: int = 16,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    lr: float = 3e-4,
    max_chars_train: int = 200_000_000,
    max_chars_val: int = 200_000,
    early_stop_factor: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate WikiText PPL at multiple training checkpoints.

    Trains the model continuously, pausing at each checkpoint to measure
    validation PPL.  Returns a trajectory dict keyed by step count.
    Mutates the model in-place (caller should discard after).

    Key defaults vs ``evaluate_wikitext_perplexity``:
    - ``max_chars_train=2_000_000`` (4x larger to reduce memorisation)
    - ``n_train_batches=0`` → auto-sized to max(checkpoint) so each batch
      is seen at most once during the longest run
    - ``early_stop_factor=2.0`` → stops training if val PPL exceeds
      ``2 * best_ppl_seen``, recording ``steps_to_divergence``

    Protocol: ``trajectory_probe_v2``
    """
    t0 = time.perf_counter()
    sorted_ckpts = sorted(checkpoints)

    # Auto-size n_train_batches: one unique batch per step at the longest
    # checkpoint, so no batch is repeated within a single run.
    if n_train_batches <= 0:
        n_train_batches = max(sorted_ckpts) if sorted_ckpts else 512

    try:
        train_batches, val_batches, n_train_tok, n_val_tok = _prepare_batches(
            variant,
            vocab_size,
            seq_len,
            train_batch_size,
            eval_batch_size,
            n_train_batches,
            n_eval_batches,
            max_chars_train,
            max_chars_val,
            device,
        )
    except Exception as e:
        logger.warning("WikiText trajectory data prep failed: %s", e)
        return {"error": f"data_failed: {e}", "checkpoints": {}}

    if not _has_usable_batches(train_batches, val_batches):
        return {"error": "insufficient_tokens", "checkpoints": {}}

    trajectory: Dict[int, Dict[str, Any]] = {}
    best_ppl: Optional[float] = None
    steps_to_divergence: Optional[int] = None
    peak_ppl: Optional[float] = None
    peak_step: Optional[int] = None
    diverged = False

    model.train()
    # Start at 10x LR for the first 100 steps to quickly calibrate the
    # lm_head (weight-tied embeddings init at std=1.0 → logits std≈16,
    # Mamba/RWKV can't overcome this at lr=3e-4 within 200 steps).
    # LinearLR decays from lr_warmup down to lr over warmup_steps.
    warmup_steps = 100
    lr_warmup = lr * 10.0
    opt = make_adamw(model.parameters(), lr=lr_warmup)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1.0,
        end_factor=lr / lr_warmup,
        total_iters=warmup_steps,
    )
    step = 0
    loss = torch.tensor(float("nan"))

    for ckpt_steps in sorted_ckpts:
        if diverged:
            # Still record the checkpoint but skip training
            trajectory[ckpt_steps] = {
                "ppl": None,
                "score": None,
                "loss": None,
                "early_stopped": True,
            }
            continue

        steps_needed = ckpt_steps - step
        compute_loss, get_last_loss = _make_scheduled_loss_fn(
            model,
            train_batches,
            vocab_size,
            step,
            steps_needed,
        )

        train_result = run_training_loop(
            model.parameters(),
            compute_loss,
            n_steps=steps_needed,
            optimizer=opt,
            optimizer_name="adamw",
            lr=lr_warmup,
            clip_grad=1.0,
            scheduler_step=scheduler.step,
        )
        step += train_result.steps_completed
        loss = get_last_loss()
        if train_result.diverged:
            logger.warning("Trajectory train loss not finite at step %d", step)
            diverged = True
            steps_to_divergence = step

        # Measure PPL at this checkpoint
        ppl = compute_perplexity(model, val_batches, vocab_size)
        score = wikitext_score_from_ppl(ppl, vocab_size)
        loss_val = (
            loss.item() if torch.is_tensor(loss) and torch.isfinite(loss) else None
        )
        trajectory[ckpt_steps] = {
            "ppl": round(ppl, 2) if ppl is not None else None,
            "score": score,
            "loss": round(loss_val, 4) if loss_val is not None else None,
        }
        logger.info(
            "Trajectory checkpoint %d: ppl=%.1f score=%.3f",
            ckpt_steps,
            ppl or 0.0,
            score or 0.0,
        )

        # Track best/peak PPL and check for divergence
        if ppl is not None:
            if peak_ppl is None or ppl < peak_ppl:
                peak_ppl = ppl
                peak_step = ckpt_steps
            if best_ppl is None or ppl < best_ppl:
                best_ppl = ppl
            elif early_stop_factor > 0 and ppl > best_ppl * early_stop_factor:
                logger.info(
                    "Early stopping: ppl %.1f > %.1f * %.1f (best=%.1f at earlier checkpoint)",
                    ppl,
                    best_ppl,
                    early_stop_factor,
                    best_ppl,
                )
                steps_to_divergence = ckpt_steps
                diverged = True

        model.train()

    # Compute improvement ratio between first two checkpoints
    improvement_ratio = None
    if len(sorted_ckpts) >= 2:
        ppl_first = trajectory.get(sorted_ckpts[0], {}).get("ppl")
        ppl_second = trajectory.get(sorted_ckpts[1], {}).get("ppl")
        if ppl_first and ppl_second and ppl_second > 0:
            improvement_ratio = round(ppl_first / ppl_second, 3)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "checkpoints": trajectory,
        "improvement_ratio": improvement_ratio,
        "peak_ppl": round(peak_ppl, 2) if peak_ppl is not None else None,
        "peak_step": peak_step,
        "steps_to_divergence": steps_to_divergence,
        "n_train_batches": len(train_batches),
        "checkpoint_steps": sorted_ckpts,
        "total_steps": step,
        "variant": variant,
        "protocol": "trajectory_probe_v2",
        "elapsed_ms": round(elapsed_ms, 1),
    }
