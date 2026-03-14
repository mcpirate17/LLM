"""WikiText perplexity evaluation for architecture robustness.

Downloads WikiText-2 (or WikiText-103) from HuggingFace, micro-trains
the candidate model on the train split, then evaluates perplexity on the
validation split. This tests whether an architecture can learn real
linguistic patterns, not just synthetic data.

Uses the existing CorpusTokenBatcher infrastructure via a cached text file.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import (
    tokenize_file,
    make_batches,
    micro_train_loop,
    compute_perplexity,
)

logger = logging.getLogger(__name__)

_WIKITEXT_CACHE_DIR = Path.home() / ".cache" / "aria" / "wikitext"

# Subset sizes (chars) to keep evaluation fast
_DEFAULT_MAX_CHARS_TRAIN = 500_000   # ~500KB of WikiText for micro-training
_DEFAULT_MAX_CHARS_VAL = 50_000      # ~50KB for validation perplexity

# Screening defaults — smaller budget for fast turnaround
_SCREENING_MAX_CHARS_TRAIN = 100_000
_SCREENING_MAX_CHARS_VAL = 20_000
_SCREENING_N_TRAIN_STEPS = 50
_SCREENING_N_TRAIN_BATCHES = 16
_SCREENING_N_EVAL_BATCHES = 4
_SCREENING_BATCH_SIZE = 4
_SCREENING_METRIC_VERSION = "screening_wikitext_v1"

# WikiText-103 for VALIDATED-stage "final boss" evaluation.
# ~103M tokens train, ~250K tokens val — 50x larger than WikiText-2.
WIKITEXT_103_VARIANT = "wikitext-103-raw-v1"
_WIKITEXT_103_MAX_CHARS_TRAIN = 20_000_000  # 20MB — enough for 4000 unique batches
_WIKITEXT_103_MAX_CHARS_VAL = 200_000       # 200KB val for reliable PPL


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
    variant: str = "wikitext-2-raw-v1",
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

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` package required for WikiText evaluation. "
            "Install with: pip install datasets"
        )

    logger.info("Downloading WikiText variant=%s ...", variant)
    ds = load_dataset("wikitext", variant, trust_remote_code=True)

    # Extract and truncate text
    for split_name, path, max_chars in [
        ("train", train_path, max_chars_train),
        ("validation", val_path, max_chars_val),
    ]:
        texts = ds[split_name]["text"]
        combined = "\n".join(t for t in texts if t.strip())
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        path.write_text(combined, encoding="utf-8")

    logger.info("WikiText cached at %s", cache_dir)
    return train_path, val_path


# ── Batch cache ──────────────────────────────────────────────────────────
# Avoids repeated tokenization + batch construction across candidates.
# Keyed by (variant, vocab_size, seq_len, batch_size, n_batches,
#            max_chars_train, max_chars_val, split, seed).

_batch_cache: Dict[tuple, List[torch.Tensor]] = {}


def _get_cached_batches(
    variant: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    max_chars: int,
    device: str,
    split: str,
    seed: int,
) -> Optional[List[torch.Tensor]]:
    """Return cached batches if available, moving to *device* if needed."""
    key = (variant, vocab_size, seq_len, batch_size, n_batches, max_chars, split, seed)
    batches = _batch_cache.get(key)
    if batches is None:
        return None
    target = torch.device(device)
    if batches[0].device != target:
        batches = [b.to(target) for b in batches]
        _batch_cache[key] = batches
    return batches


def _put_cached_batches(
    variant: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    max_chars: int,
    split: str,
    seed: int,
    batches: List[torch.Tensor],
) -> None:
    key = (variant, vocab_size, seq_len, batch_size, n_batches, max_chars, split, seed)
    _batch_cache[key] = batches


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
    train = _get_cached_batches(
        variant, vocab_size, seq_len, train_batch_size,
        n_train_batches, max_chars_train, device, "train", 42,
    )
    val = _get_cached_batches(
        variant, vocab_size, seq_len, eval_batch_size,
        n_eval_batches, max_chars_val, device, "validation", 123,
    )
    if train is not None and val is not None:
        return train, val, -1, -1  # -1 = cached, counts unknown

    train_path, val_path = _download_wikitext(variant, max_chars_train, max_chars_val)
    train_tokens = tokenize_file(train_path, vocab_size)
    val_tokens = tokenize_file(val_path, vocab_size)

    if len(train_tokens) < seq_len + 1 or len(val_tokens) < seq_len + 1:
        return None, None, len(train_tokens), len(val_tokens)

    if train is None:
        train = make_batches(train_tokens, train_batch_size, seq_len, n_train_batches, device, seed=42)
        if train:
            _put_cached_batches(
                variant, vocab_size, seq_len, train_batch_size,
                n_train_batches, max_chars_train, "train", 42, train,
            )
    if val is None:
        val = make_batches(val_tokens, eval_batch_size, seq_len, n_eval_batches, device, seed=123)
        if val:
            _put_cached_batches(
                variant, vocab_size, seq_len, eval_batch_size,
                n_eval_batches, max_chars_val, "validation", 123, val,
            )

    return train, val, len(train_tokens), len(val_tokens)


# ── Score helper ─────────────────────────────────────────────────────────

def wikitext_score_from_ppl(ppl: Optional[float], vocab_size: int = 32000) -> Optional[float]:
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
    variant: str = "wikitext-2-raw-v1",
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
            variant, vocab_size, seq_len, batch_size, batch_size,
            n_train_batches, n_eval_batches,
            _SCREENING_MAX_CHARS_TRAIN, _SCREENING_MAX_CHARS_VAL,
            device,
        )
    except Exception as exc:
        meta["screening_wikitext_status"] = "data_failed"
        meta["error"] = str(exc)
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return meta

    if train_batches is None or val_batches is None or not train_batches or not val_batches:
        meta["screening_wikitext_status"] = "insufficient_tokens"
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return meta

    # Clone model so micro-training doesn't mutate the live weights
    was_training = model.training
    try:
        clone = copy.deepcopy(model)
    except Exception as exc:
        meta["screening_wikitext_status"] = "clone_failed"
        meta["error"] = str(exc)
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return meta

    try:
        # Pre-training perplexity (eval mode on clone)
        pre_ppl = compute_perplexity(clone, val_batches, vocab_size)

        # Micro-train the clone
        train_final_loss = micro_train_loop(
            clone, train_batches, vocab_size,
            n_steps=n_train_steps, lr=lr,
        )

        # Post-training perplexity
        post_ppl = compute_perplexity(clone, val_batches, vocab_size)

        ppl_improvement = None
        if pre_ppl is not None and post_ppl is not None and pre_ppl > 0:
            ppl_improvement = round(post_ppl / pre_ppl, 4)

        meta["wikitext_perplexity"] = round(post_ppl, 2) if post_ppl is not None else None
        meta["wikitext_pre_perplexity"] = round(pre_ppl, 2) if pre_ppl is not None else None
        meta["wikitext_score"] = wikitext_score_from_ppl(post_ppl, vocab_size)
        meta["wikitext_ppl_improvement"] = ppl_improvement
        meta["train_final_loss"] = round(train_final_loss, 6)
        meta["screening_wikitext_status"] = "ok"
    except Exception as exc:
        meta["screening_wikitext_status"] = "eval_failed"
        meta["error"] = str(exc)
    finally:
        del clone
        # Restore original model's training mode
        model.train(was_training)

    meta["variant"] = variant
    meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return meta


# ── Full evaluation (investigation/validation) ──────────────────────────

def evaluate_wikitext_perplexity(
    model: nn.Module,
    vocab_size: int,
    device: str,
    variant: str = "wikitext-2-raw-v1",
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
            variant, vocab_size, seq_len, train_batch_size, eval_batch_size,
            n_train_batches, n_eval_batches,
            max_chars_train, max_chars_val,
            device,
        )
    except Exception as e:
        logger.warning("WikiText data preparation failed: %s", e)
        return {"wikitext_perplexity": None, "error": f"data_failed: {e}"}

    if train_batches is None or val_batches is None or not train_batches or not val_batches:
        return {
            "wikitext_perplexity": None,
            "error": "insufficient_tokens",
            "train_tokens": n_train_tok,
            "val_tokens": n_val_tok,
        }

    pre_ppl = compute_perplexity(model, val_batches, vocab_size)

    train_final_loss = micro_train_loop(
        model, train_batches, vocab_size,
        n_steps=n_train_steps, lr=lr,
    )

    post_ppl = compute_perplexity(model, val_batches, vocab_size)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

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


def evaluate_wikitext_trajectory(
    model: nn.Module,
    vocab_size: int,
    device: str,
    checkpoints: tuple[int, ...] = (200, 500, 1000, 2000, 4000),
    variant: str = "wikitext-2-raw-v1",
    seq_len: int = 128,
    n_train_batches: int = 0,
    n_eval_batches: int = 16,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    lr: float = 3e-4,
    max_chars_train: int = 2_000_000,
    max_chars_val: int = _DEFAULT_MAX_CHARS_VAL,
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
    import torch.optim

    t0 = time.perf_counter()
    sorted_ckpts = sorted(checkpoints)

    # Auto-size n_train_batches: one unique batch per step at the longest
    # checkpoint, so no batch is repeated within a single run.
    if n_train_batches <= 0:
        n_train_batches = max(sorted_ckpts) if sorted_ckpts else 512

    try:
        train_batches, val_batches, n_train_tok, n_val_tok = _prepare_batches(
            variant, vocab_size, seq_len, train_batch_size, eval_batch_size,
            n_train_batches, n_eval_batches,
            max_chars_train, max_chars_val,
            device,
        )
    except Exception as e:
        logger.warning("WikiText trajectory data prep failed: %s", e)
        return {"error": f"data_failed: {e}", "checkpoints": {}}

    if not train_batches or not val_batches:
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr_warmup)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=lr / lr_warmup,
        total_iters=warmup_steps,
    )
    step = 0
    loss = torch.tensor(float("nan"))

    for ckpt_steps in sorted_ckpts:
        if diverged:
            # Still record the checkpoint but skip training
            trajectory[ckpt_steps] = {
                "ppl": None, "score": None, "loss": None,
                "early_stopped": True,
            }
            continue

        # Train from current step to this checkpoint
        steps_needed = ckpt_steps - step
        for _ in range(steps_needed):
            batch = train_batches[step % len(train_batches)]
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]
            loss = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]),
                batch[:, 1:].reshape(-1),
            )
            if not torch.isfinite(loss):
                logger.warning("Trajectory train loss not finite at step %d", step)
                diverged = True
                steps_to_divergence = step
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            step += 1

        # Measure PPL at this checkpoint
        ppl = compute_perplexity(model, val_batches, vocab_size)
        score = wikitext_score_from_ppl(ppl, vocab_size)
        loss_val = loss.item() if torch.is_tensor(loss) and torch.isfinite(loss) else None
        trajectory[ckpt_steps] = {
            "ppl": round(ppl, 2) if ppl is not None else None,
            "score": score,
            "loss": round(loss_val, 4) if loss_val is not None else None,
        }
        logger.info(
            "Trajectory checkpoint %d: ppl=%.1f score=%.3f",
            ckpt_steps, ppl or 0.0, score or 0.0,
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
                    ppl, best_ppl, early_stop_factor, best_ppl,
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


# WikiText-103 defaults — 50x more data than WikiText-2
_WIKITEXT103_MAX_CHARS_TRAIN = 20_000_000   # ~20MB, covers most of train split
_WIKITEXT103_MAX_CHARS_VAL = 200_000        # ~200KB validation


def evaluate_wikitext103_validation(
    model: nn.Module,
    vocab_size: int,
    device: str,
    n_train_steps: int = 4000,
    seq_len: int = 128,
    n_train_batches: int = 0,
    n_eval_batches: int = 16,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    lr: float = 3e-4,
    max_chars_train: int = _WIKITEXT103_MAX_CHARS_TRAIN,
    max_chars_val: int = _WIKITEXT103_MAX_CHARS_VAL,
) -> Dict[str, Any]:
    """VALIDATED-stage WikiText-103 confirmation eval.

    Trains on WikiText-103 (much larger corpus than WikiText-2) to confirm
    that a model's frontier claim generalises beyond small-corpus memorisation.

    If ``wikitext103_ppl / wikitext2_peak_ppl < 2.0``, the frontier claim
    stands.  If > 2.0, the model's capability was WikiText-2-specific.

    Protocol: ``validated_wikitext103_v1``
    """
    t0 = time.perf_counter()

    if n_train_batches <= 0:
        n_train_batches = n_train_steps

    try:
        train_batches, val_batches, n_train_tok, n_val_tok = _prepare_batches(
            "wikitext-103-raw-v1", vocab_size, seq_len,
            train_batch_size, eval_batch_size,
            n_train_batches, n_eval_batches,
            max_chars_train, max_chars_val,
            device,
        )
    except Exception as e:
        logger.warning("WikiText-103 data prep failed: %s", e)
        return {"error": f"data_failed: {e}"}

    if not train_batches or not val_batches:
        return {"error": "insufficient_tokens"}

    pre_ppl = compute_perplexity(model, val_batches, vocab_size)

    train_final_loss = micro_train_loop(
        model, train_batches, vocab_size,
        n_steps=n_train_steps, lr=lr,
    )

    post_ppl = compute_perplexity(model, val_batches, vocab_size)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "wikitext103_perplexity": round(post_ppl, 2) if post_ppl is not None else None,
        "wikitext103_pre_perplexity": round(pre_ppl, 2) if pre_ppl is not None else None,
        "wikitext103_score": wikitext_score_from_ppl(post_ppl, vocab_size),
        "train_final_loss": round(train_final_loss, 6),
        "variant": "wikitext-103-raw-v1",
        "n_train_steps": n_train_steps,
        "protocol": "validated_wikitext103_v1",
        "elapsed_ms": round(elapsed_ms, 1),
    }
