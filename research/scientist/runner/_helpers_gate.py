"""Runner helpers — split from _helpers. Re-exported via _helpers."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# ── Normalized loss_ratio ──
# loss_ratio = final_loss / initial_loss is init-dependent: Kaiming init
# yields initial_loss ~250 while small/ortho init yields ~ln(V).  This makes
# screening ratios (0.008) and investigation ratios (0.24) incomparable for
# the SAME architecture.  Normalizing against ln(vocab_size) — the expected
# cross-entropy of a uniform distribution — gives a consistent, interpretable
# metric across all stages and init schemes.

_DEFAULT_VOCAB_SIZE: int = 32_000
_REFERENCE_INITIAL_LOSS: float = math.log(_DEFAULT_VOCAB_SIZE)  # ~10.37
_INFIGHT_RANDOM_BASELINE: float = 11.52


def _build_source_map(nb: Any, result_ids: List[str]) -> Dict[str, Dict]:
    """Fetch program details for *result_ids* and return a {result_id: detail} map.

    Centralises the repeated ``[d or {} for d in (nb.get_program_details(ids) or [])]``
    pattern used in investigation/validation execution.

    Also reconstructs ``_behavioral_fingerprint`` from ``fingerprint_json``
    so that post-investigation fingerprint completion (CKA) can run.
    """

    details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
    source_map = {}
    for d in details:
        rid = d.get("result_id")
        if not rid:
            continue
        # Reconstruct _behavioral_fingerprint from fingerprint_json if absent
        if "_behavioral_fingerprint" not in d and d.get("fingerprint_json"):
            try:
                fp_data = d["fingerprint_json"]
                if isinstance(fp_data, str):
                    fp_data = json.loads(fp_data)
                d["_behavioral_fingerprint"] = fp_data
            except (json.JSONDecodeError, TypeError):
                pass
        source_map[rid] = d
    return source_map


def _corpus_type_from_config(config: Any) -> str:
    """Derive corpus type tag from RunConfig for gate calibration."""
    path = str(getattr(config, "corpus_path", "") or "").lower()
    if "wikitext" in path:
        return "wikitext103"
    if "tinystories" in path:
        return "tinystories"
    if "micro_corpus" in path:
        return "micro"
    return "unknown"


_ref_losses_cache: Dict[str, float] = {}
_ref_losses_ts: float = 0.0
_REF_LOSSES_TTL: float = 300.0


def get_reference_losses(db_path: str) -> Dict[str, float]:
    """Pull latest reference losses for gate calibration (cached 300s)."""
    global _ref_losses_cache, _ref_losses_ts

    now = time.monotonic()
    if _ref_losses_ts > 0.0 and (now - _ref_losses_ts) < _REF_LOSSES_TTL:
        return _ref_losses_cache

    ref: Dict[str, float] = {}
    try:
        from ..notebook.shared_conn import get_notebook_conn

        conn = get_notebook_conn(str(db_path))
        row = conn.execute("""
            SELECT AVG(p.final_loss) as avg_loss
            FROM program_results p
            JOIN leaderboard l ON p.result_id = l.result_id
            WHERE l.reference_name = 'GPT-2'
            AND p.final_loss IS NOT NULL
            AND p.final_loss > 0
            AND p.n_train_steps >= 4000
        """).fetchone()
        if row and row["avg_loss"]:
            ref["gpt2_wikitext103_tiktoken"] = float(row["avg_loss"])
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("get_reference_losses failed: %s", exc)
    _ref_losses_cache = ref
    _ref_losses_ts = now
    return ref


def stage1_learning_gate(
    final_loss: float,
    loss_ratio: float,
    initial_loss: Optional[float],
    n_steps: int,
    corpus_type: str,
    tokenizer: str,
    reference_losses: Dict[str, float],
) -> Tuple[bool, str]:
    """Corpus-aware Stage 1 pass/fail gate.

    Returns (passed, reason).

    Gate logic:
    1. Hard kill: model made things worse (loss_ratio > 1.0)
    2. Hard kill: model clearly didn't learn (ratio > 0.95 after 200+ steps)
    3. Corpus-relative: if WikiText-103 + tiktoken, compare to GPT-2 floor
    4. Absolute floor: final_loss must be below random baseline
    """
    # Random baseline (entropy floor) — used by gates 2 and 4
    # tiktoken cl100k_base: ln(100277) ~ 11.52
    # byte tokenizer 32K vocab: ln(32000) ~ 10.37
    tok_prefix = tokenizer.split("_")[0] if tokenizer else ""
    random_baseline = {"tiktoken": 11.52, "byte": 10.37}.get(tok_prefix, 11.52)

    # Gate 1: Made things worse
    if loss_ratio > 1.0:
        return False, f"loss_ratio={loss_ratio:.3f} > 1.0 — training diverged"

    # Gate 2: Clearly didn't learn — entropy-relative threshold.
    #
    # Problem: fixed 0.95 ratio penalizes complex architectures that start
    # close to the entropy floor. A 12-op graph at init=12 has only 1.6 nats
    # of headroom above ln(vocab), so 5% improvement (0.6 nats) consumes 37%
    # of headroom. A 4-op graph at init=190 has 180 nats of headroom — 5%
    # (9.5 nats) is trivial.
    #
    # Fix: scale threshold by headroom. Require using at least 10% of
    # improvable headroom (init_loss - entropy_floor).
    # At init=190 (headroom=180): need 18 nats drop → threshold=0.905
    # At init=12  (headroom=1.6): need 0.16 nats drop → threshold=0.987
    if n_steps >= 200:
        _init = initial_loss if initial_loss and initial_loss > 0 else 100.0
        _ratio_threshold = _headroom_ratio_threshold(
            _init,
            random_baseline=random_baseline,
        )
        _headroom = max(_init - random_baseline, 0.5)
        if loss_ratio > _ratio_threshold:
            return False, (
                f"loss_ratio={loss_ratio:.3f} after {n_steps} steps — "
                f"model reduced loss by only {(1 - loss_ratio) * 100:.1f}% "
                f"(threshold={_ratio_threshold:.3f}, headroom={_headroom:.1f})"
            )

    # Gate 3: Corpus-relative gate (WikiText-103 + tiktoken only)
    if corpus_type == "wikitext103" and "tiktoken" in tokenizer:
        gpt2_floor = reference_losses.get("gpt2_wikitext103_tiktoken")
        if gpt2_floor is not None:
            relative_threshold = gpt2_floor * 4.0
            if final_loss > relative_threshold:
                return False, (
                    f"final_loss={final_loss:.3f} > "
                    f"4x GPT-2 floor ({relative_threshold:.3f}) — "
                    f"not competitive on WikiText-103"
                )

    # Gate 4: Absolute random baseline — headroom-aware.
    #
    # Models that start far above the entropy floor (e.g., evolution candidates
    # at init=200+ nats) may plateau just above the baseline after only 750
    # training steps, despite demonstrating strong learning (loss_ratio < 0.2).
    # Relax the floor for proven learners; keep it strict for non-learners.
    if loss_ratio is not None and loss_ratio < 0.20:
        # Strong learner (≥80% loss reduction) — pass regardless of absolute
        # floor.  These need more training steps, not rejection.
        pass
    elif loss_ratio is not None and loss_ratio < 0.50:
        # Moderate learner — allow slight overshoot above baseline.
        if final_loss > random_baseline * 1.05:
            return False, (
                f"final_loss={final_loss:.3f} above relaxed baseline "
                f"({random_baseline * 1.05:.3f}) — moderate learner but "
                f"not close enough (ratio={loss_ratio:.3f})"
            )
    elif final_loss > random_baseline * 0.95:
        return False, (
            f"final_loss={final_loss:.3f} near random baseline "
            f"({random_baseline:.2f}) — model learned nothing meaningful"
        )

    return True, "passed"


def _headroom_ratio_threshold(
    initial_loss: float,
    *,
    random_baseline: float,
    headroom_fraction: float = 0.10,
    min_headroom: float = 0.5,
    min_ratio: float = 0.90,
    max_ratio: float = 0.99,
) -> float:
    """Compute a loss-ratio threshold scaled by improvable entropy headroom."""
    init = initial_loss if initial_loss > 0 else 100.0
    headroom = max(init - random_baseline, min_headroom)
    min_improvement = headroom * headroom_fraction
    raw = 1.0 - min_improvement / max(init, 1.0)
    return max(min_ratio, min(max_ratio, raw))


def normalized_loss_ratio(
    final_loss: float,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
) -> float:
    """Compute init-independent loss ratio.

    Returns final_loss / ln(vocab_size), measuring what fraction of
    maximum-entropy loss the model achieves.  Lower is better.
    A value of 0.2 means the model achieved 80% of the possible
    entropy reduction from a uniform distribution over the vocabulary.

    This replaces the old final_loss/initial_loss which was wildly
    init-dependent (Kaiming gave 0.008, small-init gave 0.24 for
    the same architecture and final loss).
    """
    ref = math.log(vocab_size) if vocab_size > 0 else _REFERENCE_INITIAL_LOSS
    return final_loss / max(ref, 1e-6)


def resolve_stage1_gate_metrics(
    *,
    initial_loss: Optional[float],
    final_loss: Optional[float],
    validation_loss: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Choose the most trustworthy loss signal for the final Stage-1 gate.

    Prefer held-out validation loss when available; otherwise fall back to the
    last training loss. This avoids rejecting good learners based on a noisy
    final minibatch after a strong validation result has already been recorded.
    """
    gate_loss = validation_loss if validation_loss is not None else final_loss
    if gate_loss is None:
        gate_loss = float("inf")
    init = initial_loss if initial_loss and initial_loss > 0 else 1e-6
    gate_ratio = gate_loss / max(init, 1e-6)
    gate_source = "validation_loss" if validation_loss is not None else "final_loss"
    return gate_loss, gate_ratio, gate_source


# ── Inflight training health checks ──


@dataclass
class InflightState:
    """Mutable state for inflight training checks."""

    __slots__ = ("recent_losses", "grad_strikes", "window")
    recent_losses: List[float]
    grad_strikes: int
    window: int

    def __init__(self, window: int = 20):
        self.recent_losses = []
        self.grad_strikes = 0
        self.window = window


def check_inflight_health(
    step: int,
    loss_val: float,
    grad_norm: float,
    min_loss: float,
    initial_loss: Optional[float],
    total_steps: int,
    state: InflightState,
    spike_ratio: float = 2.0,
    spike_window: int = 10,
    cv_threshold: float = 0.5,
    progress_threshold: Optional[float] = None,
    grad_norm_limit: float = 100.0,
    grad_norm_strikes: int = 3,
) -> Optional[Dict[str, Any]]:
    """Run all inflight training health checks.

    Returns None if healthy, or a dict with 'error' and 'error_type' if
    the run should be aborted.
    """
    # Track recent losses
    state.recent_losses.append(loss_val)
    if len(state.recent_losses) > state.window:
        state.recent_losses.pop(0)

    # Check 1: loss spike far above running minimum
    if step >= spike_window and min_loss > 0 and loss_val > spike_ratio * min_loss:
        return {
            "error": (
                f"inflight_loss_spike: step {step}, "
                f"loss={loss_val:.4f} > {spike_ratio}x min={min_loss:.4f}"
            ),
            "error_type": "inflight_loss_spike",
        }

    # Check 2: wild oscillation (high CV over recent window)
    w = state.window
    if step >= w and len(state.recent_losses) >= w:
        _mean = sum(state.recent_losses) / w
        if _mean > 0:
            _var = sum((x - _mean) ** 2 for x in state.recent_losses) / w
            _cv = (_var**0.5) / _mean
            if _cv > cv_threshold:
                return {
                    "error": (
                        f"inflight_oscillation: step {step}, "
                        f"CV={_cv:.3f} over last {w} steps "
                        f"(mean={_mean:.2f}, std={_var**0.5:.2f})"
                    ),
                    "error_type": "inflight_oscillation",
                }

    # Check 3a: loss diverging — if loss exceeds initial by 50%, abort
    if step >= spike_window and initial_loss and loss_val > initial_loss * 1.5:
        return {
            "error": (
                f"inflight_divergence: step {step}, "
                f"loss={loss_val:.4f} > 1.5x initial={initial_loss:.4f}"
            ),
            "error_type": "inflight_divergence",
        }

    # Check 3b: no progress at 25% mark
    quarter = total_steps // 4
    ratio_threshold = progress_threshold
    if ratio_threshold is None and initial_loss and initial_loss > 0:
        ratio_threshold = _headroom_ratio_threshold(
            initial_loss,
            random_baseline=_INFIGHT_RANDOM_BASELINE,
        )
    if ratio_threshold is None:
        ratio_threshold = 0.95
    if step == quarter and initial_loss and loss_val >= initial_loss * ratio_threshold:
        return {
            "error": (
                f"inflight_no_progress: at step {step}/{total_steps}, "
                f"loss={loss_val:.4f} vs initial={initial_loss:.4f} "
                f"(ratio={loss_val / initial_loss:.3f}, "
                f"threshold={ratio_threshold:.3f})"
            ),
            "error_type": "inflight_no_progress",
        }

    # Check 4: persistent gradient explosion
    if grad_norm > grad_norm_limit:
        state.grad_strikes += 1
        if state.grad_strikes >= grad_norm_strikes:
            return {
                "error": (
                    f"inflight_grad_explosion: {grad_norm_strikes} consecutive "
                    f"steps with grad_norm > {grad_norm_limit:.0f} "
                    f"(last={grad_norm:.1f})"
                ),
                "error_type": "inflight_grad_explosion",
            }
    else:
        state.grad_strikes = 0

    return None


def clear_gpu_memory() -> None:
    """Release GPU memory and run garbage collection.

    Centralised cleanup to avoid duplicating torch.cuda.empty_cache() +
    gc.collect() across 13+ call sites in runner submodules.
    Also unloads any Ollama model to free VRAM for training.
    """
    import gc

    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except (ImportError, RuntimeError) as e:
        logger.debug("GPU memory cleanup skipped: %s", e)
    gc.collect()
    _unload_ollama_if_running()


_ollama_last_unload: float = 0.0


def _unload_ollama_if_running() -> None:
    """Unload Ollama model from GPU if the server is reachable.

    Debounced to at most once per 60 seconds — clear_gpu_memory() is called
    20+ times per cycle and we don't want to spam HTTP requests.
    No-ops silently if Ollama isn't running.
    """
    global _ollama_last_unload
    import time

    now = time.monotonic()
    if now - _ollama_last_unload < 60.0:
        return
    _ollama_last_unload = now

    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        if r.status_code != 200:
            return
        models = r.json().get("models", [])
        if not models:
            return
        for m in models:
            name = m.get("name", "")
            if name:
                requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "prompt": "", "keep_alive": 0},
                    timeout=5,
                )
    except (ImportError, OSError, ValueError) as e:
        logger.debug("Ollama unload skipped: %s", e)
