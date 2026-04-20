"""Associative Recall probe — measures non-local key-value retrieval.

Format: k1a k1b v1 k2a k2b v2 ... kNa kNb vN [SEP] kQa kQb [ANS] -> vQ

The model sees N=20 shuffled key-value pairs where keys are 2-token
sequences and values are single tokens (all from a restricted 256-token
vocabulary). After a separator, a random key is queried and the model
must predict the associated value at the answer position.

Why this discriminates architectures:
- Full causal attention:          passes — exact content-based retrieval
- Conv-only (conv-3, token_merge): fails — cannot bridge 63-token gap
- Mamba / SSM:                    fails — state compression is lossy
- RWKV:                           fails — same lossy compression
- Linear attention (no features): fails — lossy
- Based (feature map attention):  partially passes — designed for this

Mamba and RWKV failing is CORRECT behavior, not a probe bug. Their
failure mechanism (state compression) differs fundamentally from conv-3
(zero receptive field). The soft gate in leaderboard_scoring.py handles
this distinction via the 3-signal AND.

Random chance baseline: 1/256 ≈ 0.4%.
Pass signal: >15% accuracy at step 500.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .retrieval_eval_utils import run_retrieval_probe_learning_curve

logger = logging.getLogger(__name__)

_VOCAB_LO = 100
_VOCAB_HI = 356  # 256 tokens: IDs 100-355
_VOCAB_N = _VOCAB_HI - _VOCAB_LO
_TIMEOUT_S = 90.0
_RANDPERM_CACHE_LIMIT = 64
_RANDPERM_CACHE: "OrderedDict[tuple[int, torch.device], torch.Tensor]" = OrderedDict()


@dataclass(slots=True)
class ARResult:
    """Result from associative recall probe."""

    auc: float = 0.0
    final_acc: float = 0.0
    learning_curve: List[Tuple[int, float]] = None
    timed_out: bool = False
    above_chance: bool = False
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ar_auc": self.auc,
            "ar_final_acc": self.final_acc,
            "ar_learning_curve": self.learning_curve,
            "ar_timed_out": self.timed_out,
            "ar_above_chance": self.above_chance,
            "ar_steps_trained": self.steps_trained,
            "ar_status": self.status,
            "ar_elapsed_ms": self.elapsed_ms,
        }


def _get_special_tokens(model: nn.Module) -> Tuple[int, int]:
    """Pick SEP and ANSWER token IDs outside the restricted vocab."""
    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                vocab_size = m.num_embeddings
                break
    if vocab_size is None:
        vocab_size = 50258

    if vocab_size > 50257:
        return 50256, 50257
    return vocab_size - 2, vocab_size - 1


def _generate_ar_batch(
    batch_size: int,
    n_pairs: int,
    sep_token: int,
    ans_token: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of associative recall sequences — fully vectorized.

    Each sample draws 2*n_pairs + n_pairs = 3*n_pairs unique tokens from
    the restricted vocab (256 tokens). The first 2*n_pairs become keys
    (paired), the remaining n_pairs become values. This guarantees no
    key-key or key-value token overlap without Python set operations.

    Returns (input_ids, target_values) on `device`.
    """
    n_tokens = 3 * n_pairs  # total unique tokens needed per sample
    seq_len = 3 * n_pairs + 4

    # Draw n_tokens unique tokens per sample via batched randperm on CPU
    # (torch.randperm has no batch dim, so we stack — still faster than
    # Python loops with set filtering)
    cache_key = (int(batch_size), torch.device("cpu"))
    perms = _RANDPERM_CACHE.get(cache_key)
    if perms is None:
        perms = torch.stack([torch.randperm(_VOCAB_N) for _ in range(batch_size)])
        _RANDPERM_CACHE[cache_key] = perms
        while len(_RANDPERM_CACHE) > _RANDPERM_CACHE_LIMIT:
            _RANDPERM_CACHE.popitem(last=False)
    else:
        _RANDPERM_CACHE.move_to_end(cache_key)
    perms = perms[torch.randperm(batch_size)]
    tokens = perms[:, :n_tokens] + _VOCAB_LO  # (B, 3*n_pairs)

    keys = tokens[:, : 2 * n_pairs].reshape(batch_size, n_pairs, 2)  # (B, N, 2)
    values = tokens[:, 2 * n_pairs :]  # (B, N)

    # Shuffle pair order per sample
    pair_perms = torch.stack([torch.randperm(n_pairs) for _ in range(batch_size)])
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(pair_perms)
    keys = keys[batch_idx, pair_perms]  # (B, N, 2)
    values = values[batch_idx, pair_perms]  # (B, N)

    # Build sequences: k1a k1b v1 k2a k2b v2 ... kNa kNb vN [SEP] kQa kQb [ANS]
    batch = torch.zeros(batch_size, seq_len, dtype=torch.long)
    pair_idx = torch.arange(n_pairs)
    batch[:, pair_idx * 3] = keys[:, :, 0]
    batch[:, pair_idx * 3 + 1] = keys[:, :, 1]
    batch[:, pair_idx * 3 + 2] = values

    sep_pos = 3 * n_pairs
    batch[:, sep_pos] = sep_token
    batch[:, sep_pos + 3] = ans_token

    # Random query key per sample
    q_idx = torch.randint(0, n_pairs, (batch_size,))
    b_idx = torch.arange(batch_size)
    batch[:, sep_pos + 1] = keys[b_idx, q_idx, 0]
    batch[:, sep_pos + 2] = keys[b_idx, q_idx, 1]
    targets = values[b_idx, q_idx]

    return batch.to(device), targets.to(device)


def _generate_eval_set(
    n_eval: int,
    n_pairs: int,
    sep_token: int,
    ans_token: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed eval set with seed 42 for reproducibility."""
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    try:
        return _generate_ar_batch(n_eval, n_pairs, sep_token, ans_token, device)
    finally:
        torch.random.set_rng_state(rng_state)


def _trapezoidal_auc(curve: List[Tuple[int, float]], max_steps: int) -> float:
    """Normalized AUC via trapezoidal rule. Divides by max_steps * 1.0."""
    if len(curve) < 2:
        return curve[0][1] if curve else 0.0

    area = 0.0
    for i in range(1, len(curve)):
        dt = curve[i][0] - curve[i - 1][0]
        area += 0.5 * dt * (curve[i - 1][1] + curve[i][1])

    return area / max(max_steps, 1)


def associative_recall_score(
    model: nn.Module,
    n_pairs: int = 20,
    n_train_steps: int = 500,
    n_eval: int = 200,
    eval_every: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> ARResult:
    """Train a deepcopy on associative recall, measure key-value retrieval.

    Measures whether the model can store key-value pairs presented in a
    sequence and retrieve the correct value when queried with a key.
    Sequence length ~67 tokens ensures local ops (conv-3) cannot bridge
    the gap. Full attention passes; Mamba/RWKV/linear attention fail
    (this is correct behavior — see module docstring).

    The original model is NOT modified — a deepcopy is used.

    Returns ARResult with:
      auc:            float [0, 1] — area under accuracy learning curve
      final_acc:      float [0, 1] — accuracy at final eval step
      learning_curve: list of (step, accuracy) tuples
      timed_out:      bool — True if probe exceeded timeout
      above_chance:   bool — final_acc > 0.05 (10x random chance)
    """
    t0 = time.perf_counter()
    result = ARResult(learning_curve=[])

    try:
        sep_token, ans_token = _get_special_tokens(model)
    except Exception as e:
        result.status = f"copy_failed: {e}"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    try:
        eval_ids, eval_targets = _generate_eval_set(
            n_eval,
            n_pairs,
            sep_token,
            ans_token,
            device,
        )
    except Exception as e:
        result.status = f"eval_gen_failed: {e}"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    ans_pos = 3 * n_pairs + 3

    try:
        learning_curve, steps_trained, timed_out, status = (
            run_retrieval_probe_learning_curve(
                model,
                n_train_steps=n_train_steps,
                eval_every=eval_every,
                eval_ids=eval_ids,
                eval_targets=eval_targets,
                batch_size=batch_size,
                lr=lr,
                device=device,
                deadline=t0 + timeout_s,
                make_train_batch=lambda bs, dev: _generate_ar_batch(
                    bs,
                    n_pairs,
                    sep_token,
                    ans_token,
                    dev,
                ),
                query_pos=ans_pos,
                vocab_lo=_VOCAB_LO,
                vocab_hi=_VOCAB_HI,
            )
        )
        result.learning_curve = learning_curve
        result.steps_trained = steps_trained
        result.timed_out = timed_out
        result.status = status
    except Exception as e:
        result.status = f"train_failed: {e}"
    finally:
        del eval_ids, eval_targets

    if result.learning_curve:
        result.final_acc = result.learning_curve[-1][1]
        result.auc = round(_trapezoidal_auc(result.learning_curve, n_train_steps), 4)
        result.above_chance = result.final_acc > 0.05

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
