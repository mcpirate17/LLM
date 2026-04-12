"""Multi-hop retrieval — measures ability to chain associations across distances.

Generates chains: A→B at position P1, B→C at position P2, query A at end.
Model must produce C (2-hop retrieval).  Also tests 3-hop chains.

Tests at sequence lengths 256, 512, 1024.
Score = accuracy across lengths and hop depths.
Output column: ``robustness_long_ctx_multi_hop_score``
"""

from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_adamw

logger = logging.getLogger(__name__)

_TIMEOUT_S = 90.0
# Token ranges — same restricted vocab as AR probe
_VOCAB_LO = 100
_VOCAB_HI = 356
_VOCAB_N = _VOCAB_HI - _VOCAB_LO
# Special tokens
_SEP_TOKEN = 2
_QUERY_TOKEN = 3
_ANS_TOKEN = 4
# Chance baseline: 1/256 ≈ 0.004
_CHANCE = 1.0 / _VOCAB_N
_PASS_THRESHOLD = 3 * _CHANCE


@dataclass(slots=True)
class MultiHopResult:
    """Result from multi-hop retrieval probe."""

    score: float = 0.0
    per_config: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "multi_hop_score": self.score,
            "multi_hop_per_config": self.per_config,
            "multi_hop_elapsed_ms": self.elapsed_ms,
            "multi_hop_status": self.status,
        }


def _generate_multi_hop_batch(
    batch_size: int,
    seq_len: int,
    n_hops: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of multi-hop retrieval sequences.

    For n_hops=2: places A→B and B→C associations in the sequence,
    queries A at the end, target is C.
    For n_hops=3: places A→B, B→C, C→D, queries A, target is D.

    Each association is a 3-token group: [key1, key2, value].
    Associations are spread evenly through the first 70% of the sequence.
    The query section at the end is: [SEP] [key1_of_A] [key2_of_A] [ANS]
    """
    n_chain_tokens = n_hops + 1  # number of distinct values in the chain
    # Need 2 tokens per key + 1 value per link = 3 per association
    # Plus 4 for the query section
    min_len = 3 * n_hops + 4
    if seq_len < min_len:
        seq_len = min_len

    input_ids = torch.randint(
        _VOCAB_LO, _VOCAB_HI, (batch_size, seq_len), dtype=torch.long
    )

    # Generate chain values: need n_hops+1 distinct values, and 2 key tokens per link
    # Total unique tokens needed: 2*n_hops (keys) + (n_hops+1) (chain values)
    n_unique = 2 * n_hops + n_chain_tokens
    targets = torch.zeros(batch_size, dtype=torch.long)

    usable_len = max(1, int(seq_len * 0.7) - 3 * n_hops)

    for i in range(batch_size):
        # Draw unique tokens
        perm = torch.randperm(_VOCAB_N)[:n_unique] + _VOCAB_LO
        keys = perm[: 2 * n_hops].reshape(n_hops, 2)  # (n_hops, 2)
        chain_vals = perm[2 * n_hops :]  # (n_hops+1,)

        # Spread associations evenly
        spacing = max(1, usable_len // n_hops)
        for hop in range(n_hops):
            pos = 1 + hop * spacing
            if pos + 2 >= seq_len - 4:
                pos = max(1, seq_len - 4 - 3 * (n_hops - hop))
            # Association: key1 key2 → chain_vals[hop+1]
            # But the "key" for this link is chain_vals[hop] mapped to key tokens
            input_ids[i, pos] = keys[hop, 0]
            input_ids[i, pos + 1] = keys[hop, 1]
            input_ids[i, pos + 2] = chain_vals[hop + 1]

            # Also plant the chain value → key mapping:
            # chain_vals[hop] appears before this link so the model can learn
            # that keys[hop] "represents" chain_vals[hop]
            if hop == 0:
                # First link: keys[0] maps from chain_vals[0] (the query entity)
                # Plant chain_vals[0] right before keys[0]
                if pos > 0:
                    input_ids[i, pos - 1] = chain_vals[0]
            # For subsequent hops, chain_vals[hop] is the output of the previous
            # link, which was already placed as chain_vals[hop] = output of hop-1

        # Query section at end: [SEP] [keys[0][0]] [keys[0][1]] [ANS]
        qstart = seq_len - 4
        input_ids[i, qstart] = _SEP_TOKEN
        input_ids[i, qstart + 1] = keys[0, 0]
        input_ids[i, qstart + 2] = keys[0, 1]
        input_ids[i, qstart + 3] = _ANS_TOKEN

        # Target: last value in the chain
        targets[i] = chain_vals[n_hops]

    return input_ids.to(device), targets.to(device)


def _generate_multi_hop_eval_set(
    n_eval: int,
    seq_len: int,
    n_hops: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed eval set with seed 42."""
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    try:
        return _generate_multi_hop_batch(n_eval, seq_len, n_hops, device)
    finally:
        torch.random.set_rng_state(rng_state)


def _eval_multi_hop_accuracy(
    model: nn.Module,
    eval_ids: torch.Tensor,
    eval_targets: torch.Tensor,
    batch_size: int,
) -> float:
    """Evaluate multi-hop retrieval accuracy."""
    model.eval()
    correct = 0
    total = eval_ids.shape[0]
    ans_pos = eval_ids.shape[1] - 1

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            inp = eval_ids[start:end]
            tgt = eval_targets[start:end]
            logits = model(inp)
            pred_logits = logits[:, ans_pos, _VOCAB_LO:_VOCAB_HI]
            preds = pred_logits.argmax(dim=-1) + _VOCAB_LO
            correct += (preds == tgt).sum().item()

    return correct / max(total, 1)


def _train_multi_hop_at_config(
    model: nn.Module,
    seq_len: int,
    n_hops: int,
    n_train_steps: int,
    n_eval: int,
    lr: float,
    batch_size: int,
    device: str,
    deadline: float,
) -> Tuple[float, bool]:
    """Micro-train a deepcopy on multi-hop retrieval for one config.

    Returns (accuracy, timed_out).
    """
    probe_model = copy.deepcopy(model)
    probe_model.to(device)
    probe_model.train()

    eval_ids, eval_targets = _generate_multi_hop_eval_set(
        n_eval, seq_len, n_hops, device
    )
    opt = make_adamw(probe_model.parameters(), lr=lr)
    ans_pos = seq_len - 1
    timed_out = False

    try:
        for step in range(1, n_train_steps + 1):
            if time.perf_counter() > deadline:
                timed_out = True
                break

            input_ids, targets = _generate_multi_hop_batch(
                batch_size, seq_len, n_hops, device
            )
            opt.zero_grad(set_to_none=True)
            logits = probe_model(input_ids)
            pred_logits = logits[:, ans_pos, _VOCAB_LO:_VOCAB_HI]
            loss = F.cross_entropy(pred_logits, targets - _VOCAB_LO)

            if not torch.isfinite(loss):
                break

            loss.backward()
            nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
            opt.step()

        acc = _eval_multi_hop_accuracy(probe_model, eval_ids, eval_targets, batch_size)
    finally:
        del eval_ids, eval_targets, probe_model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return acc, timed_out


def multi_hop_retrieval_score(
    model: nn.Module,
    seq_lens: tuple[int, ...] = (256, 512, 1024),
    hop_depths: tuple[int, ...] = (2, 3),
    n_train_steps: int = 500,
    n_eval: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> MultiHopResult:
    """Micro-train deepcopies on multi-hop retrieval.

    Score = weighted mean accuracy across (length, depth) configs.
    The original model is NOT modified — deepcopies are used.
    """
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    result = MultiHopResult()
    weighted_accs: List[Tuple[float, float]] = []

    for n_hops in sorted(hop_depths):
        for seq_len in sorted(seq_lens):
            if time.perf_counter() > deadline:
                result.status = "timeout"
                break

            config_key = f"{n_hops}hop_{seq_len}len"
            try:
                acc, timed_out = _train_multi_hop_at_config(
                    model,
                    seq_len,
                    n_hops,
                    n_train_steps,
                    n_eval,
                    lr,
                    batch_size,
                    device,
                    deadline,
                )
                result.per_config[config_key] = round(acc, 4)
                weighted_accs.append((float(n_hops), acc))
                if timed_out:
                    result.status = "timeout"
                    break
            except Exception as e:
                result.per_config[config_key] = 0.0
                weighted_accs.append((float(n_hops), 0.0))
                logger.debug("multi_hop: failed for %s: %s", config_key, e)

        if result.status == "timeout":
            break

    if weighted_accs:
        total_weight = sum(w for w, _ in weighted_accs)
        if total_weight > 0:
            result.score = round(sum(w * a for w, a in weighted_accs) / total_weight, 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
