"""Graded Multi-Query Associative Recall (gMQAR) — a scale-sensitive capability probe.

WHY THIS EXISTS (see tasks/scaling_test_validation_plan.md, PRIMARY candidate):
the existing nano probes (indNear, AR-gate, nb05/nb10, ar_curriculum) are
single operating points — pass/fail "can this model bind", saturating at ceiling
on any capable nano model. They triage architectures but cannot *rank* a 100M vs
a 1.6B model. The real induction/binding/AR evals don't resolve at nano scale.

gMQAR is a synthetic in-context key->value recall task with *crankable
difficulty* (number of KV pairs, distance, distractor density). The current
zero-shot ladder resolves 30M+ checkpoints but floors on nano checkpoints, so use
it as a scaling/emergence probe rather than a nano lane-ranker. For nano models,
use the train-then-test bAbI probes that measure learnability after gradient
updates. indNear is the degenerate 1-pair case of this surface. Attention solves
gMQAR; many efficient mixers (linear attn, some SSMs) break as the pair count /
distance grows — which is exactly the discrimination "how well does this mixer
bind" that we want at scales where the task clears floor.

This is a MEASUREMENT TOOL, not an architecture — using it does not touch the
novel-architecture mandate. softmax_attention is the baseline to beat.

Scoring is IN-WEIGHTS-FREE: gMQAR is evaluated zero-shot on an already-trained
model by feeding KV pairs followed by queries in-context and reading the recalled
value at each query position. No fine-tuning, no gradient steps — so it is cheap
and the same harness runs from 27M to 1.6B.

Summary statistics over the difficulty grid:
- AUDC  : Area Under the Difficulty Curve (mean accuracy across the grid, 0..1).
          Single comparable scalar; higher = binds across more of the surface.
- D50   : the largest number of KV pairs still recalled at >= 50% accuracy
          ("breaking difficulty"). Integer; the headline "how deep can it bind".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn


# Special vocabulary layout (small, fixed ids at the bottom of the vocab):
#   0 = PAD, 1 = SEP (separates the KV block from the query block),
#   2 = QUERY marker (precedes each queried key in the query block).
# Keys and values are drawn from disjoint id ranges ABOVE these specials so the
# task is unambiguous and tokenizer-agnostic (synthetic ids only).
PAD_ID = 0
SEP_ID = 1
QRY_ID = 2
N_SPECIAL = 3


@dataclass(frozen=True, slots=True)
class GMQARConfig:
    """One difficulty cell. The *grid* is a list of these (see default_grid)."""

    vocab_size: int = 8192  # model output dim; must match the model's vocab
    n_pairs: int = 8  # KV pairs presented in context (difficulty knob)
    n_queries: int = 4  # how many of those keys are queried at the end
    distractor_tokens: int = 0  # random filler between KV block and queries
    batch_size: int = 64
    seed: int = 0
    # If >0, draw key/value/distractor ids only from [N_SPECIAL, N_SPECIAL+token_pool).
    # Restricts the task to a small, well-trained id range so gMQAR measures the
    # BINDING mechanism rather than embedding quality of rare tokens (matches
    # induction_probe.py's restricted-vocab convention). 0 = use the full vocab.
    token_pool: int = 0

    @property
    def effective_span(self) -> int:
        return self.token_pool if self.token_pool > 0 else (self.vocab_size - N_SPECIAL)

    def __post_init__(self) -> None:
        if self.n_queries > self.n_pairs:
            raise ValueError(f"n_queries ({self.n_queries}) > n_pairs ({self.n_pairs})")
        if self.token_pool and self.token_pool > self.vocab_size - N_SPECIAL:
            raise ValueError(
                f"token_pool {self.token_pool} exceeds usable vocab "
                f"{self.vocab_size - N_SPECIAL}"
            )
        if self.effective_span < 2 * self.n_pairs:
            raise ValueError(
                f"effective token span {self.effective_span} too small for "
                f"{self.n_pairs} disjoint key+value ids (need >= {2 * self.n_pairs})"
            )


def make_gmqar_batch(
    cfg: GMQARConfig,
    generator: torch.Generator,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one gMQAR batch.

    Sequence layout per row:
        [k0 v0 k1 v1 ... k{P-1} v{P-1}]  SEP  [distractors]  [QRY kq0 ? QRY kq1 ? ...]
    The model must predict, at each '?' position (right after 'QRY k'), the value
    that was bound to that key in the KV block.

    Returns
    -------
    input_ids : (B, S) long
    target_ids: (B, S) long   (-100 everywhere except the answer positions)
    answer_mask:(B, S) bool   (True exactly at the answer positions)

    Keys and values are sampled per-row from disjoint id pools so a value can
    never be confused with a key, and keys within a row are unique (each key
    binds exactly one value).
    """
    B, P, Q = cfg.batch_size, cfg.n_pairs, cfg.n_queries
    g = generator

    half = cfg.effective_span // 2
    # key pool:   [N_SPECIAL, N_SPECIAL+half)
    # value pool: [N_SPECIAL+half, N_SPECIAL+2*half)
    # With token_pool set, both stay inside the well-trained low-id range.
    key_lo, key_hi = N_SPECIAL, N_SPECIAL + half
    val_lo, val_hi = N_SPECIAL + half, N_SPECIAL + 2 * half

    seq_len = 2 * P + 1 + cfg.distractor_tokens + 3 * Q
    input_ids = torch.full((B, seq_len), PAD_ID, dtype=torch.long, device=device)
    target_ids = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
    answer_mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)

    # Sampling stays per-row and in this exact call order: the draw sequence must
    # stay bit-identical to the original per-row loop (gMQAR is the AR metric of
    # record; changing generator call shapes/order would shift every score).
    keys_rows, values_rows, distract_rows, qidx_rows = [], [], [], []
    for _ in range(B):
        keys_rows.append(_sample_unique(key_lo, key_hi, P, g, device))
        values_rows.append(
            torch.randint(val_lo, val_hi, (P,), generator=g, device=device)
        )
        if cfg.distractor_tokens:
            distract_rows.append(
                torch.randint(
                    key_lo, val_hi, (cfg.distractor_tokens,), generator=g, device=device
                )
            )
        qidx_rows.append(_sample_unique(0, P, Q, g, device))
    keys = torch.stack(keys_rows)  # (B, P)
    values = torch.stack(values_rows)  # (B, P)
    qidx = torch.stack(qidx_rows)  # (B, Q)

    # Assembly is fully vectorized: strided slice writes, no scalar kernel
    # launches, no .item() host syncs.
    kv_end = 2 * P
    input_ids[:, 0:kv_end:2] = keys
    input_ids[:, 1:kv_end:2] = values
    input_ids[:, kv_end] = SEP_ID
    if cfg.distractor_tokens:
        input_ids[:, kv_end + 1 : kv_end + 1 + cfg.distractor_tokens] = torch.stack(
            distract_rows
        )

    q0 = kv_end + 1 + cfg.distractor_tokens
    qkeys = keys.gather(1, qidx)
    qvals = values.gather(1, qidx)
    input_ids[:, q0::3] = QRY_ID
    input_ids[:, q0 + 1 :: 3] = qkeys
    # logits at the key position predict the NEXT token; teacher-force the answer.
    input_ids[:, q0 + 2 :: 3] = qvals
    answer_mask[:, q0 + 1 :: 3] = True
    target_ids[:, q0 + 1 :: 3] = qvals

    return input_ids, target_ids, answer_mask


def _sample_unique(
    lo: int, hi: int, k: int, g: torch.Generator, device: str
) -> torch.Tensor:
    """k unique ints in [lo, hi) via permutation (hi-lo guaranteed >= k by config)."""
    perm = torch.randperm(hi - lo, generator=g, device=device)[:k]
    return perm + lo


def _logits_from_model(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Match the repo convention: model(input_ids) -> logits or (logits, *aux)."""
    out = model(input_ids)
    if isinstance(out, tuple):
        out = out[0]
    return out


GMQARScoring = Literal["candidate", "full_vocab"]


def _candidate_values(input_ids: torch.Tensor, cfg: GMQARConfig) -> torch.Tensor:
    """Return the per-row in-context value candidates from the KV block."""
    return input_ids[:, 1 : 2 * cfg.n_pairs : 2]


@torch.no_grad()
def score_cell(
    model: nn.Module,
    cfg: GMQARConfig,
    device: str = "cpu",
    scoring: GMQARScoring = "candidate",
) -> float:
    """Zero-shot recall accuracy at the answer positions for one difficulty cell."""
    was_training = model.training
    model.eval()
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    input_ids, target_ids, answer_mask = make_gmqar_batch(cfg, g, device)
    logits = _logits_from_model(model, input_ids)
    # logits[:, t] predict token t+1; answer_mask marks position t whose NEXT
    # token is the value. By default, score associative recall over the row's
    # in-context values, not over the full LM vocabulary; otherwise frequent
    # non-value tokens can dominate the raw argmax and floor a model that ranks
    # the correct bound value highest among viable answers.
    if scoring == "candidate":
        answer_coords = answer_mask.nonzero(as_tuple=False)
        if answer_coords.numel() == 0:
            correct = 0
        else:
            row_idx = answer_coords[:, 0]
            row_candidates = _candidate_values(input_ids, cfg)[row_idx]
            answer_logits = logits[answer_mask]
            candidate_logits = answer_logits.gather(1, row_candidates)
            pred_idx = candidate_logits.argmax(dim=-1)
            preds = row_candidates[
                torch.arange(row_candidates.shape[0], device=row_candidates.device),
                pred_idx,
            ]
            correct = (preds == target_ids[answer_mask]).sum().item()
    elif scoring == "full_vocab":
        preds = logits.argmax(dim=-1)
        correct = (preds[answer_mask] == target_ids[answer_mask]).sum().item()
    else:
        raise ValueError(f"unknown gMQAR scoring mode: {scoring!r}")
    total = int(answer_mask.sum().item())
    if was_training:
        model.train()
    return correct / total if total else 0.0


@dataclass(slots=True)
class GMQARResult:
    cells: list[dict] = field(default_factory=list)  # [{n_pairs, distance, acc}]
    audc: float = 0.0  # mean accuracy across the grid (0..1)
    d50: int = 0  # largest n_pairs recalled at >= 50% accuracy
    chance: float = 0.0  # 1 / (#distinct values) approx, for reference


def default_grid(vocab_size: int = 8192, token_pool: int = 0) -> list[GMQARConfig]:
    """Difficulty ladder over KV-pair count and distractor distance.

    Pairs {2,4,8,16,32} crank associative load; distractors {0,128} crank
    distance/interference. Single seed per cell here (scorer is averaged over a
    batch of 64); callers can re-run with multiple seeds for CIs. token_pool
    restricts KV ids to a well-trained low-id range (see GMQARConfig); 0 = full vocab.
    """
    grid: list[GMQARConfig] = []
    for n_pairs in (2, 4, 8, 16, 32):
        for distract in (0, 128):
            grid.append(
                GMQARConfig(
                    vocab_size=vocab_size,
                    n_pairs=n_pairs,
                    n_queries=min(4, n_pairs),
                    distractor_tokens=distract,
                    token_pool=token_pool,
                )
            )
    return grid


def score_model_gmqar(
    model: nn.Module,
    grid: Optional[list[GMQARConfig]] = None,
    vocab_size: int = 8192,
    device: str = "cpu",
    token_pool: int = 0,
    scoring: GMQARScoring = "candidate",
) -> GMQARResult:
    """Run the full difficulty grid and summarise as AUDC + D50.

    AUDC = mean per-cell accuracy (area under the difficulty curve, normalised).
    D50  = max n_pairs whose accuracy (averaged over that pair-count's cells)
           is still >= 0.5 ("breaking difficulty").
    """
    if grid is None:
        grid = default_grid(vocab_size, token_pool=token_pool)
    cells: list[dict] = []
    for cfg in grid:
        acc = score_cell(model, cfg, device, scoring=scoring)
        cells.append(
            {
                "n_pairs": cfg.n_pairs,
                "distractor_tokens": cfg.distractor_tokens,
                "n_queries": cfg.n_queries,
                "acc": round(acc, 4),
                "chance": round(1.0 / cfg.n_pairs, 6)
                if scoring == "candidate"
                else round(1.0 / max(1, cfg.effective_span // 2), 6),
            }
        )
    audc = sum(c["acc"] for c in cells) / len(cells) if cells else 0.0

    # D50: group by n_pairs, average accuracy, take the largest passing >=0.5
    by_pairs: dict[int, list[float]] = {}
    for c in cells:
        by_pairs.setdefault(c["n_pairs"], []).append(c["acc"])
    d50 = 0
    for n_pairs, accs in sorted(by_pairs.items()):
        if sum(accs) / len(accs) >= 0.5:
            d50 = max(d50, n_pairs)

    # chance reference for the active scoring mode. Candidate-restricted scoring
    # has a per-cell chance of roughly 1/n_pairs; report the grid mean.
    if scoring == "candidate":
        chance = sum(c["chance"] for c in cells) / len(cells) if cells else 0.0
    else:
        span = grid[0].effective_span if grid else (vocab_size - N_SPECIAL)
        chance = 1.0 / max(1, span // 2)
    return GMQARResult(
        cells=cells, audc=round(audc, 4), d50=d50, chance=round(chance, 6)
    )
