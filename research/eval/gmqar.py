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
from typing import Optional

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


@dataclass(frozen=True)
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

    for b in range(B):
        keys = _sample_unique(key_lo, key_hi, P, g, device)
        values = torch.randint(val_lo, val_hi, (P,), generator=g, device=device)

        pos = 0
        for i in range(P):
            input_ids[b, pos] = keys[i]
            input_ids[b, pos + 1] = values[i]
            pos += 2
        input_ids[b, pos] = SEP_ID
        pos += 1
        if cfg.distractor_tokens:
            distract = torch.randint(
                key_lo, val_hi, (cfg.distractor_tokens,), generator=g, device=device
            )
            input_ids[b, pos : pos + cfg.distractor_tokens] = distract
            pos += cfg.distractor_tokens

        qidx = _sample_unique(0, P, Q, g, device)
        for j in range(Q):
            ki = int(qidx[j].item())
            input_ids[b, pos] = QRY_ID
            input_ids[b, pos + 1] = keys[ki]
            ans_pos = pos + 1  # logits at this position predict the NEXT token
            answer_mask[b, ans_pos] = True
            target_ids[b, ans_pos] = values[ki]
            input_ids[b, pos + 2] = values[ki]  # teacher-force answer into context
            pos += 3

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


@torch.no_grad()
def score_cell(
    model: nn.Module,
    cfg: GMQARConfig,
    device: str = "cpu",
) -> float:
    """Zero-shot recall accuracy at the answer positions for one difficulty cell."""
    was_training = model.training
    model.eval()
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    input_ids, target_ids, answer_mask = make_gmqar_batch(cfg, g, device)
    logits = _logits_from_model(model, input_ids)
    # logits[:, t] predict token t+1; answer_mask marks position t whose NEXT
    # token is the value. Compare argmax(logits at mask) to target at mask.
    preds = logits.argmax(dim=-1)
    correct = (preds[answer_mask] == target_ids[answer_mask]).sum().item()
    total = int(answer_mask.sum().item())
    if was_training:
        model.train()
    return correct / total if total else 0.0


@dataclass
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
        acc = score_cell(model, cfg, device)
        cells.append(
            {
                "n_pairs": cfg.n_pairs,
                "distractor_tokens": cfg.distractor_tokens,
                "n_queries": cfg.n_queries,
                "acc": round(acc, 4),
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

    # chance reference: uniform over the value pool size actually used
    span = grid[0].effective_span if grid else (vocab_size - N_SPECIAL)
    chance = 1.0 / max(1, span // 2)
    return GMQARResult(
        cells=cells, audc=round(audc, 4), d50=d50, chance=round(chance, 6)
    )
