# Session — scoring + reasoning probes overhaul (2026-05-02)

**Total wall**: ~6 hours (including iterations)
**Code shipped**: 3 new files, ~5 edits to 2 existing
**Diagnostic findings**: 6 actionable scoring problems, 3 probe bugs

## What's in production now

Code committed:

| file | change |
|---|---|
| `research/eval/selective_copy_probe.py` | NEW — Mamba-style most-recent-marker retrieval |
| `research/eval/compositional_probe.py` | NEW — train binary `(a+b) mod 16`, eval ternary |
| `research/tools/long_context_bundle.py` | NEW — multi-probe driver with champion-mode flags |
| `research/scientist/leaderboard_scoring.py` | EDIT — long-ctx + icld plumbing, template_name in family detector, per-signal bypass thresholds (ICLD lowered 0.20→0.030) |

## Champion-mode validation (the headline)

Ran 12-layer, 10K-step training + full probe bundle on b0c38826
(latent_compress_block) and 72769b4f (local_attn_ssm_hybrid):

| metric | b0c38826 | 72769b4f |
|---|---:|---:|
| final_loss | 0.001 | 0.001 |
| iv2 | **0.994** | 0.967 |
| bv2 | 0.921 | **0.999** |
| **passkey** | **0.333** | 0.000 |
| **compositional binary** | **0.867** | 0.078 |
| compositional ternary | 0.031 | 0.063 |
| selective_copy | 0.000 | 0.000 |
| multi_hop | 0.003 | 0.008 |

**Verdict**:
- Probes execute correctly at champion mode (no silent failures).
- Compositional probe is a *real discriminator* — 11× separation
  between the two architectures.
- Both architectures fail ternary composition (below chance) →
  empirical reasoning ceiling at this scale.
- selective_copy and multi_hop at noise floor → these probes need
  redesign or the task is fundamentally too hard for nano models.

## Scoring problems found (audit on val+bt rows, n≈2,400)

Spearman ρ vs `composite_score`:

| signal | weight | ρ | verdict |
|---|---:|---:|---|
| diagnostic_score | 40 | **+0.473** | strongest understanding signal — keep |
| logit_margin_velocity | 25 | +0.456 | strong driver — keep |
| iv2 | 50 | +0.421 | strong — keep |
| tinystories_score | 30 | +0.408 | strong — keep |
| erf_density | 25 | +0.268 | moderate — keep |
| bv2 | 50 | +0.189 | weak (small n) — keep but watch |
| **cross_task_score** | 30 | **−0.177** | **BROKEN** — credits balanced failure |
| hellaswag_acc | 15 | +0.088 | near-noise |
| **fp_hierarchy_fitness** | 25 | **−0.030** | **inverted** — high tree-likeness = failure |
| **blimp_overall_accuracy** | 35 | **+0.019** | **pure noise** — 35pt waste |

Plus three secondary issues:

- **v12 champion gate is over-restrictive**: rows with PPL<100 + good
  tinystories + good diagnostic but iv<0.05 get capped at 360. Most of the
  v13-promoted top entries are in this category.
- **Composite drift**: stored composite_scores diverge from current v12
  by up to ±150pts due to scoring-version migration without rescore.
- **trajectory probe leaves model in deepcopy-broken state** (root cause
  of 8e601381 silent-failure in earlier runs). Fixed by reordering probes
  so trajectory runs LAST.

## Three concrete proposals (NOT implemented — awaiting user decision)

### Proposal A: cross_task min-PPL gate

```python
def compute_cross_task_score(ppl_code, ppl_nl, threshold=200.0):
    """Only credit balance when both domains show actual learning."""
    if min(ppl_code, ppl_nl) > threshold:
        return 0.0  # both failed; no balance to credit
    return min(ppl_code, ppl_nl) / max(ppl_code, ppl_nl)
```

### Proposal B: v13 weight rebalance (validated by simulation)

| signal | v12 weight | v13 weight | Δ |
|---|---:|---:|---:|
| w_blimp | 35 | 5 | −30 |
| w_hellaswag | 15 | 5 | −10 |
| w_cross_task | 30 | 0 | −30 |
| w_hierarchy | 25 | 0 | −25 |
| w_tinystories | 30 | 50 | +20 |
| w_diagnostic | 40 | 60 | +20 |
| w_cap_logit_margin | 25 | 35 | +10 |

Net redirect: **−95pts from noise** → **+50pts to high-signal metrics**.
Total understanding-tier budget shrinks by ~45pts; the budget difference
flows naturally back through the loss-tier dominance.

Read-only simulation showed 65% of top-200 entries shift ≥5 ranks. New
top-3 are PPL=25, 63, 66 — genuinely good language models. Demoted
entries (RAG, adaptive_conv_ffn) had PPL > 600 with high cross_task
crediting balanced failure.

### Proposal C: well-learned-LM gate path

```python
def _is_well_learned_lm(kw):
    ppl = kw.get("ppl_validation") or kw.get("ppl_investigation") or kw.get("ppl_screening")
    ts  = kw.get("tinystories_score")
    diag = kw.get("diagnostic_score")
    return (
        ppl is not None and ppl < 100.0
        and ts is not None and ts >= 0.55
        and diag is not None and diag >= 0.01
    )
```

Champion gate becomes:
`qualifies = induction_qualified OR exception_allowed OR is_well_learned_LM`.

Without this, v13's better ranking gets capped at 360 for attention-class
good-language-modelers (the genuine winners under v13).

## Open follow-ups

| task | priority | ETA |
|---|---|---:|
| Apply proposal A (cross_task fix) | medium | 30 min |
| Apply proposal B (v13 weights) | medium | 15 min + leaderboard rescore |
| Apply proposal C (well-learned-LM gate) | medium | 15 min + rescore |
| Leaderboard rescore | high | 5 min (needs dashboard restart) |
| Redesign multi_hop / selective_copy | low | 1-2 days (probes underdiscriminate) |
| Investigate hierarchy_fitness inversion | low | 1 hr (decide drop vs invert) |

## Files

- `tasks/scoring_audit_2026-05-02.md` — detailed audit with tables
- `research/reports/long_ctx_bundle_champion_2inducers_v2/` — champion run output
- `research/reports/long_ctx_bundle_top10_v2/` — earlier 500-step bundle (for comparison)

## Cost summary

GPU: ~1h 40m
- Top-10 baseline (500 steps): 2 × 30 min = 60 min
- Champion 2-inducer (10K × 12 layers): 34 min
- Iteration overhead: ~6 min

DB writes: 0 (all read-only audits + JSON file outputs only)
