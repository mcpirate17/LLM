# Composite scoring audit — 2026-05-02

Run on val+bt rows of the leaderboard (n≈2,400). Computes Spearman of each
metric against `composite_score` to identify which weights are actually
contributing to ranking quality vs. burning weight on noise.

## Headline: ~90 of 175 understanding-tier points are misallocated

| signal | v10 weight | Spearman ρ | spread (val+bt) | verdict |
|---|---:|---:|---:|---|
| **wikitext PPL (loss)** | 175 (tier) | **+0.477** | 33,431 | dominant driver — keep |
| **diagnostic_score** | 40 | **+0.473** | 0.10 | high signal despite tight range |
| **logit_margin_velocity** | 25 | **+0.456** | 0.28 | strong driver |
| **iv2** | 50 (v11/12) | **+0.421** | 1.00 | strong |
| **tinystories_score** | 30 | **+0.408** | 0.48 | strong |
| erf_density | 25 | +0.268 | 1.00 | moderate |
| bv2 | 50 (v11/12) | +0.189 | 1.00 | weak — but small n=588 |
| **cross_task_score** | 30 | **−0.177** | 0.84 | **negative — broken metric** |
| hellaswag_acc | 15 | +0.088 | 0.16 | near-noise |
| icld_velocity | 10 (aux) | −0.069 | 0.11 | noise |
| fp_hierarchy_fitness | 25 | −0.030 | 0.98 | **noise — full range, no signal** |
| **blimp_overall_accuracy** | 35 | **+0.019** | 0.09 | **pure noise, 35pt waste** |

## Three concrete scoring problems

### 1. BLiMP costs 35 pts, contributes nothing

ρ=0.019. p25–p90 spread is 0.025 absolute (around chance ≈ 0.5). Every
nano-LLM in the corpus scores BLiMP within a 2.5% band — no architecture is
distinguishably better at linguistic acceptability. **Recommend dropping
BLiMP weight from 35 → 5pt floor or removing entirely.**

### 2. cross_task_score is *anti-correlated* with quality

ρ=−0.177 vs composite. Drilling deeper:
- ρ=−0.389 vs `wikitext_perplexity (inv)` — high cross_task → high PPL
- ρ=−0.364 vs `tinystories_score` — high cross_task → bad TinyStories
- ρ=−0.272 vs `final_loss (inv)` — high cross_task → high final loss

Top-7 of 8 rows by `cross_task_score` have PPL > 1000 (vs cohort median
~750). The metric measures "balance between code and NL domain
perplexity" but doesn't gate on minimum competence. A model with PPL=2000
on both domains scores identical "robustness" to a model with PPL=20 on
both. **The metric credits failure-balance.**

Fix: add a minimum-PPL gate: `cross_task_score = ratio if min(ppl_code,
ppl_nl) <= threshold else 0`. Or remove the weight and rely on
wikitext PPL alone.

### 3. fp_hierarchy_fitness is *inverted* — high tree-likeness correlates with failure

ρ=−0.030 vs composite is the surface symptom. Drilling deeper, hierarchy is
significantly anti-correlated with capability and pro-correlated with
failure:

| signal | ρ vs hierarchy |
|---|---:|
| binding_v2_investigation_auc | **−0.343** |
| tinystories_score | **−0.219** |
| induction_v2_investigation_auc | −0.077 |
| final_loss | **+0.262** |
| wikitext_perplexity | +0.186 |
| param_count | +0.254 |

`hierarchy_fitness` measures Gromov δ-hyperbolicity of the model's hidden
state distance matrix (`research/eval/hierarchy_probe.py`) — high score =
"tree-like representations". In this corpus, tree-like representations are
**a marker of failure to learn**: models that don't develop specialized
representations preserve their initial high-hyperbolicity structure.

The current weight (+25 pts higher = better) is actively misaligned.
Two paths:
- **Invert sign**: `cap_hierarchy_anchor` rewards LOW hierarchy_fitness
- **Drop weight**: set w_hierarchy=0 until we understand whether
  hyperbolicity ever signals quality

The +0.254 correlation with param_count suggests this might just be a
"smaller models have less clustered representations" effect, not an
architectural-quality measure at all.

## Anchor calibration: solid

All 6 understanding/aux anchors equal the cohort median exactly:

| signal | v10 anchor | cohort med | match? |
|---|---:|---:|---|
| blimp_overall_accuracy | 0.525 | 0.525 | ✓ |
| hellaswag_acc | 0.225 | 0.225 | ✓ |
| tinystories_score | 0.542 | 0.542 | ✓ |
| cross_task_score | 0.279 | 0.279 | ✓ |
| diagnostic_score | 0.004 | 0.004 | ✓ |
| fp_hierarchy_fitness | 0.848 | 0.847 | ✓ |

The 2026-04-26 anchor recalibration is still accurate. The problem is not
mis-calibration — it's that BLiMP / HellaSwag / cross_task / hierarchy
provide little or wrong signal to begin with.

## Coverage gaps (not blocking)

| signal | leaderboard coverage |
|---|---:|
| BLiMP | 7,812 / 7,814 (100%) |
| HellaSwag | 7,813 / 7,814 (100%) |
| TinyStories | 4,406 / 7,814 (56.4%) |
| cross_task | 4,599 / 7,814 (58.9%) |
| diagnostic_score | 5,016 / 7,814 (64.2%) |
| fp_hierarchy_fitness | 6,843 / 7,814 (87.6%) |

3 leaderboard rows are fully missing cross_task + diagnostic + tinystories
+ hierarchy. Coverage gaps aren't the bottleneck for the scoring problems
above — those are *signal* problems, not *coverage* problems.

## Proposed v13 weight rebalance (DRAFT — not implemented)

Goal: shift weight from low-signal metrics to high-signal ones.

| signal | v12 weight | proposed v13 | rationale |
|---|---:|---:|---|
| w_perf_short/medium/long | 30/40/55 | 30/40/55 | unchanged (loss tier ρ=+0.48) |
| **w_blimp** | 35 | **5** | ρ=0.019 — basically a floor |
| **w_hellaswag** | 15 | **5** | ρ=0.088 |
| **w_cross_task** | 30 | **0 (or fix metric first)** | ρ=−0.177 |
| **w_hierarchy** | 25 | **0 (pending drill)** | ρ=−0.030 |
| w_tinystories | 30 | 35 | ρ=+0.408 |
| w_diagnostic | 40 | 50 | ρ=+0.473 |
| w_cap_induction (v11/v12) | 50 | 50 | ρ=+0.421 |
| w_cap_binding (v11/v12) | 50 | 50 | needs more measurements |
| w_cap_logit_margin | 25 | 35 | ρ=+0.456 |
| w_cap_erf_density | 25 | 25 | ρ=+0.268 |

Net change: **redirect ~95 weighted points** from BLiMP+HellaSwag+
cross_task+hierarchy → diagnostic+tinystories+logit_margin. Keeps
total budget approximately equal but concentrates weight on metrics
that actually rank rows correctly.

## Next steps (open)

1. **cross_task fix** — add minimum-PPL gate before crediting balance.
2. **hierarchy_fitness drill** — read `hierarchy_probe.py`, decide if the
   25pt weight is recoverable or should be retired.
3. **v13 dry-run** — implement weight changes behind a `_V13_CONFIG`,
   recompute composite for 200 sample rows, see how the leaderboard
   reorders. If the top entries stay top, the change is safe.
4. **Leaderboard rescore** — needed regardless to clear stored-vs-current
   v12 drift documented earlier today.

## v13 simulation result + gate interaction (2026-05-02 PM)

v13 weight simulation (read-only, no DB changes) on top-200 entries:
- 130/200 entries (65%) shifted ≥5 ranks
- New top-3: 727f5b33 (token_merge, PPL=63), cfd82b3a (token_merge, PPL=66),
  7ebfde81 (latent_attn_sparse_ffn, **PPL=25**)
- RAG (54531c6a): #3 → #34 (broken cross_task credit removed)
- adaptive_conv_ffn (8e601381): #7 → #200 (was scoring on noise weights)

**Validation**: the v13 new winners are genuinely good language models
(PPL 25–95). The v13 demoted entries had PPL > 600 with high cross_task
scores (cross_task crediting balanced failure).

**However: v12 champion gate blocks v13 from working as intended.** Of
the v13 top-15 entries, 9 fail the `iv >= 0.05` induction qualification:

| entry | template | PPL | iv | bv | gate verdict |
|---|---|---:|---:|---:|---|
| 727f5b33 | token_merge_block | 63 | 0.003 | 0.434 | NEEDS BYPASS |
| cfd82b3a | token_merge_block | 66 | 0.005 | 0.449 | NEEDS BYPASS |
| **7ebfde81** | latent_attn_sparse_ffn | **25** | 0.005 | 0.500 | NEEDS BYPASS |
| 7d506d25 | token_merge_block | 95 | 0.003 | 0.116 | NEEDS BYPASS |
| 974d65bb | conditional_compute | 87 | 0.005 | 0.008 | NEEDS BYPASS |
| 6b461a8c | sparse_ffn | 102 | 0.003 | 0.039 | NEEDS BYPASS |
| 552279f6 | latent_attn_sparse_ffn | 106 | 0.003 | 0.043 | NEEDS BYPASS |
| adb74bb9 | tropical_residual | 106 | 0.002 | 0.000 | NEEDS BYPASS |
| d1389569 | token_merge_block | 202 | 0.002 | 0.056 | NEEDS BYPASS |

These are **attention-family rows with strong LM but no induction** — the
v12 non-attention bypass doesn't apply to them, and they fail induction
qualification. Under both v12 and the simulated v13, the champion gate
caps them at 360 regardless of how good the underlying language model is.

**Proposed v13 gate change — add a well-learned-LM exception**:

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

Champion gate becomes: `qualifies = induction_qualified OR exception_allowed OR is_well_learned_LM`.

This credits attention architectures that genuinely learn language
modeling at this scale, even when they don't develop induction circuits.

## Champion-mode probe results (validation)

b0c38826 (latent_compress_block) re-evaluated at champion mode (n_layers=12,
train_steps=10000):
- iv2: 0.994 (matches stored 1.0)
- bv2: 0.921 (matches stored 1.0)
- passkey: **0.333** (was 0.000 at 500 steps — probes work at scale!)
- multi_hop: 0.003 (still near zero)
- selective_copy: 0.0078 (chance — needs different probe)
- **compositional_binary_acc: 0.867** (model learned a+b mod 16!)
- **compositional_ternary_acc: 0.031** (collapsed below chance — probe
  works as discriminator)
- compositional_score: 0.036 (= ternary/binary ratio)

**Compositional probe is functional**. At champion mode it produces a
clean signal: "this model learned the binary task but cannot compose to
ternary" — exactly the reasoning failure we set out to measure.
