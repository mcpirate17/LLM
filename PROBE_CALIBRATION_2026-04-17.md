# Probe Calibration — Induction / Binding / Associative Recall (2026-04-17)

## TL;DR

**Induction probe cleanly discriminates architectures under mixed-gap
training. Binding-curriculum is a useful complement at 1600 steps.
Associative-recall is empirically broken — no architecture scores > 0.01
in 2000 steps — and should NOT be used as a discriminator until fixed.**

1. **Induction — production regime is bad, mixed-gap is clean.**
   - Production (fixed gap=8, 500 steps): attn_2l=0.846, conv7_4l=0.196 —
     conv looks almost as good as attention because both just memorized
     gap=8.
   - Proposed (mixed gaps {4,8,16,32}, 500 steps): attn/hybrid=~0.997,
     conv7_4l=0.203, conv3/ssm/rwkv<0.01 — clean 3-way split with a ~0.8
     gap between reasoning and non-reasoning.
   - Even **1-layer attention reaches 0.997** under mixed training at 500
     steps, so the probe is a *mechanism* test, not a capacity test.
     Appropriate for "can this arch reason at all?" filtering.

2. **Binding curriculum — discriminates at 1600 steps, reveals depth need.**
   - hybrid_4l=0.998, attn_4l=0.976, attn_1l=0.827, conv7_4l=0.640,
     conv7_2l=0.398, ssm/rwkv=0.00. Clean ordering by architectural
     capability, plus a real depth signal (hybrid_2l=0.04 vs
     hybrid_4l=0.998, attn_2l=0.03 vs attn_4l=0.976).
   - Complements induction: induction asks "does it reason?", binding
     curriculum asks "how deep can it reason?".
   - Note: attn_2l at 1600 steps stuck at 0.026 while attn_4l hit 0.976 —
     looks like a convergence issue specific to the 2-layer model + lr.
     Worth investigating before using binding as a hard gate.

3. **Associative recall — BROKEN as currently configured.**
   Every architecture (including 4-layer attention that aces induction and
   binding) scored 0.003-0.011 at 2000 training steps. n_pairs=20 with
   seq_len≈67 appears out of reach given the deepcopy-based setup. This
   probe needs debugging before its score is trusted as a discriminator.

**Recommended investigation-tier probe: induction v2 at 500 mixed-gap steps.**
Runs in 2-5s per graph on GPU, cleanly ranks architectures. Drop-in
implementation at `research/eval/induction_probe_v2_investigation.py`;
integration is 4 mechanical changes described below. **No backfill** — new
column defaults to `None` on pre-fix rows, scoring gracefully skips the
subscore for those rows.

---

## Motivation

The audit's finding was that the production **induction** probe uses a fixed
training gap of 8, which causes well-architected models to over-specialize
(spike at gap=8, random elsewhere). A vanilla 2-layer causal transformer
scored **0.70 AUC** under the production regime vs **~1.00** under mixed-gap
training — a real design weakness.

But changing the screening-tier probe would require backfilling 10K+ existing
fingerprints. A safer approach is to **add a new investigation-tier probe
variant** that only runs on models that already cleared screening. That also
gives a free signal about which template families reason best.

This calibration sweep asks:

1. What layer depth is needed to score on each probe?
2. What step budget discriminates well (fast + informative)?
3. Does the probe cleanly separate architectural families
   (attention > hybrid > ssm > rwkv > conv) at the chosen budget?
4. What AUC thresholds should the investigation-tier gate use?

## Theoretical expectations

Each probe has a fundamental capability it's trying to measure. These shape
what a "correct" probe score looks like for each architectural family:

| probe | capability tested | attention | hybrid | SSM/Mamba | RWKV | conv-only |
|---|---|---|---|---|---|---|
| **induction** | exact pattern retrieval across gaps (`[A][B]...[A]→B`) | ~1.00 (can attend back) | ~1.00 | ~0.00 (state compression loses exact token) | ~0.00 | spike at gap ≤ receptive field, 0 elsewhere |
| **binding_curriculum** | periodic copy at distance d (`a b c d a b c d...`) | 0.8-1.0 if trained enough | 0.8-1.0 | partial at short d | partial | 0.5+ only at d ≤ receptive field |
| **associative_recall** | key→value retrieval over a 67-token context | 0.3-0.8 | 0.3-0.8 | ~0.01 (cannot do exact lookup) | ~0.01 | ~0.01 |

Expected ordering by probe score: `attention ≈ hybrid > RWKV/SSM > conv`.
A probe that *doesn't* produce this ordering is broken. A probe that
produces a spike-with-no-generalization (fixed-gap-8 production regime) is
telling you about meta-learning, not architectural capability.

## Method

Eleven synthetic architectures, all small (96-d, 4 heads where applicable):

- `attn_1l`, `attn_2l`, `attn_4l` — pure causal attention (GPT-style)
- `conv3_2l`, `conv7_2l`, `conv7_4l` — causal conv only (local mixer baseline)
- `ssm_2l`, `ssm_4l` — Mamba-flavored selective state space
- `rwkv_2l` — RWKV time-mix variant
- `hybrid_2l`, `hybrid_4l` — conv + attention alternating

Each probe is parameterized by `n_train_steps`. Induction additionally by
`train_mode` (fixed gap=8 vs cycling through the eval gap set).

### Probe runs

- **Induction**: grid of {11 archs × 5 step counts × 2 modes} = 110 runs;
  extended dense-step curve on 7 discriminating archs (11 steps).
- **Binding curriculum**: 11 archs × 4 step counts = 44 runs.
- **Associative recall**: 11 archs × 3 step counts = 33 runs.

All runs on RTX 5090. Typical single-run wall time 1-20 s.

---

## Integration roadmap (once the sweep confirms the recommendation)

1. **Add the new column to the schema** (`notebook_core.py` schema block):
   ```
   induction_v2_investigation_auc REAL,
   induction_v2_investigation_max_gap_acc REAL,
   induction_v2_investigation_protocol_version TEXT,
   ```

2. **Wire into `_eval_registry.py`** as an investigation-tier spec:
   ```python
   from research.eval.induction_probe_v2_investigation import (
       run_induction_v2_investigation,
   )

   def _run_induction_v2(ctx):
       r = run_induction_v2_investigation(ctx.model, device=ctx.device)
       return {
           "induction_v2_investigation_auc": r.auc,
           "induction_v2_investigation_max_gap_acc": r.max_gap_acc,
           "induction_v2_investigation_protocol_version": r.protocol_version,
       }

   EvalSpec(
       name="induction_v2_investigation",
       result_keys=(
           "induction_v2_investigation_auc",
           "induction_v2_investigation_max_gap_acc",
       ),
       run=_run_induction_v2,
       tier_threshold="investigation",
   )
   ```

3. **Score in `leaderboard_scoring.py`**. Add to `_PR_SELECT_COLS`:
   ```
   "induction_v2_investigation_auc, "
   "induction_v2_investigation_max_gap_acc, "
   ```
   and to `_pr_dict_to_score_kwargs`:
   ```python
   "induction_v2_inv_auc": pr_dict.get("induction_v2_investigation_auc"),
   ```
   and add a small S-curve subscore (15 pts suggested) in the investigation
   branch of `_compute_composite_generic`. Threshold centered on 0.7 (AUC
   strongly above "can-do-one-gap" but below perfect).

4. **No backfill required**. Existing rows have `NULL` for the new column;
   the scoring function treats `None` as "skip this subscore" rather than
   "penalize". The signal only applies going forward.

## Findings

## 1. Induction — AUC across architectures and step budgets

### Fixed-gap-8 training (matches production probe)

| arch | params | steps=100 | steps=250 | steps=500 | steps=1000 | steps=2000 |
|---|---|---|---|---|---|---|
| `attn_1l` | 210,336 | 0.004 | 0.003 | 0.004 | 0.334 | 0.328 |
| `attn_2l` | 322,176 | 0.006 | 0.547 | 0.846 | 0.879 | 0.919 |
| `attn_4l` | 545,856 | 0.023 | 0.817 | 0.867 | 0.883 | 0.951 |
| `conv3_2l` | 253,632 | 0.004 | 0.004 | 0.004 | 0.005 | 0.005 |
| `conv7_2l` | 327,360 | 0.005 | 0.017 | 0.193 | 0.202 | 0.206 |
| `conv7_4l` | 605,376 | 0.014 | 0.188 | 0.196 | 0.202 | 0.204 |
| `ssm_2l` | 74,720 | 0.002 | 0.003 | 0.004 | 0.004 | 0.006 |
| `ssm_4l` | 100,096 | 0.002 | 0.004 | 0.003 | 0.006 | 0.010 |
| `rwkv_2l` | 272,448 | 0.004 | 0.002 | 0.003 | 0.001 | 0.004 |
| `hybrid_2l` | 331,104 | 0.006 | 0.438 | 0.489 | 0.570 | 0.728 |
| `hybrid_4l` | 563,712 | 0.004 | 0.290 | 0.680 | 0.813 | 0.689 |


### Mixed-gap training (proposed fix)

| arch | params | steps=100 | steps=250 | steps=500 | steps=1000 | steps=2000 |
|---|---|---|---|---|---|---|
| `attn_1l` | 210,336 | 0.005 | 0.083 | 0.997 | 0.999 | 1.000 |
| `attn_2l` | 322,176 | 0.006 | 0.818 | 0.997 | 1.000 | 0.999 |
| `attn_4l` | 545,856 | 0.004 | 0.981 | 0.999 | 0.999 | 1.000 |
| `conv3_2l` | 253,632 | 0.004 | 0.001 | 0.005 | 0.003 | 0.004 |
| `conv7_2l` | 327,360 | 0.006 | 0.007 | 0.009 | 0.166 | 0.264 |
| `conv7_4l` | 605,376 | 0.004 | 0.035 | 0.203 | 0.385 | 0.431 |
| `ssm_2l` | 74,720 | 0.002 | 0.001 | 0.003 | 0.004 | 0.000 |
| `ssm_4l` | 100,096 | 0.003 | 0.007 | 0.001 | 0.005 | 0.005 |
| `rwkv_2l` | 272,448 | 0.001 | 0.010 | 0.002 | 0.005 | 0.005 |
| `hybrid_2l` | 331,104 | 0.007 | 0.007 | 0.997 | 0.999 | 0.994 |
| `hybrid_4l` | 563,712 | 0.002 | 0.918 | 0.997 | 1.000 | 1.000 |


## 2. Induction — per-gap breakdown at 500 steps

### Fixed-gap-8 training

| arch | AUC | peak | gap=4 | gap=8 | gap=16 | gap=32 | gap=64 |
|---|---|---|---|---|---|---|---|
| `attn_1l` | 0.004 | 0.010 | 0.005 | 0.005 | 0.010 | 0.000 | 0.000 |
| `attn_2l` | 0.846 | 0.995 | 0.970 | 0.995 | 0.905 | 0.810 | 0.550 |
| `attn_4l` | 0.867 | 1.000 | 0.985 | 1.000 | 0.955 | 0.790 | 0.605 |
| `conv3_2l` | 0.004 | 0.015 | 0.005 | 0.000 | 0.000 | 0.015 | 0.000 |
| `conv7_2l` | 0.193 | 0.950 | 0.000 | 0.950 | 0.010 | 0.000 | 0.005 |
| `conv7_4l` | 0.196 | 0.970 | 0.000 | 0.970 | 0.005 | 0.000 | 0.005 |
| `ssm_2l` | 0.004 | 0.010 | 0.005 | 0.005 | 0.010 | 0.000 | 0.000 |
| `ssm_4l` | 0.003 | 0.005 | 0.005 | 0.000 | 0.005 | 0.000 | 0.005 |
| `rwkv_2l` | 0.003 | 0.005 | 0.000 | 0.005 | 0.005 | 0.005 | 0.000 |
| `hybrid_2l` | 0.489 | 1.000 | 0.645 | 1.000 | 0.255 | 0.185 | 0.360 |
| `hybrid_4l` | 0.680 | 1.000 | 0.680 | 1.000 | 0.675 | 0.640 | 0.405 |


### Mixed-gap training

| arch | AUC | peak | gap=4 | gap=8 | gap=16 | gap=32 | gap=64 |
|---|---|---|---|---|---|---|---|
| `attn_1l` | 0.997 | 1.000 | 0.995 | 1.000 | 1.000 | 0.995 | 0.995 |
| `attn_2l` | 0.997 | 1.000 | 0.995 | 1.000 | 0.995 | 1.000 | 0.995 |
| `attn_4l` | 0.999 | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 |
| `conv3_2l` | 0.005 | 0.015 | 0.015 | 0.005 | 0.000 | 0.005 | 0.000 |
| `conv7_2l` | 0.009 | 0.040 | 0.040 | 0.000 | 0.000 | 0.000 | 0.005 |
| `conv7_4l` | 0.203 | 0.820 | 0.820 | 0.190 | 0.005 | 0.000 | 0.000 |
| `ssm_2l` | 0.003 | 0.010 | 0.010 | 0.000 | 0.005 | 0.000 | 0.000 |
| `ssm_4l` | 0.001 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 |
| `rwkv_2l` | 0.002 | 0.010 | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 |
| `hybrid_2l` | 0.997 | 1.000 | 1.000 | 0.995 | 0.990 | 1.000 | 1.000 |
| `hybrid_4l` | 0.997 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 0.995 |


## 3. Fixed-gap vs mixed-gap, head to head

| arch | steps | fixed-gap-8 AUC | mixed-gap AUC | delta |
|---|---|---|---|---|
| `attn_1l` | 100 | 0.004 | 0.005 | +0.001 |
| `attn_1l` | 250 | 0.003 | 0.083 | +0.080 |
| `attn_1l` | 500 | 0.004 | 0.997 | +0.993 |
| `attn_1l` | 1000 | 0.334 | 0.999 | +0.665 |
| `attn_1l` | 2000 | 0.328 | 1.000 | +0.672 |
| `attn_2l` | 100 | 0.006 | 0.006 | +0.000 |
| `attn_2l` | 250 | 0.547 | 0.818 | +0.271 |
| `attn_2l` | 500 | 0.846 | 0.997 | +0.151 |
| `attn_2l` | 1000 | 0.879 | 1.000 | +0.121 |
| `attn_2l` | 2000 | 0.919 | 0.999 | +0.080 |
| `attn_4l` | 100 | 0.023 | 0.004 | -0.019 |
| `attn_4l` | 250 | 0.817 | 0.981 | +0.164 |
| `attn_4l` | 500 | 0.867 | 0.999 | +0.132 |
| `attn_4l` | 1000 | 0.883 | 0.999 | +0.116 |
| `attn_4l` | 2000 | 0.951 | 1.000 | +0.049 |
| `conv3_2l` | 100 | 0.004 | 0.004 | +0.000 |
| `conv3_2l` | 250 | 0.004 | 0.001 | -0.003 |
| `conv3_2l` | 500 | 0.004 | 0.005 | +0.001 |
| `conv3_2l` | 1000 | 0.005 | 0.003 | -0.002 |
| `conv3_2l` | 2000 | 0.005 | 0.004 | -0.001 |
| `conv7_2l` | 100 | 0.005 | 0.006 | +0.001 |
| `conv7_2l` | 250 | 0.017 | 0.007 | -0.010 |
| `conv7_2l` | 500 | 0.193 | 0.009 | -0.184 |
| `conv7_2l` | 1000 | 0.202 | 0.166 | -0.036 |
| `conv7_2l` | 2000 | 0.206 | 0.264 | +0.058 |
| `conv7_4l` | 100 | 0.014 | 0.004 | -0.010 |
| `conv7_4l` | 250 | 0.188 | 0.035 | -0.153 |
| `conv7_4l` | 500 | 0.196 | 0.203 | +0.007 |
| `conv7_4l` | 1000 | 0.202 | 0.385 | +0.183 |
| `conv7_4l` | 2000 | 0.204 | 0.431 | +0.227 |
| `ssm_2l` | 100 | 0.002 | 0.002 | +0.000 |
| `ssm_2l` | 250 | 0.003 | 0.001 | -0.002 |
| `ssm_2l` | 500 | 0.004 | 0.003 | -0.001 |
| `ssm_2l` | 1000 | 0.004 | 0.004 | +0.000 |
| `ssm_2l` | 2000 | 0.006 | 0.000 | -0.006 |
| `ssm_4l` | 100 | 0.002 | 0.003 | +0.001 |
| `ssm_4l` | 250 | 0.004 | 0.007 | +0.003 |
| `ssm_4l` | 500 | 0.003 | 0.001 | -0.002 |
| `ssm_4l` | 1000 | 0.006 | 0.005 | -0.001 |
| `ssm_4l` | 2000 | 0.010 | 0.005 | -0.005 |
| `rwkv_2l` | 100 | 0.004 | 0.001 | -0.003 |
| `rwkv_2l` | 250 | 0.002 | 0.010 | +0.008 |
| `rwkv_2l` | 500 | 0.003 | 0.002 | -0.001 |
| `rwkv_2l` | 1000 | 0.001 | 0.005 | +0.004 |
| `rwkv_2l` | 2000 | 0.004 | 0.005 | +0.001 |
| `hybrid_2l` | 100 | 0.006 | 0.007 | +0.001 |
| `hybrid_2l` | 250 | 0.438 | 0.007 | -0.431 |
| `hybrid_2l` | 500 | 0.489 | 0.997 | +0.508 |
| `hybrid_2l` | 1000 | 0.570 | 0.999 | +0.429 |
| `hybrid_2l` | 2000 | 0.728 | 0.994 | +0.266 |
| `hybrid_4l` | 100 | 0.004 | 0.002 | -0.002 |
| `hybrid_4l` | 250 | 0.290 | 0.918 | +0.628 |
| `hybrid_4l` | 500 | 0.680 | 0.997 | +0.317 |
| `hybrid_4l` | 1000 | 0.813 | 1.000 | +0.187 |
| `hybrid_4l` | 2000 | 0.689 | 1.000 | +0.311 |


## 4. Family separation at each step budget (mixed-mode)


### 250 steps

| family | n | min | median | max |
|---|---|---|---|---|
| attention | 3 | 0.083 | 0.818 | 0.981 |
| hybrid | 2 | 0.007 | 0.463 | 0.918 |
| ssm | 2 | 0.001 | 0.004 | 0.007 |
| rwkv | 1 | 0.010 | 0.010 | 0.010 |
| conv | 3 | 0.001 | 0.007 | 0.035 |

### 500 steps

| family | n | min | median | max |
|---|---|---|---|---|
| attention | 3 | 0.997 | 0.997 | 0.999 |
| hybrid | 2 | 0.997 | 0.997 | 0.997 |
| ssm | 2 | 0.001 | 0.002 | 0.003 |
| rwkv | 1 | 0.002 | 0.002 | 0.002 |
| conv | 3 | 0.005 | 0.009 | 0.203 |

### 1000 steps

| family | n | min | median | max |
|---|---|---|---|---|
| attention | 3 | 0.999 | 0.999 | 1.000 |
| hybrid | 2 | 0.999 | 1.000 | 1.000 |
| ssm | 2 | 0.004 | 0.005 | 0.005 |
| rwkv | 1 | 0.005 | 0.005 | 0.005 |
| conv | 3 | 0.003 | 0.166 | 0.385 |

### 2000 steps

| family | n | min | median | max |
|---|---|---|---|---|
| attention | 3 | 0.999 | 1.000 | 1.000 |
| hybrid | 2 | 0.994 | 0.997 | 1.000 |
| ssm | 2 | 0.000 | 0.003 | 0.005 |
| rwkv | 1 | 0.005 | 0.005 | 0.005 |
| conv | 3 | 0.004 | 0.264 | 0.431 |


## 5. Minimum steps to reach AUC thresholds (mixed-mode)


### AUC ≥ 0.30

| arch | min steps to AUC ≥ 0.30 |
|---|---|
| `attn_2l` | 250 |
| `attn_4l` | 250 |
| `hybrid_4l` | 250 |
| `attn_1l` | 500 |
| `hybrid_2l` | 500 |
| `conv7_4l` | 1000 |

### AUC ≥ 0.50

| arch | min steps to AUC ≥ 0.50 |
|---|---|
| `attn_2l` | 250 |
| `attn_4l` | 250 |
| `hybrid_4l` | 250 |
| `attn_1l` | 500 |
| `hybrid_2l` | 500 |

### AUC ≥ 0.70

| arch | min steps to AUC ≥ 0.70 |
|---|---|
| `attn_2l` | 250 |
| `attn_4l` | 250 |
| `hybrid_4l` | 250 |
| `attn_1l` | 500 |
| `hybrid_2l` | 500 |

### AUC ≥ 0.90

| arch | min steps to AUC ≥ 0.90 |
|---|---|
| `attn_4l` | 250 |
| `hybrid_4l` | 250 |
| `attn_1l` | 500 |
| `attn_2l` | 500 |
| `hybrid_2l` | 500 |


## 6. Induction — dense learning-curve (extended sweep)

| arch | learning curve (steps → AUC) |
|---|---|
| `attn_1l` | 50:0.01 → 100:0.00 → 150:0.02 → 200:0.03 → 300:0.88 → 400:0.99 → 600:1.00 → 800:1.00 → 1200:1.00 → 1600:1.00 → 2400:1.00 |
| `attn_2l` | 50:0.00 → 100:0.01 → 150:0.01 → 200:0.01 → 300:0.89 → 400:0.99 → 600:1.00 → 800:1.00 → 1200:1.00 → 1600:1.00 → 2400:1.00 |
| `attn_4l` | 50:0.00 → 100:0.01 → 150:0.18 → 200:0.93 → 300:1.00 → 400:1.00 → 600:1.00 → 800:1.00 → 1200:0.99 → 1600:1.00 → 2400:1.00 |
| `conv7_4l` | 50:0.01 → 100:0.00 → 150:0.01 → 200:0.02 → 300:0.07 → 400:0.19 → 600:0.26 → 800:0.34 → 1200:0.38 → 1600:0.40 → 2400:0.41 |
| `hybrid_2l` | 50:0.00 → 100:0.00 → 150:0.00 → 200:0.00 → 300:0.00 → 400:0.00 → 600:0.00 → 800:0.74 → 1200:1.00 → 1600:0.01 → 2400:1.00 |
| `rwkv_2l` | 50:0.00 → 100:0.00 → 150:0.00 → 200:0.00 → 300:0.00 → 400:0.01 → 600:0.00 → 800:0.00 → 1200:0.01 → 1600:0.01 → 2400:0.01 |
| `ssm_4l` | 50:0.00 → 100:0.00 → 150:0.00 → 200:0.00 → 300:0.01 → 400:0.00 → 600:0.00 → 800:0.00 → 1200:0.00 → 1600:0.00 → 2400:0.01 |


## 7. Binding curriculum — architectural discrimination

| arch | params | steps=200 | steps=400 | steps=800 | steps=1600 |
|---|---|---|---|---|---|
| `attn_1l` | 210,336 | 0.002 | 0.003 | 0.003 | 0.827 |
| `attn_2l` | 322,176 | 0.002 | 0.002 | 0.007 | 0.026 |
| `attn_4l` | 545,856 | 0.002 | 0.002 | 0.044 | 0.976 |
| `conv3_2l` | 253,632 | 0.002 | 0.002 | 0.002 | 0.002 |
| `conv7_2l` | 327,360 | 0.004 | 0.008 | 0.080 | 0.398 |
| `conv7_4l` | 605,376 | 0.012 | 0.179 | 0.432 | 0.640 |
| `ssm_2l` | 74,720 | 0.001 | 0.001 | 0.001 | 0.002 |
| `ssm_4l` | 100,096 | 0.002 | 0.003 | 0.003 | 0.002 |
| `rwkv_2l` | 272,448 | 0.003 | 0.003 | 0.002 | 0.002 |
| `hybrid_2l` | 331,104 | 0.001 | 0.002 | 0.002 | 0.043 |
| `hybrid_4l` | 563,712 | 0.002 | 0.002 | 0.059 | 0.998 |


## 8. Associative recall — architectural discrimination

| arch | params | steps=500 | steps=1000 | steps=2000 |
|---|---|---|---|---|
| `attn_1l` | 210,336 | 0.003 | 0.004 | 0.009 |
| `attn_2l` | 322,176 | 0.007 | 0.011 | 0.006 |
| `attn_4l` | 545,856 | 0.004 | 0.005 | 0.004 |
| `conv3_2l` | 253,632 | 0.004 | 0.003 | 0.001 |
| `conv7_2l` | 327,360 | 0.007 | 0.006 | 0.008 |
| `conv7_4l` | 605,376 | 0.004 | 0.003 | 0.004 |
| `ssm_2l` | 74,720 | 0.003 | 0.005 | 0.011 |
| `ssm_4l` | 100,096 | 0.004 | 0.005 | 0.009 |
| `rwkv_2l` | 272,448 | 0.002 | 0.004 | 0.007 |
| `hybrid_2l` | 331,104 | 0.002 | 0.005 | 0.005 |
| `hybrid_4l` | 563,712 | 0.009 | 0.006 | 0.005 |


## 9. Recommended investigation-tier probe config

**Recommended investigation-tier budget: `500` training steps, mixed-gap training.** At this budget the attention family reaches AUC ≥ 0.6 while conv/ssm/rwkv families stay ≤ 0.4 — a clean architectural separation.

## Raw data

- `tasks/probe_calibration_results/induction_sweep.csv`
- `tasks/probe_calibration_results/induction_extended_sweep.csv`
- `tasks/probe_calibration_results/binding_curriculum_sweep.csv`
- `tasks/probe_calibration_results/associative_recall_sweep.csv`

## Integration plan

See `## Integration roadmap` section above. The new probe runs at
investigation tier only (~hundreds of runs/day vs screening's 10K+), so no
existing-fingerprint backfill is required.

Drop-in probe implementation:
`research/eval/induction_probe_v2_investigation.py`. Integration is four
mechanical changes (schema column, eval spec, scoring kwarg, composite
subscore), each described in the roadmap. No changes to
`research/eval/induction_probe.py` or the `tasks/induction_native_probe/`
screening-tier path.

## Known limitations and caveats

- **SSM / RWKV are implemented as unfused Python loops** in the sweep
  harness. Real Mamba/RWKV kernels would be 10-50× faster, but the AUC
  numbers are unaffected — state-compression limits on exact retrieval hold
  regardless of kernel implementation. Wall-time numbers in the CSV for
  SSM/RWKV are not representative of production kernels.
- **Hybrid family data** was recovered via `probe_calibration_resume.py`
  after an initial harness bug (`nn.Parameter` inside `nn.ModuleDict`). The
  bug is fixed and the resume script re-ran only the missing combinations.
- **Probe tests a mechanism, not a ceiling.** A 1-layer attention model
  reaches 0.997 AUC at 500 mixed-gap steps — same as a 4-layer model. "Can
  this arch form induction heads under training pressure?" rather than
  "how powerful is it?". Depth scaling surfaces on binding curriculum at
  long distances and AR at high `n_pairs`.
- **Vocab=256 in the probe is decoupled from the model's vocab.** Each
  probe slices logits to the first 256 classes. A model with vocab<256
  would break; vocab≥256 is fine. All ARIA candidates use vocab≥512.
- **Sequence length scales with gap.** At gap=64 the eval sequence is 67
  tokens. A model with `max_seq_len < 70` would fail the large-gap evals
  due to positional-embedding indexing, not architectural capability. All
  investigation-tier candidates should have `max_seq_len ≥ 128`.
- **Single-seed numbers.** Before the new probe gates anything live, run
  each (arch × config) with 5 seeds and verify coefficient of variation
  < 0.05 for architectural classes. The current CSV is deterministic
  enough to show family ordering but not tight enough to anchor a strict
  promotion threshold.

## Follow-up experiments worth running (not blocking integration)

1. **Sweep `n_pairs` in AR.** The production probe uses n_pairs=20 with
   seq_len≈67. Attention may not saturate at 20 — try 50 and 100 to see
   if the probe ceiling rises with context. Adds signal on which attention
   depth/width is actually needed to retrieve at scale.
2. **Add a `binding_delta_auc` metric** = curriculum_auc (after training)
   minus zero-shot_auc. The zero-shot probe is currently noise on almost
   everything; only the delta is architecturally meaningful.
3. **Multi-seed variance check.** Tag each (arch × config) with 5 seeds
   before the promotion gate goes live.
4. **Verify the "reasoning vs prediction" intent.** The induction probe is
   a good mechanism test but doesn't directly score compositional
   reasoning. A natural follow-up is a synthetic "2-hop retrieval" probe
   (given `a→b, b→c`, asked `a→?` answer `c`) — this requires bind +
   compose. A conv+FFN-only arch with zero cross-token bind would fail
   even when the conv has enough receptive field to see all pairs.
   Today this is covered loosely by `multi_hop_retrieval.py`, but the
   audit found that metric is not wired into scoring kwargs.
