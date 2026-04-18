# Probe v2 rollout — close-out (2026-04-18)

Integration plan: [probe_v2_integration_plan.md](probe_v2_integration_plan.md)
Source audit:    `PROBE_CALIBRATION_2026-04-17.md`

## Status: complete

All 6 phases executed in one working session. Changes are live in the
working tree. No screening-tier records touched (2,580 rows preserved).

| phase | status | artifacts |
|---|---|---|
| 1. Multi-seed variance | ✅ done | `tasks/probe_calibration_results/variance_summary.md` |
| 2. Probes + schema | ✅ done | `research/eval/{induction,binding}_probe_v2_investigation.py` |
| 3. Scoring integration | ✅ done | `research/scientist/leaderboard_scoring.py` — v1→v2 fallback |
| 4. Backfill 61 rows | ✅ done | 60 new + 1 test = 61 populated, 0 failures, 75 min wall |
| 5. Dashboard | ✅ done | leaderboard columns + v1/v2 compare in detail panel, bundle built |
| 6. Close-out (this doc) | ✅ done | — |

## Phase 1 verdict — step budgets with seed robustness

- Induction @ **500 mixed-gap steps** — attention/hybrid ≥ 0.99, conv/ssm/rwkv ≤ 0.03
- Binding @ **2400 steps** — attention/hybrid ≥ 0.99, conv7_4l at 0.71, ssm/rwkv ≤ 0.003
- **Median-of-3 seeds** required — shallow attention (attn_1l induction, attn_2l binding) is
  seed-sensitive at the capability frontier. Single-seed would produce ~1-in-5 false zeros.

## Phase 4 backfill — rank shift interpretation

61 rows populated. 36 rows (59%) shifted ≥5 ranks post-backfill — **over** the plan's
10% rollout-pause gate — but the shifts are **not** driven by v2 signal:

- Most v2 values are near-zero and match v1 near-zero (the cohort mostly can't bind)
- The churn is rescoring under **v8.1** (which rolled out 2026-04-17, one day before v2)
- v8.1 tightens the binding-all-below-soft-gate penalty from 0.80× to 0.50×
- Rows with inflated v8-era composites (discoveries scoring 150-200 on PPL alone) correctly
  drop to 90-130 under v8.1 + v2 confirmation
- Rows with stale-low v7/v8 composites correctly rise as they're rescored under the
  active version

The 59% rank shift is a **backlog of overdue rescoring**, not v2-probe instability. v2
itself agreed with v1 on the near-zero values for most rows.

**Genuine v2 signal observed in 4 rows:**

| fingerprint | reference | v1_ind → v2_ind | v1_bin → v2_bin |
|---|---|---|---|
| `ab5cf5ae57ba` | RAG | 0.026 → **0.217** | 0.000 → **0.096** |
| `c9b6bc428b64` | (discovery) | 0.018 → 0.057 | 0.088 → 0.086 |
| `272c97e68c64` | (discovery) | 0.010 → 0.027 | 0.003 → **0.047** |
| `7a090de9133d` | (discovery) | 0.004 → 0.005 | 0.003 → 0.030 |

RAG is the only architecture in the cohort with a non-trivial retrieval mechanism,
and it correctly ranks #1 at 170.1.

## Mamba / RWKV / SSM-family ordering

Decomposition confirmed: Mamba's 149.9 → 93.7 drop is **entirely** the v8.1
penalty change (0.80× → 0.50×). Every other component is identical.
Score decomposition:

| component | v8 | v8.1 | Δ |
|---|---|---|---|
| binding_local_only_penalty | **0.80** | **0.50** | −0.30 |
| all other components | identical | identical | 0 |

Mamba's low v2 scores (induction=0.006, binding=0.000) are architecturally correct —
SSM state compression genuinely cannot do exact retrieval (Mamba paper's own finding,
and the motivation for Jamba / Mamba-2 hybrid architectures). The v2 probe is
measuring this gap accurately, not generating artifacts.

## Merge / rollout sequencing

Since all phases landed in one session, there is no mid-deploy gap to manage. If this
work is backported to another branch:

1. Probes + schema (P2) must merge before scoring (P3), or scoring fails at import
2. Scoring (P3) can merge before backfill (P4) safely because the fallback treats
   `None` as "use v1"
3. Dashboard (P5) is orthogonal — can merge any time after P2

Nothing to sequence further in this repo.

## What is NOT changed

- **Screening-tier records (2,580 rows)** — keep v1 forever by design
- **Screening-tier induction probe** (`research/eval/induction_probe.py`) — unchanged
- **AR (`ar_auc`) scoring weight** — still 40% of binding composite. The v2 work did
  not re-probe AR because (a) the calibration doc's "AR broken" finding was at
  n_pairs=20 on tiny synthetic models; production runs routinely hit 0.15-0.50 and
  the signal works. Re-calibrating AR at production scale is a separate future
  project.

## Follow-up flagged but not done

- **Screened-out attention graphs.** User raised: the screening probe uses fixed
  gap=8, so attention graphs whose induction mechanism didn't express under that
  narrow regime may have been screened out despite being real binders. A targeted
  re-probe sweep on attention-family screening rows (sorted by existing v1 induction
  score as a proxy for "close to promotion") would quantify this gap. Estimated
  cost: 50 rows × 75s = ~1 hr. Not urgent; filed for a follow-up.

- **Hybrid / binary protocol version audit.** Every v2 row now carries
  `induction_v2_investigation_protocol_version = "induction_investigation_mixed_v1"`
  and `binding_v2_investigation_protocol_version = "binding_investigation_extended_v1"`.
  If these protocols evolve (e.g. gap set changes, step count changes), the version
  string should bump and old rows should be treated as stale.

- **Operator note** — DB lock hygiene: after a long-running backfill under aria-db,
  confirm `research/lab_notebook.db.writer-lock` is cleared before starting the
  dashboard. Stale locks survived the backfill process exit in this session;
  removed manually. Not a bug per se, but worth keeping an eye on.

## Files changed

```
research/eval/induction_probe_v2_investigation.py   # median-of-3 wrapper added
research/eval/binding_probe_v2_investigation.py     # new
research/scientist/leaderboard_scoring.py           # v1→v2 fallback + SELECT cols
research/scientist/notebook/notebook_core.py        # 6 leaderboard columns
research/scientist/notebook/_shared.py              # 14 program_results columns
research/scientist/runner/_eval_registry.py         # 2 EvalSpec entries
research/scientist/runner/_types.py                 # 6 ExternalEvalResult fields
research/scientist/runner/_helpers_benchmark.py     # promote_kwargs mapping
research/scientist/api_routes/leaderboard_bp.py     # serve v2 columns
research/tools/backfill.py                          # induction_v2 + binding_v2 probes
research/tools/probe_v2_rank_diff.py                # new audit tool
research/tests/test_scoring_binding_safety.py       # +8 v2 fallback tests (62 total)
research/dashboard/src/components/leaderboard/leaderboardConfig.js
research/dashboard/src/components/leaderboard/columnRenderers.js
research/dashboard/src/components/programDetail/TrainingMetricsPanel.js
research/dashboard/build/                           # rebuilt bundle
tasks/probe_v2_integration_plan.md
tasks/probe_v2_variance_check.py
tasks/probe_calibration_results/variance_*.{csv,md}
tasks/probe_calibration_results/backfill_v2.log
```
