# Probe v2 Integration Plan (2026-04-18)

**Source audit:** `PROBE_CALIBRATION_2026-04-17.md`
**Scope:** investigation + validation tiers only. Screening untouched.

## Affected rows (2026-04-18)

| tier | rows | backfill cost (est.) |
|---|---|---|
| screening | 2,580 | **NOT TOUCHED** — keeps legacy fixed-gap induction |
| investigation | 39 | ~5 min (induction @ 500 + binding @ 2400) |
| validation | 22 | ~3 min |
| **backfill total** | **61** | **< 10 min of GPU** |

No screening-tier backfill. No legacy records invalidated.

---

## Design decisions

### D1. New columns, not replacements

v2 probes go in **new columns**. v1 columns keep their historical values. Rationale:
- Audit trail — can always A/B compare v1 vs v2 on the same model
- Screening-tier stays on v1 (fixed-gap) forever: its job is cheap early filtering, not reasoning discrimination
- Investigation/validation tiers get v2 for the composite

Columns to add:
```
induction_v2_investigation_auc           REAL
induction_v2_investigation_max_gap_acc   REAL
induction_v2_investigation_protocol_ver  TEXT
binding_v2_investigation_auc             REAL
binding_v2_investigation_max_d_acc       REAL
binding_v2_investigation_protocol_ver    TEXT
```

### D2. Step budgets + seed robustness

**Phase 1 variance check (2026-04-18) verdict:** budgets hold, but probe must
run **median-of-3 seeds** because shallow attention is seed-sensitive at the
mechanism-forming threshold. See `tasks/probe_calibration_results/variance_summary.md`.
Single-seed measurements would flag real architectures as broken once in ~5 runs
(attn_1l induction seed failure: 0.044 while 4 other seeds ≥ 0.82; attn_2l
binding seed failure: 0.0050 while 4 others ≥ 0.995). Median-of-3 collapses both.

Cost impact:
- Induction per fingerprint: 2–5s → 6–15s
- Binding per fingerprint: 20–60s → 60–180s
- 61-row backfill: ~10min → ~80min (still trivial)



**Induction @ 500 mixed-gap steps.** The doc's table 4 shows:
- 500 steps: attention 0.997 / conv7_4l 0.203 / ssm/rwkv < 0.01 → **AUC gap 0.8**
- 1000 steps: attention 0.999 / conv7_4l 0.385 → **gap narrows to 0.6**
- 2000 steps: attention 1.000 / conv7_4l 0.431 → **gap narrows to 0.57**

More steps ≠ better. Past 500, conv starts memorizing enough of the distribution to
close the gap without developing an induction mechanism. 500 is the **discrimination
maximum**, not a compute shortcut. Cost: 2-5s/graph on GPU.

**Binding @ 2400 steps (was 1600 in doc).** The 1600-step data showed convergence
anomalies (`attn_2l=0.026`, `hybrid_2l=0.043`) that look lr/seed-specific rather than
architectural. Before using binding as a hard gate we need to either:
- Push to 2400-3200 steps so slow-convergence models catch up, OR
- Sweep lr (1e-3 → 3e-4) on 2L variants to rule out training instability
Plan runs both in Phase 1 (variance check) and picks the budget where CoV < 0.05.

### D3. Scoring integration — v2 slots into existing binding composite

Current binding composite (`leaderboard_scoring.py:672-688`):
```
bc = 0.4*ar_auc + 0.3*induction_auc + 0.3*binding_auc
binding_pts = w_binding * scurve(bc / cfg["binding"], k=6)
```

**New behavior:** for investigation/validation rows, read `induction_v2_auc` and
`binding_v2_auc` when present, fall back to v1 otherwise:
```
eff_induction = induction_v2_inv_auc if not None else induction_auc
eff_binding   = binding_v2_inv_auc   if not None else binding_auc
```

- Same weights, same composite, **no new subscore branch** — v2 is a *drop-in
  replacement signal* for already-promoted rows
- Fallback path keeps pre-backfill rows scoring correctly (no composite regression
  mid-backfill)
- `ar_auc` unchanged (different probe — see AR findings note below)

**`ar_auc` stays on v1** for now. The calibration doc's "AR is broken" finding was at
`n_pairs=20, seq_len=67, tiny 4L models`. Production-size candidates get 0.15-0.50
routinely (per `test_scoring_binding_safety.py` fixtures). Re-calibrating AR is a
separate investigation (Phase 7, stretch).

### D4. Dashboard surfacing

- **Leaderboard grid** (`leaderboardConfig.js`): add two compact columns
  `Ind v2`, `Bind v2`. Hide when `tier=screening`.
- **Column renderers** (`columnRenderers.js`): format 3 decimals, color-code
  at 0.3 / 0.7 thresholds.
- **Per-fingerprint detail panel**: show v1 vs v2 side-by-side with a
  "Δ vs v1" badge. Useful for the v2 rollout audit.
- **Template observability** (`TemplateSlotObservability.js`): already tracks
  AR/induction/binding — extend to v2 columns when present.

---

## Process (phased rollout)

### Phase 1 — Verify (before any code change)
- Multi-seed variance check: 11 archs × {induction@500, binding@2400} × 5 seeds
- Confirm family ordering holds, CoV < 0.05
- Freeze step budgets based on results
- **Gate:** if CoV > 0.1 for attention family, re-tune before proceeding

### Phase 2 — Probes + schema (no scoring impact)
- Confirm `research/eval/induction_probe_v2_investigation.py` exists per doc
- Write `research/eval/binding_probe_v2_investigation.py` (parallel structure)
- Schema migration in `notebook_core.py` — 6 columns, all NULL-able
- Wire both into `_eval_registry.py` as investigation-tier `EvalSpec`
- Unit tests: probe determinism at fixed seed, NULL-safety in scoring

### Phase 3 — Scoring integration (no behavior change yet)
- Add v2 kwargs to `_compute_composite_generic` signature
- Implement v1→v2 fallback in binding composite
- Extend `_PR_SELECT_COLS` + `_pr_dict_to_score_kwargs` in `leaderboard_scoring.py`
- Parallel tests in `test_scoring_binding_safety.py`:
  - v2 None → scores identical to v1-only
  - v2 present → v2 value dominates
  - v2 + v1 both present → v2 wins
  - Mixed-row leaderboard → no ranking discontinuity

### Phase 4 — Backfill (61 rows)
- Script: `research/tools/backfill_probe_v2.py`
  - Query investigation + validation rows
  - Load model via `compile_model` from fingerprint
  - Run both v2 probes, write to new columns
  - Idempotent: skip rows already populated
  - Checkpointed: resume on interrupt
- Dry-run first: `--dry-run` prints model ids without touching DB
- **Run during merge freeze window** to avoid racing active investigation runs

### Phase 5 — Dashboard
- Add columns to `leaderboardConfig.js` + `columnRenderers.js`
- Fingerprint detail panel v1/v2 comparison
- `npm run build` → commit `dashboard/build/`
- Restart dashboard server

### Phase 6 — Rollout sequencing
1. Merge P1-P2 → deploy → probes run on *new* investigation/validation promotions
2. Run P4 backfill
3. **After backfill completes**, merge P3 (scoring change) → recompute composites
4. Merge P5 dashboard
5. Announce in commit/changelog: what changed, how to read the new columns

Merging P3 **before** P4 would cause a live scoring regression on un-backfilled rows.
Merging P3 **after** P4 guarantees every investigation/validation row has v2 available
before the composite starts reading from it.

### Phase 7 (stretch) — AR probe re-calibration
- Repeat the doc's sweep with production-sized models (≥128d, ≥4 layers)
- If AR at production scale also flatlines, design `ar_v2`
- Otherwise, document the calibration-harness caveat and close

---

## Risks & mitigations

| risk | mitigation |
|---|---|
| P3 merged before P4 completes | explicit sequencing above; P3 PR description blocks merge until backfill green |
| v2 probe diverges from v1 by > 0.3 on most rows → rank churn | Phase 4 produces a before/after rank-diff report; pause if >10% of rows move ≥5 ranks |
| Binding convergence still flaky at 2400 steps | Phase 1 catches this; bump budget or add lr sweep before locking |
| Backfill hits an un-loadable model fingerprint | skip + log; investigate post-run; these exist on screening rows, rare on investigation+ |
| Dashboard build not refreshed | add to rollout checklist; `rm -rf build && npm run build` |

---

## Files to touch (checklist)

Schema & probes:
- [ ] `research/scientist/notebook/notebook_core.py` (schema)
- [ ] `research/scientist/notebook/_shared.py` (column list)
- [ ] `research/eval/induction_probe_v2_investigation.py` (drop-in if doc's file exists)
- [ ] `research/eval/binding_probe_v2_investigation.py` (new)
- [ ] `research/scientist/runner/_eval_registry.py` (two EvalSpec entries)
- [ ] `research/scientist/runner/_types.py` (EvalResult dataclass)
- [ ] `research/scientist/runner/_helpers.py` (result → notebook field mapping)

Scoring:
- [ ] `research/scientist/leaderboard_scoring.py` (composite + select cols + kwargs)
- [ ] `research/tests/test_scoring_binding_safety.py` (parallel v2 cases)
- [ ] `research/tests/test_scoring_v8.py` (smoke)

Backfill:
- [ ] `research/tools/backfill_probe_v2.py` (new)

Dashboard:
- [ ] `research/dashboard/src/components/leaderboard/leaderboardConfig.js`
- [ ] `research/dashboard/src/components/leaderboard/columnRenderers.js`
- [ ] `research/dashboard/src/components/TemplateSlotObservability.js`
- [ ] `research/scientist/api_routes/leaderboard_bp.py` (SELECT cols)
