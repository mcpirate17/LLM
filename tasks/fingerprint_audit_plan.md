# Fingerprint Decision Audit — Implementation Plan

Based on `audit_fingerprint_decision_management.md`, `audit_fingerprint_code_map.md`, and `audit_fingerprint_followup_tasks.md`.

## Work Streams

Three independent streams, each assignable to a separate Claude Code instance. File ownership is exclusive — no two streams touch the same file.

---

## Stream A: Replication & Aggregation (P0)
**Goal**: Replace single-run/best-run decisions with per-fingerprint replicated evidence.

### A1. Per-fingerprint replication aggregates
- **Files**: `research/scientist/notebook/notebook_programs.py`, `research/scientist/notebook/notebook_leaderboard.py`
- Add `get_fingerprint_aggregates(fingerprint: str) -> dict` to notebook_programs.py
  - Query all `program_results` rows sharing the same `graph_fingerprint`
  - Return `n_runs`, `loss_mean`, `loss_std`, `novelty_mean`, `novelty_std`, `best_vs_mean_gap`
- Add columns to leaderboard: `replication_n`, `replication_loss_mean`, `replication_loss_std`
- Update `upsert_leaderboard()` to populate replication columns when n_runs > 1
- **Validation**: Write test that creates 3 results with same fingerprint, verifies aggregates are correct

### A2. Use aggregates in leaderboard scoring
- **Files**: `research/scientist/leaderboard_scoring.py`
- Modify `compute_composite_score()` to:
  - Apply a `replication_confidence` multiplier: `min(1.0, sqrt(n_runs / 3))` — scores below 3 runs get dampened
  - Penalize high `best_vs_mean_gap` (indicates lucky outlier): penalty = `max(0, gap - 0.1) * 20`
  - Use `loss_mean` instead of `loss_ratio` when n_runs >= 3
- **Validation**: Backtest against existing leaderboard entries. Write test with mock entries.

### A3. Promotion evidence uses aggregates
- **Files**: `research/scientist/api_routes/_strategy_recommendations.py`
- Modify `promotion_evidence_for_entry()` to:
  - Report replication stats alongside raw scores
  - Flag entries with n_runs < 3 as "insufficient replication"
  - Use `loss_mean ± loss_std` in evidence summary instead of single `loss_ratio`
- **Validation**: Unit test that promotion evidence includes replication fields

### Estimated scope: ~400 lines changed across 4 files

---

## Stream B: Promotion Integrity & Escalation (P0)
**Goal**: Block promotion on incomplete evidence; close the novelty-evidence loophole.

### B1. Block promotion on incomplete fingerprints
- **Files**: `research/scientist/runner/results_auto_escalate_phase7.py`
- In `_auto_escalate_investigation()`:
  - Add hard gate: refuse auto-escalation to validation unless `fingerprint_completed_post_investigation = True`
  - Log (don't silently skip) entries blocked by this gate with reason
  - Add counter `blocked_incomplete_fingerprint` to escalation results
- Current code explicitly logs "not blocking" on incomplete novelty — reverse that policy
- **Validation**: Test that creates entry without completed fingerprint, verifies escalation is refused

### B2. Investigation must complete fingerprint before marking done
- **Files**: `research/scientist/runner/execution_investigation.py`, `research/scientist/runner/continuous_investigation.py`
- After investigation training completes and `_best_inv_model` exists:
  - If `complete_fingerprint_post_investigation()` fails, mark investigation as `investigation_fingerprint_incomplete` (new status) instead of `investigation_passed`
  - Retry fingerprint completion once before giving up
- Current code catches the exception silently — make failure visible
- **Validation**: Test that investigation with failed fingerprint does not get `investigation_passed`

### B3. Novelty scoring requires artifact CKA for validation tier
- **Files**: `research/scientist/runner/execution_validation.py`
- Before entering validation seed sweep:
  - Check `cka_source == "artifact"` on the entry's fingerprint
  - If not artifact-backed, run `complete_fingerprint_post_investigation()` with CKA enabled
  - If CKA still fails, proceed but cap novelty contribution to composite score at 50%
- **Validation**: Test that validation entries without artifact CKA get capped novelty

### Estimated scope: ~250 lines changed across 4 files

---

## Stream C: Refinement & Selection Intelligence (P1)
**Goal**: Feed completed fingerprint evidence back into generation/refinement; add peer comparison.

### C1. Use completed fingerprint features in refinement source selection
- **Files**: `research/scientist/runner/synthesis.py`
- In `_build_refinement_plan()`:
  - Fetch `fingerprint_json` for each candidate source from notebook
  - When available, use behavioral features (hierarchy_fitness, interaction patterns, CKA distances) to:
    - Prefer sources with high hierarchy_fitness (indicates structured representations)
    - Prefer sources with low CKA-vs-transformer (indicates novelty from reference)
    - Weight these alongside existing loss_ratio and novelty criteria
- In `_score_refinement_candidate()`:
  - Add `fingerprint_quality_bonus`: +0.1 if source had completed behavioral fingerprint
  - Add `behavioral_diversity_bonus`: reward mutations that shift in underexplored behavioral dimensions
- **Validation**: Offline replay comparing hit rate of old vs new refinement source scoring

### C2. Nearest-historical-peer comparison
- **Files**: `research/scientist/notebook/notebook_analytics.py`, `research/scientist/judgment.py`
- Add `get_nearest_peers(graph_fingerprint: str, n: int = 5) -> list` to notebook_analytics.py
  - Compute Jaccard similarity of op-sets between the target and all prior fingerprints
  - Return top-n peers with their `loss_ratio`, `novelty_score`, `tier`, `composite_score`
- In judgment.py `score_candidate()`:
  - Fetch nearest peers for each candidate
  - Add `peer_relative_score`: how does this candidate's expected quality compare to its nearest peers?
  - Penalize candidates whose op-set is similar to historically-failed peers
- **Validation**: Test that peer retrieval returns sensible results; measure correlation with S1 pass

### C3. Higher-resolution family features (replaces coarse buckets)
- **Files**: `research/scientist/notebook/notebook_analytics.py`
- Replace `_assign_fingerprint_bucket()` (which uses hand-labeled categories like "attention-heavy") with:
  - Op-category distribution vector (11 categories from grammar → 11-dim normalized vector)
  - Top-3 routing ops present (categorical feature)
  - Template signature (first template block's name if available from metadata)
- Update `get_fingerprint_buckets()` to return richer feature dicts
- **Validation**: Verify improved predictive power for S1 survival using logistic regression on new features vs old buckets

### Estimated scope: ~500 lines changed across 3 files

---

## Dependency Map

```
Stream A (Replication)          Stream B (Promotion)           Stream C (Refinement)
─────────────────────          ──────────────────────          ─────────────────────
notebook_programs.py           auto_escalate_phase7.py        synthesis.py
notebook_leaderboard.py        execution_investigation.py     notebook_analytics.py
leaderboard_scoring.py         continuous_investigation.py    judgment.py
_strategy_recommendations.py   execution_validation.py
```

**No file overlaps between streams.** All three can run fully in parallel.

Stream B depends on Stream A's replication data for B1's "n_runs" check, but B1 can use a simple COUNT query independently. The integration point is:
- After both A and B complete, verify that promotion logic uses both replication aggregates AND completed fingerprint requirements together.

## Execution Order

1. **All three streams start simultaneously** — no blocking dependencies
2. After each stream completes, run `pytest tests/ -x --tb=short` to verify no regressions
3. After all three complete, run integration verification:
   - Start a continuous run and verify:
     - Replication aggregates populate on re-evaluated fingerprints
     - Investigation fingerprints complete before escalation
     - Refinement sources prefer entries with completed fingerprints
4. P2 tasks (surrogate model, uncertainty ranking, constant calibration) are deferred — they depend on P0/P1 data flowing correctly first

## Per-Stream CLAUDE.md Additions

Each Claude Code instance should add to `.current_work.md`:
- **Stream A**: "Claimed: notebook_programs.py, notebook_leaderboard.py, leaderboard_scoring.py, _strategy_recommendations.py — fingerprint replication aggregates"
- **Stream B**: "Claimed: results_auto_escalate_phase7.py, execution_investigation.py, continuous_investigation.py, execution_validation.py — promotion integrity"
- **Stream C**: "Claimed: synthesis.py (refinement only), notebook_analytics.py, judgment.py — refinement intelligence"

## Validation Checklist
- [x] A1: `get_fingerprint_aggregates()` returns correct n_runs/mean/std
- [x] A2: Composite score dampens for n_runs < 3 (n=1 → 57.7% of n=3 score)
- [x] A3: Promotion evidence reports replication stats (adds `replicatedEvidence` check + `replication` summary)
- [ ] B1: Auto-escalation blocked without completed fingerprint
- [ ] B2: Investigation marks incomplete fingerprint visibly
- [ ] B3: Validation caps novelty without artifact CKA
- [x] C1: Refinement sources use behavioral features when available (hierarchy_fitness, cka_vs_transformer, fingerprint_quality_bonus in synthesis.py)
- [x] C2: Nearest-peer retrieval works and feeds judgment (get_nearest_peers + _score_peer_comparison scorer)
- [x] C3: Family features are higher-resolution than old buckets (op_category_distribution, top_routing_ops, template_signature)
