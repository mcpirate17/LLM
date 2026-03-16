# Aria Continuous Run — Maintenance Session

## Status: RUNNING (investigation cycle active!)

## Plan
- [x] Ground in codebase
- [x] Diagnose observability routes 404 — FIXED (dashboard restart)
- [x] Verify dashboard rendering and live plots — all working
- [x] Start continuous run — multiple cycles completed
- [x] Check insight quality — 22+ insights with Bayesian posteriors, used by grammar
- [x] Check fingerprint system — all S1 survivors fingerprinted, novelty ~0.72-0.74
- [x] **CRITICAL FIX: Auto-investigation gate unblocked** — composite score floor was inflated by reference architectures (232.91), making all synthesized graphs ineligible. Fixed to use screening 75th percentile (96.53) when no non-reference investigation entries exist.
- [ ] Monitor investigation cycle completion
- [ ] Verify investigation results promote to leaderboard correctly
- [ ] Continue monitoring

## Issues Found & Status
1. **Observability routes 404** — FIXED: Dashboard restart
2. **Auto-investigation completely blocked** — FIXED: `get_investigation_eligible()` used 25th percentile of investigation/validation tier for composite_score floor, but only reference architectures existed there (scores 229-239). Synthesized graphs max out at ~140. Now falls back to screening 75th percentile.
   - File: `research/scientist/notebook/notebook_misc.py:591-610`
   - Before: 0 eligible candidates
   - After: 17 eligible, top 15 scored and entering investigation (2500-step training)
3. **nm_sparse_linear + core ops flagged broken** — KNOWN: Low S0 rate is composition-dependent, not op-level failure. 100% S1 rate when S0 passes.
4. **DB lock transient** — KNOWN: SQLite WAL mode with 5s timeout. Self-resolves.
5. **Stale grammar refs** — LOW: Guarded by registry checks, harmless.

## Session Metrics
- ~50 experiments completed (8/hour)
- 77+ programs stored, 65+ S1 survivors
- Best loss ratio: 0.0133 (screening), 0.2341 (investigation pending)
- Investigation now active with 2500-step training runs
