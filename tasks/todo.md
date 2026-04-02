---
status: active
created: 2026-04-01
author: claude-opus
---

# Scientist Hygiene Remediation

Full plan: `/home/tim/.claude/plans/jazzy-fluttering-book.md`

## Phase 0: Foundation
- [x] 0.1 Add SQLite indexes on program_results(timestamp, experiment_id+timestamp)
- [x] 0.2 Cache PRAGMA table_info in analytics_experiments.py

## Phase 1: Database & Caching
- [x] 1.1 Fix N+1 query in analytics_experiments.py _s1_stats()
- [x] 1.2 Cache get_reference_losses() with TTL
- [x] 1.3 Batch DB ops in execution_screening.py:713-719
- [x] 1.4 Add LIMIT to full-table scans in analytics_experiments.py
- [x] 1.5 Cache count_discovery_tiers()
- [x] 1.6 Cache gather_briefing_data()

## Phase 2: Code Deduplication
- [x] 2.1 Consolidate scoring v7/v8 → _compute_composite_generic()
- [x] 2.2 Autograd function factory in native_autograd.py (472→316 lines)
- [x] 2.3 Consolidate frontier constants (merged into config dicts)
- [x] 2.4 Graph dispatch dedup in native/dispatch.py
- [x] 2.5 LLM init dedup in persona*.py
- [x] 2.6 Op extraction consolidation (N/A — already uses mixin inheritance, no duplication)

## Phase 3: Dead Code & Hot Path
- [x] 3.1 Migrate callers + remove legacy compute_composite_score() (~457 lines deleted)
- [x] 3.2 Remove /api/observability/monitor endpoint (~140 lines deleted)
- [x] 3.3 Clean native_runner_canary.py MagicMock imports
- [x] 3.4 Label backfill functions as [MIGRATION TOOL]
- [x] 3.5 Fix duplicate mean in _helpers.py
- [x] 3.6 Single-pass seed aggregation in _helpers.py
- [x] 3.7 Fix double fingerprint call in synthesis.py
- [x] 3.8 Dashboard defaultdict

## Phase 4: Exception Handler Remediation (655 instances)
- [x] 4.1 Tier A: execution_screening.py (18 narrowed, 20 already ok)
- [x] 4.2 Tier A: execution_training.py (18 narrowed, 3 kept broad at error boundaries)
- [x] 4.3 Tier B: observability_bp.py (17 narrowed, 7 kept as route boundaries)
- [x] 4.4 Tier B: _helpers.py (17 narrowed) + dashboard.py (15 narrowed)
- [x] 4.5 Tier B: chat_bp.py (5 fixed, 14 already ok)
- [x] 4.6 Tier C: execution_candidates.py (18 narrowed, 5 kept as boundaries)
- [x] 4.7a Tier C: 11 runner files (113 narrowed, 9 kept as boundaries)
- [x] 4.7b Tier C: 11 notebook/analytics files (~46 narrowed)
- [x] 4.8 Tier D: 78 remaining files (~141 narrowed/logged)
**Result: 655→220 handlers remaining (435 fixed, 0 bare except-pass left)**

## Phase 5: Structural Performance
- [ ] 5.1 Profile + fix O(n^2) interaction scoring
- [ ] 5.2 Streaming .fetchall() replacements
- [ ] 5.3 Fix blocking HTTP in _designer.py

---

## Full Performance & Hygiene Audit (2026-04-01)
See: `tasks/audit/performance_hygiene_audit_2026-04-01.md` (report)
See: `tasks/audit/fix_plan.md` (prioritized fix plan, 7 phases, 40+ items)
