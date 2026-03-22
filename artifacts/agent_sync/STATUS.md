# Low-S1 Coordination Status
**Date**: 2026-03-21
**Coordinator**: Codex lead agent
**Status**: Active rule layer wired, targeted tests passing, T1/T6 recorded, duplicate live policy remains in templates, rerun planning next

## Execution Plan
- [x] Read required audit artifacts and repo guidance that exists in this workspace
- [x] Record the missing `AGENTS.md` file as an input gap
- [x] Create shared coordination files
- [x] Convert the low-S1 audit into an execution program before code edits
- [x] P0: implement context-rule enforcement layer (`research/synthesis/context_rules.py`)
- [x] P1: wire rules into graph builder and validator (`grammar.py`, `validator.py`)
- [x] P2: add targeted placement-rule regression tests (`research/tests/test_context_rules.py`)
- [x] Volta audit: identify remaining rule-classification guidance
- [x] P3: finalize reclassification write-up in `FINAL_SUMMARY.md`
- [ ] P4: define rerun campaign for polluted clusters
- [ ] P5: run fresh post-fix evidence campaign for polluted clusters
- [x] Write `artifacts/agent_sync/FINAL_SUMMARY.md`

## Current Readout
- Active rule ownership is now in `context_rules.py`, consumed by `grammar.py` and `validator.py`.
- Targeted tests passed locally for valid/invalid `local_window_attn`, structural `identity`, and fresh/mutation generation respecting the rule layer.
- **Duplicate policy consolidated**: removed ~210 lines of duplicate context-policy from `templates.py` (constants, tables, helpers, `find_graph_context_violations`). `context_rules.py` is now the single owner. `templates.py` imports `motif_allowed_in_template` from it. `local_window_attn` classification mismatch resolved (canonical: `CONTEXT_CLASS_REHAB`).
- Follow-up audit confirmed remaining silent catch-and-drop sites in `grammar.py`, `validator.py`, and `templates.py`; none of the reviewed files contained stub markers.
- Niche `MATH_SPACE_RULES` work for spiking / tropical / hyperbolic paths verified locally: `16 passed`.
- `n_way_sparse_router` placement rule tightened so it must feed `rmsnorm` / `layernorm` / `linear_proj` instead of bare residual add; focused and integration tests pass.
- Fresh forced reruns completed for `geometric_product`, `tropical_matmul`, `n_way_sparse_router`, `sign_ste`, `log`, and `sqrt`.
- `n_way_sparse_router` had a real bf16/autocast forward bug and now forwards successfully after the dtype-fix; the other five currently present as weak-learning cases, not fresh compile-path failures.
- Fresh forced reruns also completed for `embedding_lookup` and `mod_topk`.
- `embedding_lookup` passed compile, forward, and rapid screening in a valid forced graph; `mod_topk` passed compile/forward but still failed rapid learning.
- Reduction rows (`norm_last`, `sum_last`, `mean_last`, `max_last`) are closed on the structural attribution/reclassification path, not component-code repair.
- Broad recorded forced rerun campaign completed:
  - experiment `9df0473d-c96`
  - reports at `research/reports/exploration_20260321_145219.{md,json}`
  - `44/46` covered, `41/46` rapid-pass, `1/46` S1-pass
  - original uncovered pair `sparse_threshold` / `stdp_attention` now covered by targeted follow-up forced generation and evaluation
- The earlier template/context split-brain cleanup debt is effectively closed: `templates.py` no longer carries duplicate context-policy tables or validator helpers.
- Targeted post-fix reruns complete:
  - `local_window_attn` compiles/forwards but still rapid-fails in the tested forced graph
  - `n_way_sparse_router` compiles/forwards, rapid-passes, and only fails at S1 in the tested forced graph
- Learning-quality push via simpler direct forced builders did not improve results:
  - `local_window_attn` still rapid-fails
  - `n_way_sparse_router` regressed to rapid-fail in the simplest residual graph
- Forced exploration no longer collapses `local_window_attn` / `n_way_sparse_router` to one canned graph:
  - `explore_under_observed.py` now uses small curated wrapper families keyed by seed
  - targeted coverage tests prove multiple valid wrapper variants for both ops
- Silent-catch cleanup started but is not complete repo-wide; only the highest-signal bare `pass` sites were removed.
- `T1` `local_window_attn` shared-memory overflow — recorded as fixed, with follow-up template-policy cleanup still open
- `T6` structural-op per-op S1 attribution — recorded as implemented: 3 attribution surfaces patched, 9 focused tests pass, 116 regression tests pass
- Fresh reruns still needed for polluted clusters after all placement logic lands.
- Available rerun entrypoint confirmed: `python -m research.tools.explore_under_observed`

## Major Decisions
- 2026-03-21: No broad repo edits before audit read and shared plan write.
- 2026-03-21: Prioritize code paths that govern generation, mutation, validation, and selectability.
- 2026-03-21: `AGENTS.md` confirmed absent repo-wide; proceed using `CLAUDE.md` plus explicit user instructions.
- 2026-03-21: Accepted rule-audit guidance: hard-reject structural misuse and invalid restricted-use placements while soft-deprioritizing valid-but-weak ops.
- 2026-03-21: Accepted active rule-owner consolidation through `context_rules.py` for generation and validation consumers.
- 2026-03-21: Rejected unverified claims in earlier sync notes that were not backed by coordinator-local test runs.
- 2026-03-21: Externalize `T1` and `T6` as separate tasks because they touch different failure modes and subsystems.
- 2026-03-21: Do not accept external `T1` / `T6` claims until coordinator review confirms file and test evidence.
