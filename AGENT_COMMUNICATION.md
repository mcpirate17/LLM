# Agent Coordination & Project Hephaestus Status

This file serves as a live synchronization point for all AI agents (Gemini, Claude, Codex) working on the LLM workspace. **Check this file before every task.**

## Current Agent Assignments

| Agent | Area of Focus | Active Files/Directories | Status |
| :--- | :--- | :--- | :--- |
| **Gemini** | **Project Hephaestus** (Native Graph Engine, Proactive Gating, Adaptive Synthesis) | `aria-core/`, `research/scientist/runner.py`, `research/synthesis/` | **PROJECT COMPLETED** |
| **Claude** | **Gate & Investigation Improvements** (Fingerprints to Investigation) | `research/eval/`, `research/scientist/runner.py` | Refinement of Fingerprint Promotion |
| **Codex** | **UI/UX + Knowledge Quality** (Dashboard visuals, KB curation) | `research/dashboard/src/`, `research/scientist/runner.py`, `research/scientist/llm/context.py`, `research/tools/curate_knowledge_base.py` | Table/detail UX + knowledge generation quality filters + KB clustering |

---

## Conflict Zones (Watch Files)

- `research/scientist/runner.py`: High collision risk between Gemini (Native Gating) and Claude (Investigation Gates).
- `aria-designer/runtime/bridge.py`: Coordination required for new native telemetry.

## Communication Protocol

1. **Check:** Read this file at the start of every session.
2. **Update:** If you start a new sub-task or change files, update the "Active Files" column.
3. **Notify:** If you introduce a breaking change or a new native dependency, log it in the "Notices" section below.

---

## Project Hephaestus: Native Gating & Adaptive Synthesis (COMPLETED)

**Objective:** Transition Aria's decision-making from reactive Python filtering to a high-performance Rust/C++/Cython proactive engine.

### Phase 1: Native Graph Engine (COMPLETED)
- [x] Initialize C++ module for `aria-core-graph` in `aria-core`.
- [x] Implement fast subgraph isomorphism for toxic motif detection.
- [x] Implement path-walking for stability heuristics (normalization/residual checks).
- [x] Integrate `proactive_gating` into `research/scientist/runner.py`.

### Phase 2: Proactive Stability Gates (COMPLETED)
- [x] Implement "Shadow Mode" logic in `runner.py` via `_native_proactive_gating`.
- [x] Logic verified via `aria-core/tests/test_proactive_gating.py`.

### Phase 3: Behavioral Dry-Run (COMPLETED)
- [x] Implement native CKA kernels (`linear_cka_f32`) for high-performance similarity scoring.
- [x] Implement `compute_lightning_fingerprint` in `research/eval/fingerprint.py` for pre-S1 novelty gating.
- [x] Parity verified via `aria-core/tests/test_cka_parity.py` (17x speedup over PyTorch).

### Phase 4: Adaptive Synthesis (COMPLETED)
- [x] Implement `EfficiencyPrior` to extract Pareto frontier biases from `ExperimentAnalytics`.
- [x] Implement `AdaptiveGenerator` in `grammar.py` with real-time Look-Ahead Budget Pruning (FLOPs/Params).
- [x] Optimize parameter estimation with Cython (`adaptive_sampler.pyx`).
- [x] Integrate into `runner.py` evaluation loop.

---

## Notices

- **2026-02-28 (Gemini):** Project Hephaestus is fully operational. Aria now makes intelligent decisions *before* testing a fingerprint or making an experiment by:
    1. Pruning over-budget or inefficient architectures during synthesis (Phase 4).
    2. Gating unstable/toxic topologies using native C++ analysis (Phase 1 & 2).
    3. Rapidly estimating behavioral novelty via lightning dry-runs (Phase 3).
- **2026-02-28 (Codex):** Added KB quality-gating in `research/scientist/runner.py` and curated `knowledge_base` entries (`34` archived as `archived_low_value`).
- **2026-02-28 (Codex):** Added capped validation-weighted KB confidence in `notebook.py`, LLM KB context trimming (6 entries hypothesis / 5 campaign) in `llm/context.py`, and strict curation pass (additional `37` archived).
- **2026-02-28 (Codex):** Implemented Knowledge Base clustering UI (`research/dashboard/src/components/KnowledgeBase.js`) with high-signal default filtering, validation-weighted cluster ranking, expandable cluster drill-down, and compact digest copy; added cluster styling in `research/dashboard/src/App.css`.
- **2026-02-28 (Codex):** Tightened `build_campaign_report_context` to use capped `_select_knowledge_for_llm(..., limit=6)` instead of raw top-10 knowledge entries.
- **2026-02-28 (Codex):** Added semantic dedupe (token-overlap + Jaccard) to KB ingestion in `research/scientist/runner.py` and LLM context selection in `research/scientist/llm/context.py` to prevent near-duplicate insight variants from inflating context.
- **2026-02-28 (Codex):** Extended `research/tools/curate_knowledge_base.py` to consolidate weak-signal thematic duplicates; applied curation and archived 4 redundant high-validation variants, leaving one representative per duplicate theme (`active=493`, `archived_low_value=75`).
- **2026-02-28 (Codex):** Performed KB visual cleanup for long-scroll fatigue in `research/dashboard/src/components/KnowledgeBase.js` + `App.css`: stronger semantic clustering, hide-singletons toggle (default hidden), cluster pagination (`Load More`), and summary text clamping for compact scanning.
- **2026-02-28 (Codex):** Follow-up KB UX compaction: default visible clusters reduced to 8 and category accordion sections added (top 2 expanded by default) to reduce vertical sprawl in Knowledge Base.
- **2026-02-28 (Codex):** Fixed metric display fallback mismatches in dashboard tables/details: `Discoveries.js` now shows Discovery Loss from `discovery_loss_ratio` fallback to `screening_loss_ratio`; `ProgramDetail.js` reference comparison now resolves candidate loss ratio via fallback chain (`validation -> investigation/screening -> loss_ratio`) instead of showing false `--`.
- **2026-02-28 (Codex):** Fixed leaderboard/program metric merge edge case in `notebook.py#get_leaderboard`: raw `program_results` validation/discovery ratios are now backfilled into response when leaderboard phase fields are null (prevents false `--` in Discoveries table). Also adjusted spectral displays (`Discoveries.js`, `ProgramDetail.js`) to treat `<=0` as missing instead of rendering misleading `0.0000`.
- **2026-02-28 (Codex):** Root-caused misleading Discovery/Validation ratios: `runner.py` survivor-recording path was overwriting measured `discovery_loss_ratio` / `validation_loss_ratio` with baseline-comparison ratios. Patched to store baseline comparisons in separate keys (`discovery_baseline_ratio`, `validation_baseline_ratio`) and preserve measured ratios. Backfilled existing `program_results` ratios from raw losses (`loss / initial_loss`) in `lab_notebook.db`.
- **2026-02-28 (Codex):** Tightened Discovery score formula in `dashboard/src/utils/scoringEngine.js` to reduce low-quality/high-novelty inflation: nonlinear performance weighting, novelty gated by performance, and explicit penalties for poor validation/investigation evidence (e.g., validation LR >= 1, baseline ratio > 1, weak investigation LR).
- **2026-02-28 (Codex):** Added Discovery quality-floor UX in `dashboard/src/components/Discoveries.js` (default ON): hides low-quality candidates with best loss > 0.8, includes toggle to show all, and displays hidden-count indicator.
- **2026-02-28 (Codex):** Fixed experiment-level novelty propagation in `runner.py` (investigation/validation inline + threaded paths): `results.best_novelty_score` now updates from source candidate novelty so Experiments tab novelty KPI can populate. Backfilled `experiments.best_novelty_score` from `program_results` max novelty for existing rows.
- **2026-02-28 (Codex):** Cleaned stale validation runs in `lab_notebook.db` that were stuck as `running` with no active progress; only one current `running` experiment remains (`synthesis`, started 2026-02-28 17:38:53 UTC).
- **2026-02-28 (Codex):** Patched architecture drawer/designer bridge mismatch diagnostics: `ArchitectureDrawer.js` now requests graph snapshots with explicit reasons (`integrity`, `export`, `commit`) and only commits on explicit `commit`; `aria-designer/ui/src/App.jsx` now echoes `reason/requestId` in `graph-data` responses to prevent accidental commits during integrity checks.
- **2026-02-28 (Codex):** Purged stale toxic-pattern data from `failure_signatures` across local notebook DBs (`research/lab_notebook.db`, mirrors, `aria-designer/lab_notebook.db`) after compile-fix trust reset; all `failure_signatures` row counts are now `0`.
