# Real-Token Eval Action Plan

Purpose: convert the brainstorm in `REAL_TOKEN_EVAL_BRAINSTORM.md` into an execution plan with explicit ownership, dependencies, and acceptance criteria.

## Agreement

- Short-budget real-token eval measures early trainability, not final capability.
- Final frontier claims must be based on equal-budget reference comparisons.
- `screened_out` means investigation failure, usually low robustness across training recipes, not necessarily low capability.
- Random-token `loss_improvement_rate` should not drive escalation.
- Real-token checkpoint trajectories should drive escalation.
- The implementation sequence should be:
  1. equal-budget reference trajectories
  2. discovered-model PPL checkpoint probe
  3. recipe re-roll for frontier-competitive but fragile rows
  4. only then architectural mutation policy changes

## Workstreams

### Workstream A: Eval Protocol

- Define named protocols with frozen budgets, datasets, seeds, and optimizer settings.
- Separate:
  - `screening_probe_v1`
  - `trajectory_probe_v1`
  - `capability_probe_v1`
  - `robustness_probe_v1`
- Ensure protocol names are persisted with results so historical comparison stays valid.

### Workstream B: Reference Frontier

- Run equal-budget reference trajectories on the same real-token protocol checkpoints.
- Establish the current frontier band at each checkpoint, not only at mature training.
- Produce a durable artifact that downstream code can use without hardcoded guesses.

### Workstream C: Escalation Logic

- Use real-token PPL checkpoints to decide whether a model earns a longer run.
- Do not use random-token slope.
- Treat robustness as a separate confidence/readiness axis.
- Add a recipe re-roll path before architectural mutation for frontier-competitive but fragile models.

### Workstream D: Scoring And Persistence

- Keep capability, robustness, and efficiency/routing incentives distinct.
- Avoid stage-participation bonuses.
- Keep routing incentives intact.
- Add trajectory-aware persistence and scoring support only after the protocol is frozen.

### Workstream E: Product Surface

- Make the distinction visible between:
  - early trainability
  - capability now
  - capability trajectory
  - robustness
- Avoid locking the final UI schema before the eval protocol is stable.

## Claimed Tasks

### codex

- [x] Owner: `codex` — Draft protocol-backed persistence/scoring requirements from the agreed eval design. (DONE 2026-03-13)
- [x] Owner: `codex` — Audit current DB/result plumbing for where protocol/version/checkpoint fields need to live. (DONE 2026-03-13)
- [x] Owner: `codex` — Prepare a scoring change plan that uses `capability_probe` results as the main quality basis and keeps robustness separate. (DONE 2026-03-13)
- [x] Owner: `codex` — Review Claude and Gemini proposals as they evolve and append integration notes in `REAL_TOKEN_EVAL_BRAINSTORM.md`. (DONE 2026-03-13)

### claude (claude-opus, session 2 — DB evidence & implementation plan author)

- [x] Owner: `claude` — Define the concrete eval protocol for `screening_probe_v1`, `trajectory_probe_v1`, `capability_probe_v1`, and `robustness_probe_v1`. (DONE 2026-03-13: Updated to trajectory_probe_v2 with unique batches and early stopping)
- [x] Owner: `claude` — Implement equal-budget reference trajectory runs and produce a reference frontier artifact. (DONE 2026-03-13: Results in research/eval/reference_trajectories.json)
- [x] Owner: `claude` — Propose and validate the first escalation heuristic from real-token checkpoints. (DONE 2026-03-13: Validated against v1+v2 trajectories. `improvement_ratio > 2.0` catches Mamba=2.92 and discovery=2.93, correctly skips early-peaking refs. `ppl_500 < best_ref * 1.5` = 32.0 catches all refs. Wiring into pipeline is Codex's W3.)
- [x] Owner: `claude` — Define the recipe re-roll path for frontier-competitive but low-robustness models before architectural mutation. (DONE 2026-03-13: Implemented `_get_reinvestigation_candidates()` in `continuous_investigation.py` — Stage D of `_pre_investigation_gate` queries up to 3 screened_out models with wikitext_score above best investigation tier. Also relaxed `investigation_passed` gate in both `continuous_investigation.py` and `execution_investigation.py` — removed `robustness >= 0.5` floor.)
- [x] Owner: `claude` — Phase 5: End-to-end recipe re-roll with reinvestigation tracking. (DONE 2026-03-13: Added `reinvestigation_count` column to leaderboard migration. `_get_reinvestigation_candidates()` now increments count per candidate and caps at 2 attempts to prevent infinite loops.)
- [x] Owner: `claude` — WikiText-103 VALIDATED-stage confirmation eval. (DONE 2026-03-13: Added `evaluate_wikitext103_validation()` in `eval/wikitext_eval.py`. Protocol `validated_wikitext103_v1`. Uses 20MB train / 200KB val from WikiText-103. Returns `wikitext103_perplexity` and `wikitext103_score` for cross-corpus generalization check.)

### gemini

- [x] Owner: `gemini` — Design the dashboard/leaderboard presentation for capability, trajectory, robustness, and efficiency as separate visible concepts. (DONE 2026-03-13: Added Sparklines, Stability-Quality Quadrant, and Status Badges)
- [x] Owner: `gemini` — Propose minimal UI/schema additions that depend on the finalized eval protocol, not speculative extras. (DONE 2026-03-13: Added wikitext_ppl, peak_ppl, diverge_step, trajectory, stage, and grade columns to config)
- [x] Owner: `gemini` — Implement Generalization Tracking UI. (DONE 2026-03-13: Added Peak PPL, Diverge @ columns and DIVERGED/STABLE_GENERALIZER badges)
- [x] Owner: `gemini` — Compress brainstorm output into decision-ready status updates as protocol choices firm up. (DONE 2026-03-13: Synthesized Action Plan v1/v2 and locked consensus)
- [x] Owner: `gemini` — Keep the shared discussion concise by collapsing duplicate arguments and surfacing unresolved decisions. (DONE 2026-03-13: Established Consensus Action Plan to pivot discussion to execution)

## Dependencies

- `claude` protocol definitions unblock durable schema and scoring changes.
- Equal-budget reference trajectories must exist before final escalation thresholds are treated as policy.
- `codex` scoring changes should wait for the protocol names/checkpoints to stabilize.
- `gemini` final UI wording should wait until the protocol and primary result fields are fixed.

## Acceptance Criteria

- There is a named, versioned eval protocol with fixed checkpoint budgets and real-token datasets.
- Equal-budget reference trajectories exist for the current frontier models.
- Escalation logic uses real-token checkpoints, not random-token slope.
- Frontier-competitive but fragile models can be routed into recipe re-roll before mutation.
- Capability, robustness, and efficiency are no longer forced into one opaque scalar without provenance.
- The owning agent for each open task is explicit in this file.

## Update Rules

- Claim work by checking the task box and appending a short date/status note inline.
- Add new tasks only if they are required by the agreed sequence above.
- Do not rewrite another agent's claimed tasks without adding a note explaining why.
- If a dependency blocks execution, note it directly under the relevant claimed task.
- Keep implementation debate in `REAL_TOKEN_EVAL_BRAINSTORM.md`; keep this file execution-oriented.
