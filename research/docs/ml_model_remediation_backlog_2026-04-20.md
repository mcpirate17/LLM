# ML Model Remediation Backlog

Generated: 2026-04-20

Source audit: current runtime artifacts, current code paths, current DB-backed corpus builders, and current runtime wiring.

This backlog replaces any soft interpretation of the model state. It is ordered by dependency and expected operational lift.

## Current Truth

- `screening_ensemble` is the only learned component that is currently allowed to steer search.
- `gbm_gate` and `graph_predictor` are useful only as ensemble inputs, not as standalone production gates.
- `gbm_rank` is not trustworthy as currently implemented because it mixes `composite_score_best` and `wikitext_perplexity_best` into one head while runtime treats the output as a single lower-is-better score.
- `investigation_predictor` is currently blocked by policy. The live saved report says `spearman_rho = 0.16324262346620802` and `n_test = 235`.
- `interaction_model` has no persisted holdout evidence and should not directly steer anything.
- `bayesian_tracker` has no direct holdout ROC/PPV/NPV evidence and should remain advisory only.
- `use_learned_candidate_weights = false`, `use_screening_signal_weights = false`, and `use_learned_grammar_weights = false` in recent synthesis runs, so Aria is still weakly informed before candidate generation.
- `predictor_metrics_report.json`, `model_registry.json`, and `ml_hardening_plan_2026-04-08.md` do not currently present one consistent source of truth.

## Priority Summary

| Priority | Workstream | Why it matters | Expected lift |
| --- | --- | --- | --- |
| P0 | Evaluation unification | Current reports disagree with runtime artifacts and each other. | Restores trust in every downstream decision. |
| P0 | `gbm_rank` repair | Current ranking head has mixed target semantics and wrong runtime interpretation. | High lift for ranking quality and practical search quality. |
| P0 | Runtime threshold cleanup | Current config naming and operating points are stale and misleading. | Better PPV/recall tradeoff and clearer behavior. |
| P1 | Temporal validation | Current trust gating relies too much on grouped random splits. | Better production realism and lower leakage risk. |
| P1 | Segment feature promotion | Current segment-family analysis is the clearest underused signal source. | Likely lift for ROC AUC, PR AUC, top-k quality, and triage. |
| P1 | Investigation predictor rehab or retirement | Current model is blocked but still configured on by default. | Prevents false trust and clarifies whether it is worth keeping. |
| P2 | Policy/doc/registry sync | Current docs are stale and contradict runtime. | Reduces operator error and future confusion. |
| P2 | Controlled rollout of disabled learned influence paths | These are the main missing early-stage learned influence paths. | Potential high lift, but only after validation. |

## Backlog

### P0.1 Canonical Evaluation Artifacts

Status: not started

Problem:

- `predictor_metrics_report.json` currently mixes "fresh train" evaluation and saved-runtime-artifact evaluation.
- The ensemble headline metrics in the report are stronger than the persisted calibration metadata for the same component.
- `model_registry.json` contains stale metrics and stale status labels.

Required changes:

- Build one canonical evaluator for all saved model artifacts.
- Persist raw holdout predictions for every production-relevant model.
- Persist the exact split definition used for each evaluation.
- Persist one metrics bundle per artifact:
  - ROC AUC
  - PR AUC
  - PPV
  - NPV
  - precision
  - recall
  - F1
  - specificity
  - sensitivity
  - confusion matrix
  - Brier score
  - ECE
  - threshold sweep
  - reliability bins
  - bootstrap confidence intervals
- Separate "evaluate saved runtime artifact" from "train a fresh model and report its holdout".
- Regenerate the registry and hardening surfaces from the same artifact-backed evaluator.

Files:

- `research/tools/train_predictors.py`
- `research/scientist/intelligence/predictor_artifacts.py`
- `research/runtime/learning/predictor_metrics_report.json`
- `research/runtime/learning/model_registry.json`
- `research/docs/ml_hardening_plan_2026-04-08.md`

Acceptance criteria:

- A saved artifact has exactly one authoritative metrics bundle.
- The runtime policy reads the same artifact metrics that operators see in docs and the dashboard.
- No metric in `model_registry.json` differs from the current saved report for the same artifact.

### P0.2 Repair `gbm_rank`

Status: not started

Problem:

- `predictor_gbm.py` trains one rank head on a mixed target:
  - `composite_score_best` when available
  - `wikitext_perplexity_best` otherwise
- Runtime interprets rank predictions as one lower-is-better score.
- This is semantically broken and suppresses ranking quality.

Required changes:

- Replace the mixed-target rank head with one of these:
  - separate `gbm_rank_composite` and `gbm_rank_ppl` heads, or
  - one normalized quality target with consistent direction and scale.
- Update `planning_score` to use only correctly interpreted rank outputs.
- Add per-head ranking diagnostics:
  - Spearman
  - Pearson
  - Kendall
  - NDCG@k
  - top-k hit behavior

Files:

- `research/scientist/intelligence/predictor_gbm.py`
- `research/scientist/intelligence/predictor_ensemble.py`
- `research/runtime/learning/gbm_predictor.json`

Acceptance criteria:

- No rank head mixes incompatible target semantics.
- Runtime no longer assumes lower-is-better for a sometimes higher-is-better target.
- Rank diagnostics improve over the current mixed-head baseline.

### P0.3 Threshold and Naming Cleanup

Status: not started

Problem:

- `gbm_gate_threshold` is a stale config name. Runtime is actually gating on ensemble `p_pass`, not standalone GBM.
- `planning_score` is being used for sorting but is not a calibrated probability.
- The saved ensemble threshold is currently recall-heavy, while runtime uses a harder floor.

Required changes:

- Rename `gbm_gate_threshold` to something accurate like `screening_ensemble_p_pass_floor`.
- Separate gate policy from reranking policy.
- Persist multiple approved operating points per saved artifact:
  - collection mode
  - balanced mode
  - precision mode
- Log which threshold profile was used for each experiment.
- Never present `planning_score` as calibrated probability.

Files:

- `research/scientist/runner/_types.py`
- `research/scientist/runner/execution_experiment_phase3.py`
- `research/scientist/ml_influence_policy.py`
- `research/scientist/intelligence/predictor_ensemble.py`

Acceptance criteria:

- Runtime config names match actual behavior.
- Operators can tell whether a run used recall-biased or precision-biased gating.
- `planning_score` is only used as ranking signal.

### P1.1 Temporal Validation

Status: not started

Problem:

- Production trust is currently derived mostly from grouped random split logic.
- `investigation_predictor` still uses a non-temporal first-80-percent split after fingerprint ordering.
- The project needs time-aware evidence, not just duplicate-aware evidence.

Required changes:

- Add temporal holdout evaluation for:
  - ensemble screening gate
  - GBM gate
  - graph predictor gate
  - investigation predictor
- Keep grouped-by-fingerprint validation as a secondary check.
- Use temporal holdout metrics for promotion and policy gating.

Files:

- `research/scientist/intelligence/ml_corpus.py`
- `research/scientist/intelligence/predictor_gbm.py`
- `research/scientist/intelligence/gnn_predictor.py`
- `research/scientist/intelligence/predictor_ridge.py`

Acceptance criteria:

- Every production-facing model has a persisted temporal holdout section.
- Policy promotions reference temporal metrics, not only grouped random metrics.

### P1.2 Promote Segment Features Into The Screening Stack

Status: not started

Problem:

- Current segment-family analysis is promising but disconnected from runtime.
- Hybrid segment features outperform baseline-only features for several real targets.

Evidence from current analysis:

- `stage1_any_passed`: hybrid AUC `0.785`
- `binding_positive`: hybrid AUC `0.819`
- `induction_positive`: hybrid AUC `0.772`
- `hellaswag_positive`: hybrid AUC `0.732`
- `loss_ratio_best`: hybrid Spearman `0.489`

Required changes:

- Add segment-derived features to the screening feature pipeline, or
- build a small persisted segment sidecar whose score is fed into the ensemble.
- Evaluate incremental lift over current ensemble and current GBM.
- Prioritize lift on:
  - PR AUC
  - top-k precision
  - recall capture in top-ranked candidates
  - screening PPV at practical thresholds

Files:

- `research/scientist/intelligence/graph_segments.py`
- `research/scientist/intelligence/predictor_gbm.py`
- `research/scientist/intelligence/predictor_ensemble.py`

Acceptance criteria:

- Segment-augmented screening beats the current ensemble on at least one trusted holdout regime without harming calibration beyond acceptable tolerance.

### P1.3 Investigation Predictor: Rehab Or Retire

Status: not started

Problem:

- Current runtime policy blocks the investigation predictor, correctly.
- The current saved report is materially worse than the April 11, 2026 snapshot.
- The current runtime threshold `0.7` is too loose to be operationally informative.

Required changes:

- Re-evaluate using temporal split.
- Test realistic triage thresholds around:
  - `0.45`
  - `0.50`
  - `0.55`
  - `0.60`
- Expand investigation-tier labels, especially near the decision boundary.
- If temporal Spearman and triage behavior remain weak, retire the runtime path.

Files:

- `research/scientist/intelligence/predictor_ridge.py`
- `research/scientist/runner/continuous_investigation.py`
- `research/scientist/ml_influence_policy.py`

Acceptance criteria:

- Either the model clears temporal triage thresholds and is re-enabled, or it is explicitly downgraded to offline analysis only.

### P2.1 Policy, Registry, and Docs Sync

Status: not started

Problem:

- `ml_hardening_plan_2026-04-08.md` says the screening ensemble is off by default. Current code says it is on and allowed.
- `model_registry.json` reports stale metrics and stale trust levels.
- Current policy code is closer to truth than the docs, but naming and commentary are still stale.

Required changes:

- Regenerate `model_registry.json` from current artifact-backed evaluation.
- Replace stale hardening-plan content with current reality.
- Make the ML influence status page clearly distinguish:
  - active generation influence
  - active screening influence
  - advisory-only influence
  - blocked-by-policy components

Files:

- `research/runtime/learning/model_registry.json`
- `research/docs/ml_hardening_plan_2026-04-08.md`
- `research/scientist/api_routes/_ml_influence_status.py`
- `research/scientist/ml_influence_policy.py`

Acceptance criteria:

- No model is labeled production, usable, or trustworthy without matching current artifact evidence.

### P2.2 Controlled Rollout For Currently Disabled Learned Influence Paths

Status: not started

Problem:

- `use_learned_candidate_weights`, `use_screening_signal_weights`, and `use_learned_grammar_weights` are off.
- This is one of the main reasons Aria remains weakly informed before screening.
- Turning them on globally now would be unprincipled because they are not yet validated.

Required changes:

- Add experiment-level instrumentation for each path.
- Run controlled comparisons:
  - control
  - candidate-weights only
  - screening-signal only
  - grammar-weights only
  - combined
- Measure effect on:
  - compile pass rate
  - stage0 and stage05 pass rate
  - stage1 PPV
  - downstream investigation candidate quality
  - diversity collapse risk

Files:

- `research/scientist/runner/execution_candidates.py`
- `research/scientist/runner/execution_screening.py`
- `research/scientist/runner/continuous_modes.py`
- `research/scientist/ml_influence_policy.py`

Acceptance criteria:

- A path is enabled by default only after it demonstrates lift with acceptable regression risk.

### P2.3 Remove Unsupported Direct Steering Claims

Status: not started

Problem:

- `interaction_model` and `bayesian_tracker` still have implied steering value in some surfaces.
- Their direct steering evidence is not sufficient.

Required changes:

- Remove unsupported "production" language where it implies direct decision-grade trust.
- Keep Bayesian framed as advisory grammar prior only.
- Keep interaction framed as research or sidecar feature source only until holdout evidence exists.

Files:

- `research/runtime/learning/model_registry.json`
- `research/docs/ml_hardening_plan_2026-04-08.md`
- any dashboard copy or API text that overstates trust

Acceptance criteria:

- No UI or doc surface claims direct steering trust for models without direct holdout evidence.

## Recommended Execution Order

1. P0.1 Canonical evaluation artifacts
2. P0.2 Repair `gbm_rank`
3. P0.3 Threshold and naming cleanup
4. P1.1 Temporal validation
5. P1.2 Segment feature promotion
6. P1.3 Investigation predictor rehab or retire
7. P2.1 Policy, registry, and docs sync
8. P2.2 Controlled rollout for disabled learned influence paths
9. P2.3 Remove unsupported direct steering claims

## Definition Of Done

This remediation effort is complete only when all of the following are true:

- every production-facing model has one authoritative saved-artifact evaluation bundle
- policy gating reads those artifact metrics directly
- registry and docs are generated from the same source of truth
- no mixed-semantics rank head remains in runtime
- ensemble threshold profiles are explicit and reproducible
- temporal validation exists for every production-facing model
- disabled learned influence paths are either validated and promoted or explicitly kept experimental
- no stale doc claims contradict current runtime behavior
