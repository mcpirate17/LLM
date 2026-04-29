# Bug And Scoring Fixes Action Plan

This is a research/action-plan document only. Do not change scoring, UI, database rows, or promotion behavior until the current runs and their benchmark/probe tails have settled.

## Goals

1. Make the live investigation UI tell the truth about what is still running.
2. Fix the live loss chart so it is segmented by candidate/program metadata, not by accidental step resets.
3. Rebalance scoring so loss cannot carry a weak model into champion range by itself.
4. Make induction/binding/long-context capability matter without rigging the system for attention.
5. Reuse the tests and evals already in the repo first; add new tests only for gaps.

## Current Evidence

From the current `research/lab_notebook.db` leaderboard snapshot:

- `0cffa5cff90c3bc5` is already capped at `360.0`, but it still has `induction_v2=0.0`, `binding_v2=0.0`, `ar_auc=0.004`, and `long_ctx_combined=0.002`. That supports keeping the trust ceiling, but it also shows the ranking needs a clearer champion eligibility rule.
- The top validation rows include models with near-zero induction v2 but very high total scores. Example: the current top row has score `547.2`, `induction_v2=0.005`, `ar_auc=0.005`, and `long_ctx_combined=0.317`. That may be a real special case, but it should have to prove itself across multiple non-loss signals.
- `leaderboard_scoring.py` currently gives the loss tier up to about `230` points: `perf_short=35`, `perf_medium=50`, `perf_long=65`, `param_efficiency=30`, `learning_efficiency=15`, `speed=25`, `early_convergence=10`. Understanding is about `175` points, capability v11 adds more, and aux trajectory is about `20`.
- The live loss event payload only contains `experiment_id`, `step`, `loss`, `total_steps`, `phase`, and sometimes `grad_norm` / `routing_aux_loss`. It does not identify `source_result_id`, candidate index, training program index, seed, or training-program label. The dashboard therefore has to infer segments from step resets.
- Investigation completion can be emitted before background benchmark/probe writes are finished. `_complete_investigation()` publishes `experiment_completed` and `investigation_completed`, while `_submit_benchmark_eval()` and `_submit_v2_probe_eval()` run in a background pool and record later.

## External Research Takeaways

- Mamba should be evaluated by behavior, not by attention internals. The Mamba paper argues that selective state spaces address content-based sequence reasoning and should be checked on language modeling, downstream performance, selective copying, induction-style behavior, and long-sequence scaling: https://arxiv.org/abs/2312.00752
- Long-context scoring should not rely on passkey alone. RULER explicitly treats simple needle retrieval as incomplete and adds multi-hop tracing and aggregation tasks: https://arxiv.org/abs/2404.06654
- There is evidence that Mamba can do in-context learning, but also evidence that standard downstream ICL can lag comparable Transformers. That means Mamba deserves a fair exception path, not a free pass: https://arxiv.org/abs/2402.03170 and https://openreview.net/forum?id=C3t6GMPnC5

## Existing Tests And Evals To Reuse

Use these before adding new benchmark code:

| Area | Existing module/test | Why it matters |
| --- | --- | --- |
| Induction | `research/eval/induction_probe.py`, `research/eval/induction_probe_v2_investigation.py`, `research/tests/test_induction_probe.py`, `research/tests/test_induction_probe_v2_investigation.py` | Behavioral `[A][B] ... [A] -> B` induction across gaps. This is the main non-local pattern completion signal. |
| Binding | `research/eval/binding_range.py`, `research/eval/binding_probe_v2_investigation.py`, `research/eval/binding_curriculum.py`, `research/tests/test_binding_curriculum_probe.py` | Copy/binding across distance. Useful for distinguishing local-only models from models with non-local memory. |
| Associative recall | `research/eval/associative_recall.py`, `research/eval/long_range_ar.py`, `research/tests/test_ar_probe_internals.py` | Strict key-value retrieval. Important evidence, but do not make this the sole Mamba gate because the module itself expects Mamba/RWKV-style compressed-state models to struggle. |
| Long context | `research/eval/long_context.py`, `research/eval/passkey_retrieval.py`, `research/eval/multi_hop_retrieval.py`, `research/scientist/runner/_eval_registry.py` | Already wires scaling, passkey, long-range AR, and multi-hop retrieval, then aggregates them. |
| Downstream language | `research/eval/wikitext_eval.py`, `research/eval/blimp_eval.py`, `research/eval/hellaswag_eval.py`, `research/eval/tinystories_eval.py` | Loss plus actual language/commonsense checks. HellaSwag and BLiMP are already used in trust scoring. |
| Scoring safety | `research/tests/test_leaderboard_trust_scoring.py`, `research/tests/test_scoring_binding_safety.py`, `research/tests/test_loss_ratio_fix.py` | Existing scaffolding for trust ceilings, binding safety, and loss ratio regressions. Extend these before adding new test files. |
| Dashboard/runtime | `research/tests/test_dashboard_runtime_status_guards.py`, frontend tests under `research/dashboard/src` | Good place for status/terminal-event regressions and score breakdown consistency. |

## Action Plan

### Phase 0: Freeze Live Mutations

- Do not rescore the DB.
- Do not retune weights live.
- Do not change UI buckets until a dry-run report shows the before/after rank movement.
- Let the currently running investigations and their benchmark/probe tails settle before implementing runner changes.

### Phase 1: Fix Runtime Truthfulness

1. Add stable metadata to every `training_step` event:
   - `source_result_id`
   - candidate index / total candidates
   - training program index / total programs
   - training program label or seed
   - stage/run kind: screening, investigation, validation, benchmark, v2-probe
2. Store the same metadata in `_live_loss_curve`.
3. Update the dashboard loss chart to split by explicit metadata, not only `step` resets.
4. Render per-segment labels from candidate/program metadata.
5. Make y-axis behavior explicit:
   - either per-segment local scaling, or
   - global scaling with clear visual normalization.
6. Change terminal lifecycle semantics:
   - emit `investigation_training_complete` after candidate training finishes,
   - keep status as `finalizing` while benchmark/probe futures are still pending,
   - emit `investigation_completed` only after required benchmark/probe writes are finished or explicitly marked skipped/timed out.

Tests to reuse/add:

- Extend `research/tests/test_dashboard_runtime_status_guards.py`.
- Add a small runner/unit test that a queued benchmark future prevents terminal `completed`.
- Add a frontend unit test for loss-curve segmentation with two candidates and three training programs.

### Phase 2: Score Rebalance Behind A New Version

Create a new scoring version, e.g. `v12`, and dry-run it only.

Recommended first-pass loss budget:

| Component | Current | Proposed |
| --- | ---: | ---: |
| `perf_short` | 35 | 30 |
| `perf_medium` | 50 | 40 |
| `perf_long` | 65 | 55 |
| `param_efficiency` | 30 | 20 |
| `learning_efficiency` | 15 | 10 |
| `speed` | 25 | 15 |
| `early_convergence` | 10 | 5 |
| Total | 230 | 175 |

Keep understanding and capability weights stable for the first dry run. Do not inflate the whole scoreboard just to compensate for loss; that makes the total harder to reason about. The safer first change is to bring loss back to parity.

Dry-run checks:

- Top 100 old-vs-new rank diff.
- List every model that drops out of top 20 and why.
- List every model that enters top 20 and which non-loss signals caused it.
- Verify `0cffa5cff90c3bc5` and similar rows remain below champion range unless they have real non-loss evidence.

Tests to reuse/add:

- Extend `research/tests/test_leaderboard_trust_scoring.py` with a max-loss-budget assertion.
- Add fixture rows where strong loss alone cannot outrank induction-qualified candidates.
- Add a score-breakdown sum test so UI buckets match backend decomposition.

### Phase 3: Champion Eligibility Gate

Add an explicit champion-eligibility layer separate from raw additive score.

Baseline rule:

- A model with missing/near-zero effective induction should not beat an induction-qualified model in the same trust tier unless it clears a strict exception path.
- Effective induction should prefer `induction_v2_investigation_auc` when present, then fall back to `induction_auc`.
- Effective binding should prefer `binding_v2_investigation_auc` when present, then fall back to `binding_auc`.

Suggested thresholds for the first dry run:

- `induction_qualified`: effective induction >= `0.05`.
- `strong_induction`: effective induction >= `0.30`.
- `binding_qualified`: effective binding >= `0.20`.
- `strong_binding`: effective binding >= `0.50`.

Exception path for Mamba/SSM-like or other non-attention models:

- Must have reproduced BPE loss, not byte-era or unknown-tokenizer loss.
- Must pass at least two non-loss sequence tests, such as:
  - diagnostic selective copy,
  - passkey retrieval,
  - multi-hop retrieval,
  - long-context scaling,
  - ICLD / trajectory learning signal,
  - BLiMP or HellaSwag above cohort median.
- Must not rely on associative recall alone, because AR is a strict exact-retrieval task and the repo already documents that Mamba/RWKV-style compressed state models can fail it for architectural reasons.

This keeps the policy mechanism-neutral: attention can win through induction/binding/retrieval; Mamba-like models can win through selective memory, long-context scaling, ICL-style behavior, and downstream language quality. Loss alone cannot win.

Tests to reuse/add:

- Extend `test_leaderboard_trust_scoring.py` with:
  - no-induction + strong-loss candidate capped below champion eligibility,
  - induction-qualified candidate allowed above the cap,
  - Mamba-like exception allowed only with low BPE loss plus at least two non-loss sequence signals,
  - Mamba-like exception rejected when the only strong signal is loss.

### Phase 4: Long-Context Score Repair

Do not tune weights first. Verify that the long-context metrics are real and populated correctly.

1. Trace the data path:
   - `_eval_registry.py` produces `long_ctx_scaling_score`, `long_ctx_assoc_score`, `long_ctx_passkey_score`, `long_ctx_multi_hop_score`, `long_ctx_retrieval_aggregate`, and `long_ctx_combined_score`.
   - `_shared.py`, `program_result_merge.py`, `leaderboard_maintenance.py`, and `notebook_leaderboard.py` carry these into `program_results` and `leaderboard`.
   - `leaderboard_scoring.py` reads `robustness_long_ctx_combined_score` or `robustness_long_ctx_score`.
2. Add missing-result diagnostics:
   - distinguish not-run, timed-out, failed, chance-level, and true zero.
   - display protocol version and subtest statuses.
3. Fix scoring only after the data path is verified.
4. If the graph is questionable, add a UI warning when a metric is missing or stale rather than plotting it as a confident zero.

Tests to reuse/add:

- Add a merge test that all long-context sub-scores survive `program_results -> leaderboard`.
- Add a scorer test that missing long-context data is not confused with a real `0.0`.
- Add frontend tests for status labels: missing, failed, timed out, chance-level, measured.

### Phase 5: UI Bucket Separation

Separate "Loss" and "Understanding" everywhere in the UI. They are different kinds of evidence:

- Loss: WikiText/BPE PPL, screening/investigation/validation loss ratios, param efficiency, learning efficiency, speed, early convergence.
- Understanding: BLiMP, TinyStories, cross-task robustness, diagnostic suite, HellaSwag, hierarchy fitness.
- Capability: AR, induction, binding, ERF density, ID collapse, ERF decay, logit margin.
- Long context should either be its own bucket or a visible sub-bucket under capability, not hidden inside generic robustness.

Tests to reuse/add:

- Extend `research/dashboard/src/utils/backendScore.test.js`.
- Add a UI snapshot/unit test that loss and understanding totals are rendered separately.
- Add a backend/frontend parity test using the same fixture breakdown.

### Phase 6: Dry-Run Report Before Any DB Rescore

Build a report-only tool or notebook query that emits:

- top 100 current v11,
- top 100 proposed v12,
- deltas by rank and score,
- reason codes for all major moves,
- candidates capped by no-induction rule,
- candidates allowed by Mamba/SSM exception,
- candidates affected by missing long-context metrics.

Only after reviewing this report should anything write new composite scores to the DB.

## If We Need More Tests

Add tests only if the current suite cannot answer the question.

Potential additions:

1. A mechanism-neutral selective-copy length sweep based on `diagnostic_tasks.py`, using lengths like `64, 128, 256, 512`.
2. A small ICL decay test that measures few-shot pattern use across context length, using the existing ICLD/trajectory machinery if possible.
3. A LAMBADA-style continuation test if there is no local language-continuation equivalent already available.
4. A calibration harness with reference families:
   - local/conv-only,
   - GPT-2/attention,
   - Mamba/SSM reference,
   - RWKV-like/recurrent if available,
   - routed MoE candidates.
5. Score-diff regression fixtures that lock the policy:
   - no-induction + loss-only cannot become champion,
   - induction-qualified can win with mediocre but sane loss,
   - Mamba-like can win only with low BPE loss plus multiple non-loss sequence signals,
   - long-context missing data cannot masquerade as measured zero.

## Suggested First Implementation Order

1. Runtime truthfulness: metadata-rich loss events and `finalizing` status.
2. Long-context data-path tests and missing-vs-zero repair.
3. UI bucket separation.
4. v12 scoring dry-run, report only.
5. Champion eligibility gate dry-run.
6. Review rank-diff report.
7. Only then rescore or change default scoring version.
