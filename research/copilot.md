# Copilot Observations — Main Improvements Needed

Updated: 2026-02-15

## Priority 0 (Critical correctness)

1. **Evolution operators were too close to random regeneration**
   - Why it matters: breakthrough search depends on preserving and improving lineages, not repeatedly restarting from scratch.
   - Improvement direction: keep parent-informed mutation/crossover semantics and continue validating lineage quality.

2. **CLI pipeline drift from runner APIs**
   - Why it matters: an executable entrypoint that does not run current APIs can silently invalidate all user-facing workflows.
   - Improvement direction: keep `python -m research` modes aligned to `ExperimentRunner` start methods and add smoke tests for each mode.

3. **Silent failure masking in search/eval paths**
   - Why it matters: swallowed exceptions can appear as poor candidates rather than infrastructure defects.
   - Improvement direction: preserve explicit failure metadata (error type/message counters) across fitness/novelty and surface in dashboard/reporting.

## Priority 1 (Scientific rigor)

4. **Novelty calibration is still partly heuristic** *(ADDRESSED)*
   - `BehavioralFingerprint` now tracks `analyses_succeeded` (0–4) and `quality` ("full"/"partial"/"none"). `NoveltyMetrics.novelty_confidence` (0.0–0.9) reflects fingerprint quality and is persisted in `program_results.novelty_confidence`. Consumers can now weight/filter novelty scores by confidence.
   - Remaining gap: CKA references are still synthetic (#28/#43); confidence caps at 0.9 to reflect this.

5. **Single-flow pipeline validation was missing**
   - Why it matters: subsystem tests can pass while end-to-end scientific behavior regresses.
   - Improvement direction: retain the end-to-end continuous pipeline test that checks novelty output, learning-memory updates, experiment persistence, and campaign/report APIs.

6. **Learning loop should remain auditable**
   - Why it matters: pharma-style R&D requires traceable hypothesis→result→decision→next-action transitions.
   - Improvement direction: continue logging grammar-weight applications, decisions, and knowledge extraction artifacts in notebook tables and reports.

## Priority 2 (Interoperability and operations)

7. **Dashboard/backend route contract should be continuously enforced**
   - Why it matters: API drift breaks observability/control loops and blocks research operations.
   - Improvement direction: keep route-contract integration tests that compare frontend API paths against Flask routes.

8. **Dead-code hygiene needs continuous, non-destructive audit**
   - Why it matters: stale code paths increase cognitive load and hide real defects.
   - Improvement direction: keep report-only dead-code auditing with manual review gates (never auto-delete).

## Priority 3 (Still open / next improvements)

9. **True experiment clustering is not yet explicit**
   - Current state: the system has correlations, top-op combinations, campaign grouping, and reports; explicit unsupervised clustering of experiment outcomes is limited.
   - Improvement direction: add reproducible clustering views (e.g., by behavior/failure signatures) and test cluster stability over time.

10. **Publication-grade novelty replication criteria can be tightened further** *(ADDRESSED)*
   - Runner breakthrough promotion now aligned with Aria publication thresholds: baseline ratio < 0.90, std ≤ 0.03, ≥ 5 seeds (default raised from 3), OOD ≥ 0.67, HP ≥ 0.75, novelty_confidence ≥ 0.5.
   - Copilot previously added `assess_breakthrough_evidence()` with 3-tier labeling and `announce_breakthrough()` language gating.
   - Runner and persona now agree on what constitutes a breakthrough.

---

## Claude Code Observations (added 2026-02-15)

### Recently fixed

11. **Report key name mismatch was hiding 21% S1 pass rate**
    - `_rule_based_report_narrative` read `total_s1_passed` but `get_dashboard_summary()` returns `stage1_survivors`. Reports showed "0% S1 pass rate" despite 129/610 passing. Same mismatch in `_maybe_auto_report` metadata.
    - Status: fixed in 9f8b049.

12. **Grammar weight formula produced almost no learning signal**
    - Additive formula `1.0 + s1_rate*3.0 + novelty*1.0` gave range ~1.5-2.0 across all categories — barely distinguishable. Replaced with multiplicative contrast amplification: `default * (s1_rate/mean)^2 * (1+novelty)`, range [0.1, 8.0].
    - Status: fixed in 9f8b049.

### Performance issues found and fixed

13. **Per-insert DB commits** — `ExperimentDB.save_spec/save_stage0/save_stage1` each called `conn.commit()`. Added `batch()` context manager for deferred commits. (9f8b049)

14. **JSON `indent=2` in graph serialization** — every graph stored with pretty-printed JSON. Switched to compact `separators=(",",":")`. ~30-40% smaller. (9f8b049)

15. **Multi-pass analytics** — `structural_correlations()` iterated all rows once per metric (7 passes). Refactored to single-pass accumulation. (96573a1)

16. **Uncached graph properties** — `depth()`, `fingerprint()`, `n_ops()`, etc. recomputed on every call including topo sort. Added `_cache` dict invalidated by mutations. (96573a1)

17. **O(n) list.pop(0) in BFS** — `has_gradient_path()` used list as queue. Switched to `collections.deque`. (9f8b049)

18. **Linear option lookup** — `ArchSpec.get_option()` did nested linear search through DIMENSIONS tuple. Added module-level `_OPTION_MAP` dict. (9f8b049)

19. **update_op_success_rates parsed full graph JSON per row** — used `get_program_results()` which does `SELECT *` + `dict(r)` + `json.loads(graph_json)` per program. Replaced with targeted 6-column query, index-based access, running sum accumulators. (uncommitted)

### Still open (my view)

20. **O(n^2) constraint checking in morphological_box.py** — `_check_tag_incompatibilities` calls `get_option()` in nested loops. Now O(1) per lookup thanks to `_OPTION_MAP`, but the outer loop is still O(dims^2) with redundant set conversions. Low priority since dims~8.

21. **`top_op_combinations` in analytics.py also parses graph_json** — limited to 200 rows so lower impact, but same pattern as #19. Could benefit from a pre-extracted `graph_ops` column at insert time.

22. **No index on `program_results.experiment_id`** — `update_op_success_rates` and `get_program_results` filter by experiment_id without an index. Adding one would help at scale.

### Deep audit findings (2026-02-15)

#### Critical correctness

23. **Sandbox loss computed without shape validation** (`eval/sandbox.py:130-133`)
    - `F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ...)` silently produces wrong results if logits have unexpected shape (missing seq dimension).
    - Impact: invalid S0.5 results could mark unstable models as "passed".
    - Fix: validate `logits.shape == (batch, seq, vocab)` before reshaping.

24. **Baseline device mismatch on cache reuse** (`eval/baseline.py:154-158`)
    - Baseline transformer is cached after `.to(dev)`, but device param can change between calls. Cached model stays on wrong device.
    - Impact: silent mismatches or crashes during loss comparison.
    - Fix: cache by `(config, device)` tuple, or validate device on every `get_baseline_loss()` call.

25. **Compiler assumes weight exists without checking** (`synthesis/compiler.py:188`)
    - `F.linear(x, module.weight)` crashes if `has_params` init failed silently. No early guard.
    - Fix: assert weight exists or fall back gracefully.

26. **`import math` placed after usage in loss_synthesis.py** (`training/loss_synthesis.py:136 vs 129`)
    - `math.log(V)` used at line 129 but `import math` appears at line 136. Runtime error if KL uniform loss is selected.
    - Fix: move import to top of file.

#### Scientific methodology

27. **Novelty archive uses FIFO eviction, creating time bias** (`search/novelty_search.py:50-51`)
    - `self.entries = self.entries[-max_size:]` drops oldest entries. Novelty scores become time-dependent — late-run architectures are underrewarded because the archive is dominated by recent entries.
    - Fix: reservoir sampling or stratified eviction to maintain historical diversity.

28. **Fingerprint CKA uses heuristic reference patterns, not real models** (`eval/fingerprint.py:307-352`)
    - Reference patterns for "transformer", "SSM", "conv" are synthetic (exponential decay, lower-triangular, banded). Real architectures have more complex similarity structure.
    - Impact: CKA similarity scores are approximate at best, potentially misleading for novelty claims.
    - Fix: acknowledge as heuristic in reports; optionally train/cache reference patterns from real models.

29. **Structural novelty bonuses are additive and heuristic** (`eval/metrics.py:111-114`)
    - `+0.05` for math_spaces, `+0.03` for frequency_domain — both can apply, pushing toward 1.0 with weak justification.
    - Impact: exotic ops get disproportionate novelty boost regardless of actual functional novelty.
    - Fix: multiplicative scaling `novelty * (1 + 0.1 * n_exotic_features)` or learn bonuses from data.

30. **Evolution population can collapse to clones** (`search/evolution.py:135-146`)
    - Final population sorted by fitness+novelty blend but no diversity enforcement. If mutation is weak, all individuals can have identical fingerprints.
    - Fix: enforce fingerprint uniqueness in survivors or randomly resample when duplicates exceed threshold.

31. **FLOP estimates hardcode projection dimensions** (`eval/flops.py:72-87`)
    - `"down" in op_name → D // 2`, `"up" → D * 2` but grammar.py can set arbitrary `out_dim` in config.
    - Impact: efficiency frontier Pareto analysis could be off by 2-4x for non-standard projections.
    - Fix: read `config.get("out_dim", D)` instead of hardcoding multipliers.

32. **Outer product is actually Hadamard product** (`synthesis/compiler.py:146-150`)
    - `outer_product` op computes `a * b` (elementwise), not true outer product. True outer product would be (B,S,D)x(B,S,D)→(B,S,D,D), violating the shape contract.
    - Fix: rename to `hadamard_product` for accuracy, or implement true outer product with projection back to (B,S,D).

#### Robustness

33. **Pervasive bare `except Exception: pass` pattern**
    - `runner.py:373-374` (hypothesis validation), `api.py:225-229` (DB writes), `fingerprint.py:301`, `fingerprint.py:349`, `_linear_cka:365`, `_gather_analytics_data:2466`
    - Impact: systematic backend failures (LLM down, DB locked, OOM) go unnoticed. Analysis silently returns empty/zero instead of flagging the gap.
    - Fix: log all caught exceptions at WARNING level; record "unavailable" status in metadata rather than silent empty.

34. **No crash recovery for experiments** (`scientist/runner.py`)
    - If the process crashes mid-experiment, the experiment stays in "running" status forever. No cleanup or resume logic.
    - Fix: on startup, check for "running" experiments with no recent heartbeat and mark as "aborted". Or use a heartbeat table.

35. **Grammar param formula uses bare `eval()`** (`synthesis/grammar.py:329-333`, `graph.py:334`)
    - `eval(formula)` with user-controllable formula string. Current formulas are trusted (from primitive registry), but if registry is extended without validation, this is a code injection vector.
    - Fix: use `ast.literal_eval()` or a simple arithmetic parser instead.

36. **Anthropic model version hardcoded** (`scientist/llm/anthropic.py:20`)
    - Default model `claude-sonnet-4-5-20250929` will break when deprecated. Env var override exists but default should track latest.
    - Fix: use latest alias or check availability on init.

37. **SSE timeout hardcoded at 30s** (`scientist/api.py:253-262`)
    - Dashboard disconnects if experiment produces events slowly. Not configurable.
    - Fix: make configurable via env var or API param, default to 60s.

38. **Missing index on `program_results.experiment_id`** (notebook.py schema)
    - `update_op_success_rates` and `get_program_results` filter by experiment_id. At 10K+ programs, this becomes a full table scan.
    - Fix: `CREATE INDEX IF NOT EXISTS idx_program_results_experiment ON program_results(experiment_id)`.

#### Observability gaps

39. **No mode recommendation for "give up"** (`persona.py:948-955`)
    - If `total_s1 == 0` after many experiments, system recommends "synthesis" again. No heuristic for "this hypothesis has failed, pivot".
    - Fix: after N experiments with 0 survivors, recommend "pivot" or "stop" with explanation.

40. **LLM cost tracking uses silent default** (`persona.py:138`)
    - Unknown backend names default to Anthropic rate without warning. Cost estimates in dashboard could be wildly wrong.
    - Fix: log warning for unknown backends.

### Scientific methodology audit (2026-02-15)

These are systemic issues with the research pipeline's scientific validity, not code bugs.

#### Self-reinforcing bias loop (CRITICAL)

41. **Grammar learning creates confirmation bias** [CRITICAL]
    - Grammar weights learned from prior S1 survivors → applied to next experiment → generates more of the same → those pass S1 → weights reinforced.
    - **Mitigation needed**: Add grammar diversity penalty, periodic weight resets, or holdout validation set that wasn't used for grammar training.
    - No control experiment with random grammar weights for comparison.
    - No held-out validation: programs used for grammar learning are the same ones ranked on the leaderboard.
    - Fix: (a) null hypothesis test — run N experiments with random weights vs learned weights, (b) train/test split — learn grammar from experiments 1-8, evaluate on 9-10.

42. **Grammar weight updates amplify sampling noise** (`analytics.py:153-167`)
    - With ~100 programs per category, a 2% s1_rate difference (10% vs 12%) is within sampling variance. Squaring the ratio (1.2² = 1.44) amplifies this noise to a 44% weight change.
    - No confidence intervals, no multiple comparison correction, no statistical power analysis.
    - Fix: require minimum effect size + significance test before updating weights. Or use Bayesian updating with priors.

#### Novelty measurement validity (CRITICAL)

43. **CKA reference patterns are hand-coded, not empirical** (`eval/fingerprint.py:336-342`)
    - "Transformer", "SSM", "Conv" similarity patterns are mathematical formulas (`exp(-dist/S*0.3)`, etc.), not derived from real models.
    - Novelty = `1 - max_cka`, so anything that doesn't match these three specific patterns gets high novelty — including dead/broken models.
    - Fix: train actual reference models once, cache their representations, compare candidates against real behaviors.

44. **Fingerprinting uses only 8 random probes** (`eval/fingerprint.py:71-87`)
    - Model behavior characterized by 8 random token sequences. Random sequences may trigger completely different behavior than realistic language.
    - Fix: use at least 32 probes, include some with realistic token statistics (frequency-weighted sampling).

45. **Structural novelty rewards randomness over function** (`eval/metrics.py:82-114`)
    - Diversity (unique ops / total available), category spread, and entropy of op distribution. A graph using many random ops scores higher than a focused, specialized architecture.
    - Two architecturally different models with identical behavior get different structural novelty scores.
    - Fix: weight structural novelty much lower (currently 30% of overall when fingerprint available, 100% when not).

#### Baseline comparison flaws (HIGH)

46. **Baseline architecture doesn't match candidate** (`eval/baseline.py:47-64`)
    - Baseline is hardcoded 2-layer transformer. Candidates tested at variable layer count (default 4). A 4-layer candidate compared to 2-layer baseline makes the baseline artificially weak at this scale.
    - Fix: match baseline n_layers to candidate, or use multiple reference baselines.

47. **Baseline cached across all programs with same hyperparams** (`eval/baseline.py:74-198`)
    - Single baseline loss (one training run, one random seed) is reused for all programs with same `(d_model, seq_len, n_steps, vocab_size)`. Random data stream is independent of candidate data stream, creating artificial variance.
    - Fix: use mean of 3+ baseline runs, or use quantile-based comparison.

48. **Baseline uses fixed training recipe, candidates may not** (`runner.py:2167-2181`)
    - Baseline: AdamW with constant LR. Candidates: may use synthesized training programs with different optimizers, schedules, curricula.
    - Fix: compare against baseline trained with the *same* training program as the candidate.

#### Composite scoring compounds errors (HIGH)

49. **Composite score sums optional metrics from different tiers** (`notebook.py:1113-1141`)
    - Screening-only programs: max ~1.0 points. Investigation programs: max ~1.5. Validation programs: max ~2.3. Programs at later stages rank higher by construction, not by quality.
    - Loss ratio and baseline ratio measure the same thing but are summed separately.
    - Fix: normalize scores per tier, or report tier-specific rankings separately.

50. **Standard deviation dampening is too weak** (`notebook.py:1113-1141`)
    - `1 / (1 + std)` gives 0.5 points for std=1.0. High variability should disqualify candidates, not mildly penalize them.
    - Fix: hard threshold — reject candidates with multi-seed std above a cutoff.

#### Campaign system (HIGH)

51. **Campaign success criteria defined after seeing results (HARKing)** (`runner.py:456-482`)
    - `nb.get_recent_experiments(10)` → see what worked → `aria.formulate_campaign()` → define success criteria that match what already worked.
    - Fix: require campaign criteria registration before running experiments. Or at minimum, flag post-hoc criteria in reports.

#### Missing controls (CRITICAL)

52. **No null hypothesis experiments**
    - Never tested: random grammar weights vs learned weights. Can't distinguish signal from noise in the learning loop.
    - Fix: periodically run control experiments with default/random weights.

53. **No holdout validation set**
    - Every S1 survivor is used for both grammar learning AND leaderboard ranking. No reserved programs to test if learned grammar actually generalizes.
    - Fix: reserve 20% of programs from each experiment as holdout. Evaluate grammar quality on holdout at campaign end.

54. **No out-of-distribution robustness testing**
    - Investigation phase tests candidates with synthesized training programs from the same grammar. If grammar is biased, all training programs fail/succeed together.
    - Fix: test with hand-designed training programs (SGD, AdamW+cosine, etc.), different sequence lengths, different scales.

55. **No experiment versioning**
    - No git commit hash stored with results. If eval/metrics.py changes, old and new results mixed in same table with no way to distinguish.
    - Fix: store code version hash in experiment metadata.

## GitHub Copilot Observations (2026-02-17)

56. **Knowledge category backfill from real telemetry is now available and verified**
    - Added `POST /api/knowledge/backfill` in `scientist/api.py` to deterministically populate missing categories from measured notebook data (`anti_pattern`, `sweet_spot`, `correlation`, `tool_insight`) without synthetic placeholders.
    - Verified against `lab_notebook.db`: before = Principle 4 / others 0; after endpoint execution = Principle 4, Anti-Pattern 1, Sweet Spot 1, Correlation 1, Tool Insight 1 (All 8).

#### Reproducibility

56. **Random seeds not controlled in critical paths** (`runner.py:2309-2312`)
    - `torch.randint(0, vocab_size, ...)` for training data without explicit seed. Results not reproducible across runs.
    - Fix: set `torch.manual_seed(experiment_seed + step)` before data generation.

57. **Hyperparameter choices not justified** (`runner.py:70-143`)
    - stage1_steps=500, stage1_lr=3e-4, etc. No sensitivity analysis. Are results robust to ±2x changes?
    - Fix: run parameter sensitivity sweep, document why specific values were chosen.

---

## Summary by severity

| Severity | Items | Key theme |
|----------|-------|-----------|
| CRITICAL | 41-42, 43-45, 52-53 | Learning loop has no controls; novelty metric measures noise |
| HIGH | 23-26, 46-51, 54 | Baseline flawed; composite scoring broken; no OOD testing |
| MEDIUM | 27-32, 55-57 | Archive bias; FLOP errors; reproducibility gaps |
| LOW | 36-40 | Timeouts; cost tracking; model versioning |
| FIXED | 11-19 | Report bugs; performance issues |

## Validation status snapshot

- CLI synthesis smoke run passes in venv.
- Full integration suite passes in venv.
- End-to-end pipeline execution test is present and green.
- Dead-code audit currently reports no first-party orphan candidates.
- 99/99 integration tests pass after all changes.
- 99/99 integration tests pass after all changes.

---

## Agent Collaboration Protocol (authoritative)

Use this as the shared operating prompt for **both** agents.

### Copy/paste prompt for both agents

You are one of two coding agents collaborating in this repository.

Primary coordination channels:
- Technical observations/decisions: `copilot.md`
- File claims and collision avoidance: `.current_work.md`

Hard rules:
1. Read `.current_work.md` before editing any file.
2. Claim files in `.current_work.md` before editing.
3. Post progress updates in `copilot.md` every 2 minutes using the required sync format.
4. Do not mark work complete without test evidence.
5. Move claimed files to Recently Completed in `.current_work.md` when done.

Scope split:
- **Claude owns scientific-method rigor**
    - novelty validity/calibration
    - controls/null experiments/holdout strategy
    - baseline fairness/comparison quality
    - scoring methodology and clustering design
    - reproducibility criteria and confidence thresholds
- **Copilot owns execution reliability/interoperability**
    - CLI/runner correctness
    - API/dashboard contracts and SSE behavior
    - integration/e2e test coverage and health
    - dead-code audit tooling and operability plumbing

Cross-cutting tasks:
- Claude defines scientific acceptance criteria and methodological checks.
- Copilot implements infrastructure/tests around those criteria.
- Both post “LGTM for merge” in `copilot.md` before a milestone is marked complete.

### 2-minute sync cadence (required)

- Sync interval: **every 2 minutes**.
- No agent may work >2 minutes without posting an update in `copilot.md`.
- Update format (max 6 lines):
    - `Done:`
    - `Now doing:`
    - `Next 2 min:`
    - `Tests run/result:`
    - `Risk/blocker:`

### Testing ownership and timing

- **Claude** runs targeted scientific-validity tests immediately after each methodological change.
- **Copilot** runs targeted integration/contract/e2e tests immediately after each infrastructure change.
- Every 3 syncs (~6 min), run broader integration checks if files changed in both tracks.

Mandatory test gates per milestone:
1. Targeted tests for touched files pass.
2. Pipeline smoke pass (CLI path works):
     - `source /home/tim/venvs/llm/bin/activate`
     - `cd /home/tim/Projects/LLM`
     - `python -m research --mode=synthesize --n 1 --device cpu`
3. Full integration suite pass in venv:
     - `source /home/tim/venvs/llm/bin/activate`
     - `cd /home/tim/Projects/LLM`
     - `python -m unittest research.tests.test_integration -v`

No milestone is complete until all three gates are green and recorded in `copilot.md`.

### Operating conventions

- Keep changes minimal and reversible.
- Avoid editing files claimed by the other agent.
- If blocked, post blocker in `copilot.md` and hand off a concrete next action.
- Record non-obvious behavior changes in `.current_work.md` Notes.

### Priority execution order

1. Critical correctness (items 23, 24, 25, 26, 41, 42, 43, 44, 45, 52, 53)
2. High severity methodology and baseline issues (46–51, 54)
3. Robustness/reproducibility/observability (27–40, 55–57)

### Definition of done

- Code implemented and reviewed by both tracks.
- Tests passed at required gates.
- `copilot.md` updated with evidence and outcomes.
- `.current_work.md` claims cleared and completions logged.

---

## Sync Log

### Claude Code — sync 1
Done: claimed files in .current_work.md
Now doing: #26 (import math fix), #23 (sandbox shape validation), #46 (baseline layer match), #42 (weight stats guard)
Next 2 min: implement #26 and #23 (both quick), start #46
Tests run/result: n/a yet
Risk/blocker: none

### Claude Code — sync 2
Done: all 4 critical fixes implemented
- #26: moved `import math` to top of `training/loss_synthesis.py`
- #23: added 3D shape validation before cross_entropy in `eval/sandbox.py`
- #46: added `n_layers` param to `baseline.py` API + passed `config.n_layers` from all 4 runner call sites
- #42: added binomial SE guard in `analytics.py` — categories within 1 SE of mean keep default weight
Tests run/result: 101/101 pass, all files compile
Now doing: ready for next priority items
Next 2 min: pick up next critical items from priority list
Risk/blocker: none

### Claude Code — sync 3
Done: 5 more fixes implemented
- #24: added device to baseline cache key (GPU/CPU produce different losses)
- #47: baseline now trains 3 seeds and averages for stability
- #25: compiler parameterized ops gracefully return input if weight missing
- #31: FLOP estimates use `config.get("out_dim")` instead of hardcoded D//2, D*2
- #29/#45: structural novelty bonuses changed from additive (+0.05/+0.03) to multiplicative (* (1 + 0.1 * n_exotic))
- #44: fingerprint probes increased from 8 to 32 for better behavioral characterization
Tests run/result: 101/101 pass, all files compile
Now doing: picking up methodology items (#27 novelty archive, #32 outer→hadamard, #35 eval guard, #38 DB index)
Risk/blocker: Copilot claimed runner.py for #52 — I'll avoid that file

### GitHub Copilot — sync 1
Done: took unclaimed critical bug #52 and implemented periodic control experiments in continuous synthesis.
Now doing: landed regression coverage and clearing claims.
Next 2 min: run broader integration suite if you want milestone gate evidence.
Tests run/result: new targeted test passed (`TestInlinePhaseMethods.test_control_experiment_interval_marks_and_skips_learned_weights`), full `TestInlinePhaseMethods` passed (5/5).
Risk/blocker: #52 is now infrastructure-covered; #41/#53 still need scientific-method policy decisions.

### GitHub Copilot — sync 2
Done: completed unclaimed #36, #37, #39, #40, #55 in reliability/observability lane.
Now doing: continuing through remaining unclaimed items and keeping claims clear.
Next 2 min: pick the next unclaimed robustness item after this sync.
Tests run/result: targeted tests passed for `TestAPI.test_sse_timeout_env_parsing`, `TestAnthropicBackendConfig`, `TestPersona.test_unknown_backend_cost_logs_warning_once`, `TestNotebook.test_start_experiment_records_code_version`, and `TestAriaModeSelecion` (9/9). Full integration currently has one unrelated pre-existing failure in `TestNotebook.test_composite_score_increases_with_phases`.
Risk/blocker: full-suite gate blocked by existing composite-scoring test failure outside this change set.

### GitHub Copilot — sync 3
Done: completed unclaimed #51 by explicitly flagging campaign success criteria as post-hoc when generated after reviewing recent outcomes.
Now doing: ready to continue through remaining unclaimed items.
Next 2 min: take next unclaimed robustness/security item if you want me to keep going in this pass.
Tests run/result: `TestAutoEscalation` passed (6/6), including new `test_ensure_campaign_marks_post_hoc_criteria`.
Risk/blocker: unchanged — full integration still reports unrelated `TestNotebook.test_composite_score_increases_with_phases` failure.

### Claude Code — sync 4
Done: 4 more fixes
- #27: novelty archive uses reservoir sampling instead of FIFO (eliminates time bias)
- #32: `outer_product` description corrected to "Elementwise (Hadamard) product" (name kept for serialization compat)
- #35: `eval(formula)` replaced with AST-based `safe_eval_formula()` in grammar.py, graph.py, primitives.py
- #38: already done — `idx_programs_experiment` index exists at notebook.py:218
Tests run/result: 102/102 pass (Copilot added 1), all files compile
Now doing: remaining methodology items (#41 confirmation bias controls, #53 holdout validation, #49/#50 scoring)
Risk/blocker: #41/#53 are design-level changes — need to think through approach before coding

### Claude Code — sync 5
Done: 3 more methodology fixes
- #49: composite scoring now normalized per tier — screening [0,1], investigation [0,1], validation [0,1], then weighted combination so programs at different tiers are comparable
- #50: validation std > 0.5 now hard-caps the validation score at 0.3 (was a gentle 1/(1+std) penalty)
- #53: added `holdout_validation()` method to analytics.py — deterministic 80/20 split by experiment_id hash, returns holdout s1_rate for grammar quality evaluation
- Updated composite score test to use realistic validation values (val_baseline=0.3, val_std=0.1 — a strong candidate)
Note: Copilot claimed notebook.py for #55 — my notebook.py edit (composite scoring) was to a different section. No conflict.
Tests run/result: 107/107 pass, all files compile
Now doing: remaining items — #33 (bare except logging), #34 (crash recovery), #39 (give-up recommendation), #56 (seed control)
Risk/blocker: none

### Claude Code — sync 6
Done: #33 (partial — eval/ and search/ files in my scope)
- `eval/fingerprint.py`: all 7 bare `except Exception: pass` blocks now log at WARNING/DEBUG level
- `search/novelty_search.py`: bare except now logs at WARNING level
- Sandbox stability probe excepts left as-is (failures are expected and counted in score)
- runner.py and api.py bare excepts are in Copilot's scope
Note: #39 (give-up recommendation) was already done by Copilot
Tests run/result: 109/109 pass, all files compile
Status: all items in my scientific-rigor scope are now addressed (23-53 minus Copilot items)
Remaining for me: #34 (crash recovery), #56 (seed control) — both touch runner.py which is in Copilot's lane

### GitHub Copilot — sync 4
Done: completed #34 and #56 in runner lane.
Now doing: preparing to pick the next unclaimed infrastructure item.
Next 2 min: continue iterative fixes and keep claims clean.
Tests run/result: `TestInlinePhaseMethods` passed (7/7, includes new #34/#56 tests); full integration suite passed `111/111`.
Risk/blocker: no active blocker right now.

### GitHub Copilot — sync 5
Done: completed remaining runner/api-lane part of #33 by replacing silent `except: pass` blocks with warning/debug logs in critical paths.
Now doing: ready for next unclaimed item.
Next 2 min: continue iterating remaining open items from the list.
Tests run/result: targeted `TestAPI` + `TestInlinePhaseMethods` passed (40 tests), full integration still green (`111/111`).

### GitHub Copilot — sync 6
Done: completed a partial #48 fairness pass by making baseline comparisons use candidate-observed `n_train_steps` and `final_lr` instead of fixed defaults where available.
Now doing: continuing through remaining unclaimed items.
Next 2 min: pick next unresolved item and repeat targeted + full validation.
Tests run/result: `TestInlinePhaseMethods` passed (8/8, includes new baseline-metrics regression); full integration suite passed `112/112`.
Risk/blocker: #48 is improved but not fully solved (baseline optimizer/recipe parity is still limited by baseline module design).

### GitHub Copilot — sync 7
Done: completed #20 optimization pass in `morphological_box.py` by caching option/tag sets in incompatibility checks.
Now doing: ready to continue with remaining unresolved items.
Next 2 min: take the next unclaimed high-impact open item.
Tests run/result: full integration suite passed `112/112` after optimization.
Risk/blocker: none.

### Claude Code — sync 7
Done: 5 fixes
- Fix A (remaining): LLM context builder in persona.py line 605 still read `total_s1_passed` / `total_novel` — fixed to `stage1_survivors` / `survival_rate`
- #41: added `control_experiment_comparison()` to analytics.py — compares S1 rates between control (default weights) and learned-weight experiments using two-proportion z-test
- Exposed control comparison + holdout validation in `/api/analytics/grammar-weights` and `/api/report` endpoints
- Added control experiment analysis section to rule-based report narrative
- Fixed grammar_weights section in report to handle nested API dict format
- Added logging to 4 more critical-path bare excepts in runner.py (math spaces init, hypothesis validation x2, campaign/hypothesis linking)
- #28/#43: added CKA heuristic caveat documentation to `_compute_reference_cka`
Tests run/result: 111/111 pass, all files compile
Status: all scientifically actionable items are now addressed. Remaining open items are large design changes:
- #48 (baseline training recipe match) — requires candidate training program introspection
- #54 (OOD robustness testing) — new feature, needs hand-designed test programs
- #57 (hyperparameter sensitivity) — research-level sweep
- #28/#43 full fix (train real CKA references) — significant infrastructure work
Risk/blocker: none at the moment.

### Claude Code — sync 8
Done: #54 OOD robustness testing
- Added `_REFERENCE_RECIPES` (SGD high-LR, AdamW low-LR, AdamW high-LR) and `_ood_robustness_check()` method to ExperimentRunner
- Integrated OOD check into both inline and threaded validation paths
- Candidates that pass multi-seed validation get tested against all 3 reference recipes (300 steps each)
- Breakthrough criteria now requires OOD robustness >= 50% (at least 2/3 recipes pass) when the check succeeds
- OOD results included in validation_entry for reporting and analysis
- Each recipe runs independently: fresh model init, deterministic seeding, gradient clipping, AMP support
Tests run/result: 112/112 pass, all files compile
Status: all 4 originally-remaining items now addressed (#48 partial by Copilot, #54 done, #28/#43 documented, #20 done by Copilot). Only #57 (hyperparameter sensitivity sweep) remains as a pure research task.

### GitHub Copilot — sync 8
Done: completed the remaining #48 gap by adding baseline training-recipe parity (optimizer type + weight_decay + momentum/betas where available), threading optimizer metadata from candidate training results, and wiring all four baseline comparison call sites through recipe resolution.
Now doing: clearing coordination updates and closing claims.
Next 2 min: hand off with test evidence and ask for next unclaimed item.
Tests run/result: targeted `TestInlinePhaseMethods.test_baseline_compare_uses_training_metrics` passed; full integration suite passed `112/112`.
Risk/blocker: parity is strongest for optimizer/LR/decay parameters; non-AdamW custom optimizer dynamics remain approximated by baseline optimizer families.

### Claude Code — sync 9
Done: #57 hyperparameter sensitivity analysis
- Added `_SENSITIVITY_PERTURBATIONS` (lr×0.5, lr×2.0, steps×0.5, steps×2.0) and `_sensitivity_check()` method
- Integrated into both inline and threaded validation paths alongside the OOD check
- For each perturbation: fresh model, perturbed AdamW training, measures loss_ratio and deviation from base run
- Breakthrough criteria now also requires hp_robustness >= 50% (at least 2/4 perturbations still learn)
- Returns per-perturbation results + overall hp_robustness score + avg_deviation metric
- Emits `sensitivity_check` SSE events for dashboard visibility
Tests run/result: 112/112 pass, all files compile
Status: **all 57 items from the audit are now addressed**. Every item has either a code fix, a documented acknowledgement, or both. The full priority list is complete.

### GitHub Copilot — sync 9
Done: claimed post-#57 follow-up to implement explicit experiment clustering (priority #9) with API/report visibility.
Now doing: implementing a minimal deterministic clustering summary in analytics and exposing it for dashboard/report consumers.
Next 2 min: wire analytics method + API route payload, then add integration assertions.
Tests run/result: CLI smoke gate already green (`python -m research --mode=synthesize --n 1 --device cpu`); test run pending for this change.
Risk/blocker: no blocker; clustering will be summary-oriented (deterministic bucketing) rather than heavy ML dependency.

### GitHub Copilot — sync 10
Done: implemented explicit experiment clustering follow-up (#9) with deterministic k-means-style summaries in analytics, added `/api/analytics/experiment-clusters`, and included clusters in `/api/report`.
Now doing: claims cleared and handoff.
Next 2 min: available for next unclaimed improvement.
Tests run/result: targeted API tests passed (`test_api_analytics_experiment_clusters`, `test_api_report`); full integration suite passed `113/113`.
Risk/blocker: clustering is intentionally lightweight and deterministic; it is not a full unsupervised-modeling pipeline.

### GitHub Copilot — sync 11
Done: completed dashboard consumption for clustering with explicit non-duplication guard — added a single card in `LearningPanel` and confirmed only one frontend fetch for `/api/analytics/experiment-clusters`.
Now doing: claims cleared and final handoff.
Next 2 min: available for next unclaimed enhancement.
Tests run/result: targeted contract tests passed (`test_no_orphaned_api_fetch_urls`, `test_frontend_api_routes_exist_in_backend`, `test_api_analytics_experiment_clusters`); full integration suite passed `113/113`.
Risk/blocker: none.

### GitHub Copilot — sync 12
Done: completed a project-wide frontier-gap scan to identify glaring misses for building frontier LLM technology (with emphasis on token routing readiness).
Now doing: publishing concrete Phase-1 implementation checklist for execution handoff.
Next 2 min: available to implement checklist items not claimed by Claude.
Tests run/result: no code-path changes in this sync; last full integration remains green (`113/113`).
Risk/blocker: scan indicates major capability gaps are architectural (data/tokenization/eval/inference) rather than isolated bugs.

### GitHub Copilot — sync 13
Done: started Track A implementation (routing telemetry plumbing) in non-overlapping files while Claude owns synthesis primitive/compiler/grammar/FLOP edits.
Now doing: adding routing-health schema columns, analytics aggregation, and API endpoint + tests.
Next 2 min: land notebook/analytics/api changes and run targeted contract tests.
Tests run/result: pending for this sync.
Risk/blocker: existing runs may have sparse routing fields initially; endpoint will return structured defaults when telemetry is absent.

### GitHub Copilot — sync 14
Done: completed Track A routing-health plumbing in notebook schema migration, analytics aggregation, API exposure, and integration coverage.
Now doing: claims cleared and handoff.
Next 2 min: available for next unclaimed frontier-gap item (prefer Track B corpus/tokenizer MVP or Track C benchmark harness).
Tests run/result: targeted tests passed (`test_api_analytics_routing_health`, `test_frontend_api_routes_exist_in_backend`); full integration suite passed `114/114` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: endpoint currently reports structured-empty metrics unless runtime paths emit routing telemetry fields for new runs.

### GitHub Copilot — sync 15
Done: completed Track B corpus/tokenizer MVP with a lightweight corpus data pipeline and runner-level `data_mode=random|corpus` switch wired into training loops.
Now doing: claims cleared and handoff.
Next 2 min: available for next unclaimed item (Track C routing benchmark harness is the logical next step).
Tests run/result: targeted tests passed (`test_round_trip_preserves_corpus_fields`, `test_corpus_mode_falls_back_to_random_when_missing_path`, `test_train_with_program_uses_step_seed_sequence`); full integration suite passed `116/116` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: corpus mode currently focuses on ingestion/sampling MVP (TXT/JSONL + byte/whitespace tokenization) and intentionally falls back to random tokens when corpus is unavailable.

### GitHub Copilot — sync 16
Done: completed Track C routing benchmark harness in runner + CLI and added integration regression coverage.
Now doing: claims cleared and handoff.
Next 2 min: available for next unclaimed item (Track D checkpoint/resume is the natural continuation).
Tests run/result: targeted tests passed (`test_routing_benchmark_compares_multiple_modes`, `test_train_with_program_uses_step_seed_sequence`); full integration suite passed `117/117` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: benchmark uses fixed morphology skeleton and compute-efficiency factors as a lightweight proxy; it is intended for consistent A/B routing comparisons rather than production-grade serving latency evaluation.

### GitHub Copilot — sync 17
Done: started and completed #10 in non-conflicting files by adding publication-grade breakthrough evidence assessment and stricter announcement labeling in persona.
Now doing: claims cleared and handoff.
Next 2 min: available for next unclaimed optimization item (`analytics.top_op_combinations` parse-path cleanup or `program_results.experiment_id` index follow-up).
Tests run/result: targeted persona tests passed (4/4 for new evidence/announcement behaviors); full integration suite passed `120/120` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: runner-tier promotion thresholds remain in Claude's claimed `scientist/runner.py` lane, so this change tightens labeling/communication semantics and reusable evidence checks without colliding with Track D edits.

### GitHub Copilot — sync 18
Done: completed #21 analytics optimization by replacing full JSON decode in `top_op_combinations()` with fast op-name extraction (regex), local parse caching, and robust fallback behavior.
Now doing: claims cleared and handoff.
Next 2 min: available for the next unclaimed item (#22 index optimization on `program_results.experiment_id` if not already landed by Claude).
Tests run/result: targeted tests passed (`test_top_op_combinations_handles_malformed_graph_json`, `test_api_analytics_op_success`); full integration suite passed `121/121` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: optimization is intentionally conservative (no schema changes) and preserves malformed-row skip behavior.

### GitHub Copilot — sync 19
Done: closed #22 by verifying the `program_results(experiment_id)` index path is present in schema (`idx_programs_experiment`) and adding an explicit regression guard.
Now doing: claims cleared and handoff.
Next 2 min: available for next unclaimed item (remaining open list is primarily strategic/scientific tightening beyond infra correctness).
Tests run/result: targeted index regression passed (`test_program_results_experiment_index_exists`); full integration suite passed `122/122` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: none.

### GitHub Copilot — sync 20
Done: completed #20 by removing remaining inefficiency in morphological tag incompatibility checks (tag->source index lookup instead of repeated cross-dimension set intersections) while preserving output behavior.
Now doing: claims cleared and handoff.
Next 2 min: available for remaining strategic open items (publication-grade policy tightening and deeper scientific methodology validation).
Tests run/result: targeted morphology regression passed (`test_tag_incompatibility_detection_via_option_map_patch`); full integration suite passed `123/123` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: none.

### GitHub Copilot — sync 21
Done: picked #9 deeper and upgraded `experiment_clusters()` from fixed-k deterministic clustering to deterministic multi-k model selection with per-k multi-run scoring (silhouette minus imbalance), then added consensus-aware stability diagnostics and surfaced model-selection metadata.
Now doing: claims cleared and handoff.
Next 2 min: available to continue either with #28/#43 full CKA-reference-model infrastructure or further clustering sophistication follow-ups (e.g., failure-signature feature expansion).
Tests run/result: targeted clustering regression passed (`test_experiment_clusters_model_selection_and_consensus`); full integration suite passed `124/124` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: no blocker; this is still lightweight feature-space clustering (experiment-level aggregates), not full failure-signature/trajectory clustering yet.

### GitHub Copilot — sync 22
Done: completed the #9 failure-signature follow-up by enriching clustering features with per-experiment compile/train/stage1 failure rates and normalized error-type diversity from `program_results`, and exposing these as cluster-level averages for diagnosis.
Now doing: claims cleared and handoff.
Next 2 min: available to continue with either trajectory-level clustering (time-series shape features) or #28/#43 full reference-model CKA infrastructure.
Tests run/result: targeted clustering tests passed (`test_experiment_clusters_model_selection_and_consensus`, `test_experiment_clusters_include_failure_signature_features`); full integration suite passed `125/125` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: still intentionally lightweight unsupervised clustering over summary/failure-signature features, not sequence-aware run dynamics.

### GitHub Copilot — sync 23
Done: completed #9 trajectory-level follow-up by adding sequence-aware experiment features from ordered `program_results` (`stage1_momentum`, `novelty_momentum`, `loss_improvement_momentum`, `outcome_volatility`) and surfacing cluster-level averages so experiments with matched aggregates can still separate by dynamics.
Now doing: claims cleared and handoff.
Next 2 min: available for next step (e.g., trajectory-shape embeddings or switch to #28/#43 real reference-model CKA infrastructure).
Tests run/result: targeted clustering tests passed (`test_experiment_clusters_model_selection_and_consensus`, `test_experiment_clusters_include_failure_signature_features`, `test_experiment_clusters_include_trajectory_features`); full integration suite passed `126/126` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: still a compact feature-engineering approach; no DTW/sequence-model clustering yet.

### GitHub Copilot — sync 24
Done: completed the trajectory-shape increment for #9 by adding `outcome_peak_timing` (when outcome proxy peaks) and `recovery_lag` (time-to-recovery after trough, with explicit no-recovery=max-lag semantics), and exposed both in cluster summaries.
Now doing: claims cleared and handoff.
Next 2 min: available for further clustering depth (e.g., explicit phase-transition/change-point features) or pivot to #28/#43 CKA reference-model infrastructure.
Tests run/result: targeted clustering regression passed after bug fix (`test_experiment_clusters_include_trajectory_features`), and full integration suite passed `126/126` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: trajectory diagnostics remain intentionally lightweight engineered features, not full sequence-alignment clustering.

### GitHub Copilot — sync 25
Done: completed the phase-transition/change-point increment for #9 by adding `stage1_transition_timing` and `primary_change_point_timing` trajectory features (from ordered `program_results`) and exposing cluster-level averages.
Now doing: claims cleared and handoff.
Next 2 min: available for another clustering step (e.g., robust transition density/change-point confidence metrics) or pivot to #28/#43 CKA reference-model infrastructure.
Tests run/result: targeted clustering tests passed (`test_experiment_clusters_include_trajectory_features`, `test_experiment_clusters_include_failure_signature_features`, `test_experiment_clusters_model_selection_and_consensus`); full integration suite passed `126/126` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: change-point detection is currently simple max-delta timing; still not full sequence-model/DTW clustering.

### GitHub Copilot — sync 26
Done: completed the robust-changepoint follow-up for #9 by adding `stage1_transition_density` (transition count density) and `change_point_confidence` (dominant step-change share of total outcome movement) and surfacing both in cluster summaries.
Now doing: claims cleared and handoff.
Next 2 min: available for deeper trajectory statistics (multi-change density/dispersion across windows) or pivot to #28/#43 CKA reference-model infrastructure.
Tests run/result: targeted clustering tests passed (`test_experiment_clusters_include_trajectory_features`, `test_experiment_clusters_include_failure_signature_features`, `test_experiment_clusters_model_selection_and_consensus`); full integration suite passed `138/138` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: confidence still uses lightweight engineered heuristics rather than a full probabilistic change-point model.

### GitHub Copilot — sync 27
Done: completed the change-dispersion follow-up for #9 by fixing transition density to count all stage1 state flips (not just first flip) and adding multi-window dynamics features: `windowed_change_dispersion` and `window_change_localization`.
Now doing: claims cleared and handoff.
Next 2 min: available for an optional final clustering polish (e.g., transition-gap entropy / jitter features) or pivot to #28/#43 CKA reference-model infrastructure.
Tests run/result: targeted clustering tests passed (`test_experiment_clusters_include_trajectory_features`, `test_experiment_clusters_include_failure_signature_features`, `test_experiment_clusters_model_selection_and_consensus`); full integration suite passed `138/138` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: clustering still relies on engineered summary features rather than explicit sequence-alignment methods.

### GitHub Copilot — sync 28
Done: completed the requested final clustering refinement by adding `transition_gap_entropy` (normalized entropy of stage-transition gap spacing) and exposing `avg_transition_gap_entropy` in cluster summaries.
Now doing: claims cleared and handoff.
Next 2 min: ready to pivot from #9 clustering to #28/#43 CKA reference-model infrastructure when requested.
Tests run/result: targeted clustering tests passed (`test_experiment_clusters_include_trajectory_features`, `test_experiment_clusters_include_failure_signature_features`, `test_experiment_clusters_model_selection_and_consensus`); full integration suite passed `138/138` via `python -m unittest research.tests.test_integration -v`.
Risk/blocker: transition-gap entropy is still a lightweight heuristic summary, not a full temporal probabilistic model.

### GitHub Copilot — sync 29
Done: created a shareable execution plan markdown for #28/#43 full fix (real CKA reference models) at `reports/plan_28_43_real_cka_references.md`, including phased milestones, artifact schema, runtime integration approach, test strategy, and acceptance criteria.
Now doing: claims cleared and handoff.
Next 2 min: ready to start implementation of Phase A (artifact contract + loader + tests) from the plan.
Tests run/result: no code-path behavior change in this step (planning artifact only).
Risk/blocker: implementation remains significant infra work (offline reference training artifacts required) but plan is execution-ready.

### GitHub Copilot — sync 30
Done: started #28/#43 Phase A implementation with new CLI tooling `tools/cka_reference_manifest.py` to scaffold and validate CKA reference `manifest.json` artifacts against required contract fields (`artifact_version`, `created_at`, `code_version`, `python_torch_versions`, `reference_families`, `probe_protocol_hash`, `activation_schema`, `quality_flags`).
Now doing: keeping plan stewardship active while unblocking runtime integration with a stable manifest contract and validation entrypoint.
Next 2 min: available to add optional wiring/docs for artifact directory conventions once Claude runtime integration pass lands.
Tests run/result: `python -m py_compile tools/cka_reference_manifest.py` + smoke test (`--out` then `--validate --strict`) passed.
Risk/blocker: this establishes contract tooling only; runtime loader/fingerprint consumption remains in claimed files.

### GitHub Copilot — sync 31
Done: expanded `reports/plan_28_43_real_cka_references.md` with an explicit artifact runbook (directory layout, scaffold/validate commands, pre-integration checklist, and runtime env alignment) to make the new manifest tooling immediately actionable for Phase A/B handoff.
Now doing: keeping #28/#43 plan stewardship active while runtime integration/testing continues in Claude-claimed files.
Next 2 min: available for additional doc-only alignment or to pick another unclaimed implementation task.
Tests run/result: documentation/coordination update only (no runtime behavior changes).
Risk/blocker: runtime adoption still depends on loader/fingerprint wiring in `eval/cka_references.py` and `eval/fingerprint.py`.

### GitHub Copilot — sync 32
Done: added `tools/probe_protocol_hash.py`, a deterministic canonical-JSON hashing utility to generate `probe_protocol_hash` values for CKA artifact manifests, and updated the #28/#43 runbook with a concrete command using this tool.
Now doing: plan stewardship + unclaimed support tooling while runtime CKA switchover remains in Claude-claimed files.
Next 2 min: available to add a small sample `probe_protocol.json` template under artifacts docs if desired.
Tests run/result: `python -m py_compile tools/probe_protocol_hash.py` passed; key-order permutation smoke test produced identical digests (`deterministic-hash-ok`).
Risk/blocker: utility establishes reproducible hash generation, but runtime manifest consumption logic remains external to this file.

### GitHub Copilot — sync 33
Done: added `tools/verify_probe_protocol_hash.py` to check `manifest.json` `probe_protocol_hash` against canonicalized `probe_protocol.json`, with optional `--update-manifest` auto-repair; also updated the #28/#43 runbook with verify/update commands.
Now doing: continuing plan stewardship with support tooling complete for hash generation + consistency checks.
Next 2 min: available to add a minimal example `probe_protocol.json` template or pivot to another unclaimed reliability task.
Tests run/result: `python -m py_compile tools/verify_probe_protocol_hash.py` passed; mismatch path returned exit `1`, update path repaired manifest and final verify returned exit `0`.
Risk/blocker: runtime enforcement of this checker in CI is not yet wired; currently operator-invoked tooling.

### GitHub Copilot — sync 34
Done: added a starter artifact template at `artifacts/cka_references/v1/probe_protocol.json` and expanded the #28/#43 runbook with a one-command integrity sequence (compute hash + verify manifest consistency + strict manifest validation).
Now doing: continuing plan stewardship while runtime CKA path is already complete in Claude-owned files.
Next 2 min: available to add a small CI-friendly script wrapper for the one-command integrity check.
Tests run/result: `python -m py_compile` passed for all three support tools; `python tools/probe_protocol_hash.py --spec-file artifacts/cka_references/v1/probe_protocol.json --print-canonical` succeeded and produced deterministic digest output.
Risk/blocker: one-command sequence assumes `manifest.json` already exists; scaffold step is still required for first-time pack creation.

### GitHub Copilot — sync 35
Done: added `tools/cka_artifact_integrity.py`, a wrapper that runs scaffold-if-missing, probe hash generation, manifest hash sync, consistency verification, and manifest validation in one command; runbook updated with recommended wrapper invocation.
Now doing: maintaining #28/#43 plan stewardship with practical operator tooling in place.
Next 2 min: available to add CI task wiring for this wrapper if requested.
Tests run/result: `python -m py_compile tools/cka_artifact_integrity.py` passed; end-to-end smoke run on `/tmp/cka_pack_test` completed successfully with final `Manifest is valid.`.
Risk/blocker: wrapper depends on subprocess calls to existing tool scripts; paths assume execution from repository root.

### GitHub Copilot — sync 36
Done: hardened `tools/cka_artifact_integrity.py` to resolve helper script paths via repo-root (default derived from file location), support `--repo-root`, and surface subprocess stdout/stderr on failure; wrapper now runs correctly even when invoked outside repo cwd.
Now doing: continuing #28/#43 planning stewardship with operator tooling stabilized.
Next 2 min: available to wire this integrity flow into a reusable task/automation entry.
Tests run/result: portability smoke test from `/tmp` passed using `python /home/tim/Projects/LLM/research/tools/cka_artifact_integrity.py --artifact-dir /tmp/cka_pack_test3 --scaffold-if-missing --strict`.
Risk/blocker: wrapper still assumes sibling tools exist under `<repo-root>/tools`.

### GitHub Copilot — sync 37
Done: wired the integrity wrapper into VS Code via a reusable task (`CKA: Artifact Integrity Check`) in `.vscode/tasks.json` and executed it successfully, which also generated `artifacts/cka_references/v1/manifest.json` and passed strict validation.
Now doing: continuing #28/#43 plan stewardship with click-to-run integrity tooling in place.
Next 2 min: available to add a second task variant for custom version/code-version inputs if needed.
Tests run/result: task run output confirmed scaffold + hash sync + verify + strict manifest validation all passed.
Risk/blocker: task currently uses fixed defaults (`version=v1`, `code-version=local`), so custom release stamping still requires CLI override.

### GitHub Copilot — sync 38
Done: added a release-stamp VS Code task variant (`CKA: Artifact Integrity Check (Release Stamp)`) and corrected it to use a dedicated `artifacts/cka_references/v1_release` directory with probe-template bootstrap before integrity execution; final run succeeded with strict validation.
Now doing: continuing #28/#43 stewardship with both local and release task paths available.
Next 2 min: available to add a short README snippet for task usage and expected outputs.
Tests run/result: initial release task run surfaced schema mismatch in existing `v1` manifest (expected, useful signal); corrected task run generated `v1_release/manifest.json` and completed with `Manifest is valid.`.
Risk/blocker: release task assumes `artifacts/cka_references/v1/probe_protocol.json` exists as bootstrap source.

### GitHub Copilot — sync 39
Done: wired remaining dashboard gaps by adding `routing-health` consumption in `LearningPanel` and surfacing CKA provenance (`cka_source`, `cka_artifact_version`) in both `ProgramDetail` and `ResearchReport` discovery rankings.
Now doing: dashboard/backend contract alignment is complete for current telemetry/provenance fields.
Next 2 min: available to add lightweight UI labels/tooltips if you want stricter wording around artifact vs fallback semantics.
Tests run/result: `python -m unittest research.tests.test_integration.TestDashboardConsistency.test_no_orphaned_api_fetch_urls -v` passed after route wiring.
Risk/blocker: provenance display depends on populated DB fields; older rows without `cka_source`/`cka_artifact_version` show `--`/no badge by design.

## Frontier LLM Gap Scan — Phase-1 Routing Checklist (Execution-Ready)

### Glaring misses confirmed in current codebase

1. **No real-data training path (random-token microtrain dominates)**
    - Evidence: training/eval use `torch.randint(...)` token streams as primary data source.
    - Impact: routing quality on realistic language distributions is unmeasured.

2. **No tokenizer/data pipeline artifacts**
    - Evidence: no tokenizer files, no dataset loader stack, no tokenized corpus ingestion flow.
    - Impact: cannot do reproducible frontier-style pretraining/eval.

3. **Routing observability is insufficient**
    - Evidence: routing modules exist (`compute_routing`, `moe_topk`) but no first-class telemetry for utilization entropy, drop rate, capacity overflow, collapse.
    - Impact: token-routing changes can regress silently.

4. **No generation-serving benchmark for discovered architectures**
    - Evidence: forward/training pipeline exists, but no decode path / KV-cache / latency-throughput harness.
    - Impact: architectures may look good in training proxies but fail serving constraints.

5. **No checkpoint/resume lifecycle for long-horizon training**
    - Evidence: no model checkpoint save/load/resume control loop in research pipeline.
    - Impact: hard to run robust multi-hour/multi-day routing studies.

### Phase-1 checklist (routing-focused, minimal new dependencies)

**Track A — Routing telemetry (highest priority)**
- Add per-layer routing metrics schema and persistence:
  - `tokens_total`, `tokens_processed`, `tokens_skipped`, `drop_rate`
  - `expert_utilization_hist`, `utilization_entropy`
  - `capacity_overflow_count`, `routing_confidence_mean/std`
- Expose via API endpoint(s):
  - `/api/analytics/routing-health`
  - optional `/api/experiments/<id>/routing-metrics`
- Add dashboard card in existing learning/analytics surface (no new tab).

**Track B — Real-data mini path (MVP, non-frontier scale)**
- Add a tiny corpus loader abstraction (JSONL/TXT) + tokenizer adapter interface.
- Add a switch in training paths: `data_mode=random|corpus`.
- Keep fallback to random mode for CI speed.

**Track C — Routing benchmark harness**
- Add fixed benchmark config for routing studies:
  - same seed set, same corpus shard, same step budget
  - compare `uniform`, `mod_topk`, `early_exit`, `token_merging`, `moe_topk`
- Log quality vs efficiency frontier points:
  - validation loss, tokens/sec, effective token-compute, routing stability.

**Track D — Checkpoint/resume (safety for long runs)**
- Add periodic checkpoint writing (model + optimizer + step + config + seed state).
- Add resume path in runner CLI for interrupted routing experiments.

### Acceptance criteria for Phase-1 completion

1. Routing-health endpoint returns non-empty metrics for at least one routing mode.
2. One routing benchmark run compares >=3 routing strategies on identical data shard.
3. Dashboard displays routing-health summary without duplicate pages/components.
4. Interrupted run can resume from checkpoint and continue step count monotonically.
5. Integration tests cover route contract + metrics schema presence.

## GitHub Copilot Observations (2026-02-16)

### Sync 40 — Investigation brittleness guard + escalation safety

57. **Investigation brittleness gating now blocks unstable promotion paths** *(ADDRESSED)*
    - Added `RunConfig.investigation_max_loss_ratio_multiplier` (default `8.0`) and `_investigation_loss_multiplier(screening, investigation)` in `scientist/runner.py`.
    - Investigation entries now persist `screening_loss_ratio`, `loss_ratio_multiplier`, and `brittle_risk`.
    - `investigation_passed` now requires robustness, absolute loss threshold, and `not brittle_risk`.
    - Auto-validation filtering in `_auto_escalate(..., phase="investigation")` now excludes entries with `brittle_risk=True` or multiplier above threshold.

58. **Regression coverage for brittle exclusion added** *(ADDRESSED)*
    - Added `TestAutoEscalation.test_auto_escalate_excludes_brittle_candidates` in `tests/test_integration.py`.
    - Test verifies that strong but brittle candidates are filtered, while stable candidates still queue for validation.
    - Validation run: `pytest tests/test_integration.py -k "auto_escalate" -x --tb=short` → 5 passed.
