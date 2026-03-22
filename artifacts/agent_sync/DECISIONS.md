# Decisions

## Accepted
- Accept that `AGENTS.md` is a real repo-input gap.
  Reason: direct file search and direct read both failed; no substitute file named `AGENTS.md` exists in the workspace.

- Accept the audit conclusion that `local_window_attn` is the only implementation fix justified right now.
  Reason: direct audit evidence points to a real code-path issue and fresh valid-context rerun success.

- Accept that the highest-ROI work is a context-rule enforcement layer wired into generation, mutation, and validation.
  Reason: most low-S1 findings are context misuse, structural misuse, or stale telemetry rather than broken op code.

- Accept reclassification as a first-class engineering outcome.
  Reason: structural and niche ops should not be judged as standalone learners and should not be inserted into invalid contexts.

- Accept the Claude terminal rule audit.
  Reason: it matches the existing artifact evidence and identifies the smallest viable integration pattern: one shared context-policy map reused across templates, grammar/mutation, and validator.

- Accept hard rejection for structural misuse and invalid restricted-use placement.
  Reason: these are context violations, not weak-performance cases, and should fail loudly.

- Accept soft deprioritization, not blocking, for valid-but-weak components.
  Reason: the audit does not support turning weak-but-valid ops into forbidden ops.

- Accept the coordinator-local consolidation that makes `context_rules.py` the active policy owner for `grammar.py` and `validator.py`.
  Reason: this removes active runtime split-brain across modules and gives the low-S1 rule layer one authoritative entry point for generation and validation.

- Accept the targeted low-S1 regression tests added in `research/tests/test_context_rules.py`.
  Reason: they directly cover the required placement-rule behaviors that were previously untested.

## Rejected
- Reject broad component rewrites or deletions to improve low-S1 metrics.
  Reason: violates explicit hard rules and is not supported by audit evidence.

- Reject weakening S1 or reinterpreting low-S1 as automatic proof of broken code.
  Reason: the audit explicitly shows polluted evidence and valid-but-weak cases.

- Reject accepting any merge claim without file/path/test evidence.
  Reason: explicit hard rule from the task.

- Reject solving the low-S1 execution program only through `op_roles.py`.
  Reason: current role labels are too coarse to encode the audited predecessor/successor/motif/causal/residual constraints.

- Reject the current implementation draft as a merge target in its present form.
  Reason: the rule layer is duplicated between `templates.py` and `context_rules.py`, and the reviewed test diffs do not yet prove the required placement and mutation protections.

- Reject the unverified claims in earlier sync notes about `27` new tests and `156` regression-test passes.
  Reason: those were not validated in the coordinator-local review and should not be treated as accepted evidence.

## T1: local_window_attn shared-memory overflow — RESOLVED

- Accept the T1 fix as complete.
  Reason: Two root causes identified and fixed with evidence.
  1. `except (ImportError, RuntimeError, AttributeError)` did not catch `triton.runtime.errors.OutOfResources` — Triton error escaped, crashed forward pass, Python fallback never reached. Fix: broadened to `except Exception`.
  2. `templates.py` offered `window_size ∈ {8, 16, 32}` without dim check. At D≥256, W=32 deterministically exceeds GPU shared memory (151KB > 100KB limit). Fix: clamped to `{8, 16}` when `cur_dim >= 256`, with runtime clamp in the op handler as defense-in-depth.
  Verification: 14 regression tests pass. 4/4 default-search-context forward passes (was 0/4). Valid residual-attention context still produces learning signal.
  Residual risk: Triton kernel may have other undiscovered failure modes at exotic (S, D) combos. The `except Exception` fallback ensures Python path always catches these.

## T6: Structural-op S1 attribution exemption — IMPLEMENTED

- Accept the T6 implementation.
  Reason: Three attribution surfaces patched with minimal, targeted changes. 9 focused tests pass. 116 regression tests pass (0 failures).
  Changes:
  1. `context_rules.py`: added `S1_EXEMPT_STRUCTURAL_OPS` frozenset (10 ops: identity, split2, split3, concat, causal_mask, sliding_window_mask, norm_last, sum_last, mean_last, max_last).
  2. `analytics_grammar.py:_gather_category_stats()`: structural ops excluded from S1 total aggregation — they no longer drag down category weights.
  3. `analytics_experiments.py:compute_op_weights()`: structural ops excluded from eligible set and mean — they no longer get penalized in per-op weight feedback.
  4. `observability_bp.py:_get_component_health()`: structural ops classified as "structural" status, bypassing TF-IDF blame entirely — they no longer appear as "broken" or "degraded" in the dashboard.
  Verification: Tests prove structural ops excluded from category aggregation, per-op weights, and health blame. Tests prove non-structural ops still fully attributed. Tests prove model S1 screening (screening_rapid.py, sandbox.py) does not reference the exemption set.
  Residual risk: If a structural op is actually broken (e.g., concat producing wrong shapes), it will no longer be flagged by the health grid. Mitigation: compilation failures (S0) still surface in op_success_rates, and find_graph_context_violations catches structural misuse at generation time.

## Follow-up audit decisions

- Accept the broader `templates.py` work as intentionally retained code, not waste.
  Reason: user explicitly requested improvement-only handling and no rollback of broad template additions just for scope.

- Reject leaving duplicate live context-policy owners in place as a steady state.
  Reason: `templates.py` and `context_rules.py` currently disagree on `local_window_attn` classification and both still contain active policy helpers; that is a real correctness and maintenance risk, not just style debt.

- Reject silent catch-and-drop as an acceptable final pattern in active synthesis paths.
  Reason: verified `except ...: pass` sites remain in `grammar.py`, `validator.py`, and `templates.py`; these should be made explicit or instrumented when that cleanup is scheduled.

- Accept the niche `MATH_SPACE_RULES` additions for spiking / tropical / hyperbolic ops.
  Reason: focused tests pass and the changes match the audited domain-specific enforcement boundary better than the generic context-rule layer.

- Accept the `n_way_sparse_router` placement-rule tightening in `context_rules.py`.
  Reason: coordinator-local reproduction showed that invalid direct residual placement was still being accepted; focused tests now reject the bad form and keep normalized forms valid.

- Reject treating `n_way_sparse_router` as a confirmed compiler divisibility bug at this stage.
  Reason: coordinator-local compile/forward sweeps across non-divisible `(D, n_ways)` combinations did not reproduce a crash. The validated issue is placement, not divisibility.

- Accept the `n_way_sparse_router` bf16/autocast fix in `compiler.py`.
  Reason: a fresh forced rerun reproduced a real forward bug under `safe_eval(...)` (`scatter(): Expected self.dtype to be equal to src.dtype`), and the dtype-normalization patch removed the forward failure in the same graph.

- Reject immediate code rewrites for `geometric_product`, `tropical_matmul`, `sign_ste`, `log`, and `sqrt`.
  Reason: fresh targeted forced reruns now compile and forward in valid contexts; the observed remaining failure mode is poor rapid learning, which is not enough evidence for a code-path bug by itself.

- Reject treating `embedding_lookup` as still proven-broken based on the stale 0% table row.
  Reason: fresh targeted forced rerun passed compile, forward, and rapid screening. The remaining miss is weak S1 improvement over the tested 200-step micro-train, not a fresh forward-path failure.

- Reject immediate code surgery for `mod_topk`.
  Reason: fresh targeted forced rerun passed compile and forward; current evidence points to weak-learning behavior rather than a fresh implementation crash.

- Accept that `norm_last`, `sum_last`, `mean_last`, and `max_last` should be closed on the structural attribution path rather than revisited as component-code fixes.
  Reason: they are already covered by the structural S1 exemption set and corresponding attribution tests; treating them as standalone learner failures would contradict the audit and the accepted T6 change.

- Accept the broad recorded forced rerun campaign as the new reference evidence for under-observed ops.
  Reason: it completed with `430` recorded results and materially replaced stale low-S1 intuition with fresh compile/forward/rapid/S1 coverage.

- Reject treating the remaining weak-learning rows from the broad campaign as automatic code bugs.
  Reason: many ops now have strong fresh evidence of valid placement plus successful compile/forward or even rapid-pass behavior; the remaining miss is usually weak S1 learning, not implementation breakage.

- Accept `sparse_threshold` and `stdp_attention` as the highest-value remaining uncovered blockers.
  Reason: they were the only two ops not covered (`0/10`) in the recorded campaign and both sit in the spiking shape/space-constrained bucket.

- Accept the spiking reachability fix in `motifs.py` and `explore_under_observed.py`.
  Reason: the blocker was generation reachability, not runtime failure. After correcting the invalid motif chain and adding a direct forced-graph path, both `sparse_threshold` and `stdp_attention` now generate, compile, forward, and pass rapid screening.

- Accept the template policy-owner cleanup as effectively complete.
  Reason: `templates.py` no longer defines the duplicate context-policy tables or graph-violation helpers; the active owner remains `context_rules.py`.

- Accept the fresh post-fix evidence that `n_way_sparse_router` is in a better state than `local_window_attn`.
  Reason: in targeted reruns, `n_way_sparse_router` now reaches rapid pass and only fails at S1, while `local_window_attn` still fails at rapid learning in the tested forced graph.

- Reject a blind sweep over all remaining `templates.py` exception fallbacks.
  Reason: the remaining sites are heterogeneous behavior-preserving template fallbacks, not one repeated bug pattern. They need case-by-case review to avoid breaking intentional template degradation paths.

- Reject the hypothesis that simply simplifying the forced builders improves learning quality for `local_window_attn` / `n_way_sparse_router`.
  Reason: the direct residual builders did not improve `local_window_attn`, and they made `n_way_sparse_router` worse at rapid-stage learning than the earlier richer forced graph.

- Accept replacing the single forced wrapper for `local_window_attn` / `n_way_sparse_router` with a small curated family of valid wrappers.
  Reason: this preserves graph-space diversity while enforcing sane surrounding context, which matches the audit and the user requirement not to collapse exploration into one minimized graph.

- Reject using the latest noisy CUDA probe as quantitative learning-quality evidence for the new wrapper families.
  Reason: the ad hoc run was dominated by Triton `kernel_fallback` spam and did not produce clean coordinator-grade comparisons. The builder/test change is accepted; the learning delta remains unproven.
