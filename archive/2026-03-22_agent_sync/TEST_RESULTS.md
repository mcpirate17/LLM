# Test Results

## 2026-03-21

- Command: `rg --files | rg '^(AGENTS\\.md|CLAUDE\\.md|tasks/todo\\.md|artifacts/low_s1_components\\.md|artifacts/low_s1_root_cause_audit\\.md|artifacts/component_context_rules\\.md|artifacts/low_s1_fresh_reruns\\.md|artifacts/low_s1_before_after\\.md)$'`
  Outcome: `CLAUDE.md`, `tasks/todo.md`, and all required low-S1 artifacts were present; `AGENTS.md` was not listed.

- Command: `sed -n '1,220p' AGENTS.md`
  Outcome: failed with `No such file or directory`.

- Command: `rg --files | rg '(^|/)AGENTS\\.md$'`
  Outcome: no matches; `AGENTS.md` is absent repo-wide.

- Command: `sed -n '1,260p' CLAUDE.md` and required artifact reads
  Outcome: read completed; audit inputs support a rule-layer-first execution program.

- Command: `rg -n "local_window_attn|context rule|restricted-use|novelty insertion|mutation|builder|graph builder|selectability|search mode|structural-only" research tasks artifacts`
  Outcome: confirmed relevant implementation surfaces in `research/synthesis/*`, `research/search/evolution.py`, and tests under `research/tests/`.

- Command: `sed -n ... research/synthesis/motifs.py`, `research/synthesis/templates.py`, `research/synthesis/grammar.py`, `research/synthesis/validator.py`, `research/search/evolution.py`
  Outcome: identified current integration points for context enforcement and placement restrictions.

## Implementation Verification (2026-03-21, coordinator-local)

### Niche math-space rule verification
- Command: `pytest -q research/tests/test_math_space_rules.py`
- Result: `16 passed in 1.64s`
- Interpretation: verified spiking, tropical, and hyperbolic `MATH_SPACE_RULES` enforcement with focused acceptance and rejection coverage

- Command: `pytest -q research/tests/test_context_rules.py research/tests/test_synthesis_integration.py research/tests/test_ir_roundtrip.py`
- Result: `29 passed, 1 warning in 2.60s`
- Interpretation: niche-rule additions did not regress the active synthesis rule layer or IR roundtrip coverage

### New context-rule regression tests
- Command: `pytest -q research/tests/test_context_rules.py`
- Files changed: `research/synthesis/context_rules.py` (new), `research/tests/test_context_rules.py` (new)
- Result: `5 passed in 1.12s`
- Interpretation: Verified valid and invalid `local_window_attn` placement, structural `identity` misuse, and fresh/mutation generation respecting the rule layer when restricted ops are sampled.

### Exploration config smoke test
- Command: `pytest -q research/tests/test_under_observed_coverage.py -k 'exploration_config_creation or exploration_config_custom_boost'`
- Files changed: `research/synthesis/grammar.py`
- Result: `2 passed, 10 deselected in 0.94s`
- Interpretation: Verified the new `GrammarConfig.exploration(...)` path still behaves as expected.

### T1: local_window_attn shared memory overflow fix
- Command: `pytest -q research/tests/test_local_window_attn.py -m unit`
- Result: `14 passed in 2.42s`
- Tests cover:
  - Forward pass for all (D, W) combos: D∈{128, 256, 512}, W∈{8, 16, 32} — 9 parametrized tests
  - Window clamping at D≥256 (W=32→16)
  - W=32 OK at D<256
  - Causal masking correctness
  - Gradient flow through Python fallback
  - Valid residual-attention context: compile + forward + learning signal (20 steps)
- Repro of failing scenario: `GrammarConfig.exploration(frozenset(['local_window_attn']), boost_factor=20.0)` → 50 seeds → 4/4 forward pass, 0 failures (was 100% failure before fix)
- Files changed:
  - `research/synthesis/compiler_ops_attention.py` — clamped W to 16 for D≥256, broadened except clause to catch `triton.runtime.errors.OutOfResources`
  - `research/synthesis/templates.py` — restricted window_size choices to `[8, 16]` when `cur_dim >= 256`
  - `research/tests/test_local_window_attn.py` — 14 new regression tests

### Follow-up audit: duplicate policy / silent-catch scan
- Command: `rg -n "except .*:|\\bpass\\b|TODO|NotImplemented|raise NotImplementedError|return None$" research/synthesis/templates.py research/synthesis/grammar.py research/synthesis/validator.py research/synthesis/context_rules.py research/synthesis/compiler_ops_attention.py research/tests/test_local_window_attn.py research/tests/test_context_rules.py`
- Outcome: no stub markers found in reviewed files; concrete silent catch-and-drop sites remain in `research/synthesis/grammar.py`, `research/synthesis/validator.py`, and `research/synthesis/templates.py`

- Command: `rg -n "_OP_CONTEXT_CLASS|_MOTIF_TEMPLATE_ALLOWLIST|find_graph_context_violations|apply_context_rule_priors|CONTEXT_CLASS_" research/synthesis/templates.py research/synthesis/context_rules.py`
- Outcome: confirmed duplicated context-policy tables and helpers in both modules; `local_window_attn` classification disagrees (`restricted-use` in `templates.py`, `rehab` in `context_rules.py`)

- Command: `rg -n "apply_context_rule_priors\\(|find_graph_context_violations\\(|_op_context_class\\(|_motif_context_class\\(|_motif_allowed_in_template\\(" research/synthesis/templates.py`
- Outcome: confirmed template-local policy helpers remain behaviorally live; `_motif_allowed_in_template(...)` is still used by motif selection in `templates.py`

### Worktree state for scoped files
- Command: `git status --short -- research/synthesis/context_rules.py research/synthesis/grammar.py research/synthesis/validator.py research/tests/test_context_rules.py`
- Result: `research/synthesis/grammar.py` and `research/synthesis/validator.py` modified; `research/synthesis/context_rules.py` and `research/tests/test_context_rules.py` present as untracked new files.

### T6: Structural op S1 exemption
- Command: `pytest -q research/tests/test_structural_s1_exemption.py`
- Files changed:
  - `research/synthesis/context_rules.py` — added `S1_EXEMPT_STRUCTURAL_OPS` frozenset (10 ops)
  - `research/scientist/analytics/analytics_grammar.py` — `_gather_category_stats()` skips structural ops from S1 total
  - `research/scientist/analytics/analytics_experiments.py` — `compute_op_weights()` excludes structural ops from eligible set and mean
  - `research/scientist/api_routes/observability_bp.py` — `_get_component_health()` classifies structural ops as "structural" (bypass TF-IDF blame)
  - `research/tests/test_structural_s1_exemption.py` — 9 new tests
- Result: **9 passed in 1.37s**
- Tests cover:
  - Category weight exemption: structural ops excluded from S1 total aggregation
  - Non-structural ops still fully counted in category stats
  - All 10 exempt ops actually excluded from category aggregation
  - Per-op weight exemption: structural ops excluded from mean and output
  - Mean not dragged down by zero-S1 structural ops
  - Exempt set correctness (matches coordinator's 10-op target list)
  - Non-structural ops confirmed NOT in exempt set
  - Model S1 screening code does not reference exemption set (screening_rapid, sandbox)

### T6 regression check
- Command: `pytest -q research/tests/test_context_rules.py research/tests/test_structural_s1_exemption.py research/tests/test_component_health.py research/tests/test_recommendation_aggregates.py research/tests/test_observability_api.py research/tests/test_notebook.py`
- Result: **116 passed, 0 failures**
- Interpretation: No regressions in context rules, component health, recommendation aggregates, observability API, or notebook tests.

### n_way_sparse_router placement-rule tightening
- Command: `pytest -q research/tests/test_context_rules.py`
- Result: `7 passed in 1.21s`
- Interpretation: added focused coverage for valid and invalid `n_way_sparse_router` placement

- Command: `pytest -q research/tests/test_math_space_rules.py research/tests/test_context_rules.py research/tests/test_synthesis_integration.py research/tests/test_ir_roundtrip.py`
- Result: `47 passed, 1 warning in 3.09s`
- Interpretation: combined niche-rule and router-rule changes passed focused and integration synthesis coverage

- Command: inline `_validate_graph(...)` spot-check for `rmsnorm -> n_way_sparse_router -> add` and `rmsnorm -> n_way_sparse_router -> rmsnorm -> add`
- Outcome: direct-add graph rejected with `n_way_sparse_router must feed rmsnorm/layernorm/linear_proj, not stand alone`; renormalized graph accepted

- Command: inline compile/forward sweep across `(D, n_ways)` pairs including non-divisible cases
- Outcome: no coordinator-local reproduction of the previously suspected divisibility crash in `research/synthesis/compiler.py`

### Targeted forced reruns: geometric_product / tropical_matmul / n_way_sparse_router
- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, s1_steps=200, rapid_steps=200)` for `geometric_product`, `tropical_matmul`, and `n_way_sparse_router`
- Outcomes:
  - `geometric_product`: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=10.725 final=10.609`)
  - `tropical_matmul`: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=10.532 final=10.476`)
  - `n_way_sparse_router`: compile pass, forward **failed** before code fix with `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`

### n_way_sparse_router bf16/autocast bug fix
- Command: `pytest -q research/tests/test_n_way_sparse_router_regression.py`
- Result: `1 passed in 2.16s`
- Interpretation: exact `safe_eval(...)` reproduction no longer hits the scatter dtype mismatch

- Command: inline `generate_forced_graph('n_way_sparse_router', ...)` + `evaluate_graph(..., device='cuda', run_s1=True, s1_steps=200, rapid_steps=200)` after the dtype fix
- Outcome: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=11.059 final=10.847`)

- Command: `pytest -q research/tests/test_math_space_rules.py research/tests/test_context_rules.py research/tests/test_n_way_sparse_router_regression.py`
- Result: `24 passed in 3.99s`
- Interpretation: the router dtype fix and placement rule changes did not regress the focused synthesis rule suite

### Targeted forced reruns: sign_ste / log / sqrt
- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, s1_steps=200, rapid_steps=200)` for `sign_ste`, `log`, and `sqrt`
- Outcomes:
  - `sign_ste`: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=10.571 final=10.424`)
  - `log`: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=10.877 final=10.810`)
  - `sqrt`: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=11.134 final=10.914`)

### Targeted forced reruns: embedding_lookup / mod_topk
- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, s1_steps=200, rapid_steps=200)` for `embedding_lookup` and `mod_topk`
- Outcomes:
  - `embedding_lookup`: compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9867275066433112`)
  - `mod_topk`: compile pass, forward pass, rapid fail (`No learning after 200 steps: init=10.734 final=10.589`)

### Structural attribution verification
- Command: `sed -n '1,240p' research/tests/test_structural_s1_exemption.py`
- Outcome: verified existing focused coverage for `norm_last`, `sum_last`, `mean_last`, and `max_last` as structural S1-exempt ops

### Spiking coverage unblock: sparse_threshold / stdp_attention
- Command: `pytest -q research/tests/test_under_observed_coverage.py -k 'forced_generation_covers_spiking_threshold_and_stdp or exploration_config_creation or exploration_config_custom_boost'`
- Result: `4 passed, 10 deselected in 1.10s`
- Interpretation: forced graph generation now reaches both previously uncovered spiking ops

- Command: `pytest -q research/tests/test_math_space_rules.py research/tests/test_under_observed_coverage.py -k 'math_space_rules or forced_generation_covers_spiking_threshold_and_stdp'`
- Result: `18 passed, 12 deselected in 1.90s`
- Interpretation: spiking reachability fix did not regress the accepted math-space rule suite

- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=False, rapid_steps=200)` for `sparse_threshold` and `stdp_attention`
- Outcomes:
  - `sparse_threshold`: compile pass, forward pass, rapid pass
  - `stdp_attention`: compile pass, forward pass, rapid pass

- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, rapid_steps=200, s1_steps=200)` for `sparse_threshold` and `stdp_attention`
- Outcomes:
  - `sparse_threshold`: compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9851205470194088`)
  - `stdp_attention`: compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9980838167519692`)

### Template policy-owner cleanup verification
- Command: `rg -n "def find_graph_context_violations|def apply_context_rule_priors|_OP_CONTEXT_CLASS|_MOTIF_TEMPLATE_ALLOWLIST|_CONTEXT_CLASS_PRIORS|CONTEXT_CLASS_" research/synthesis/templates.py`
- Outcome: no matches; `templates.py` no longer carries the duplicate context-policy tables or validator helpers

- Command: `pytest -q research/tests/test_context_rules.py research/tests/test_math_space_rules.py research/tests/test_under_observed_coverage.py -k 'forced_generation_covers_spiking_threshold_and_stdp or context_rules or math_space_rules or exploration_config_creation or exploration_config_custom_boost'`
- Result: `27 passed, 10 deselected in 1.79s`
- Interpretation: context-rule ownership remains stable after the cleanup and spiking reachability changes

### Post-fix targeted reruns: local_window_attn / n_way_sparse_router
- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, rapid_steps=300, s1_steps=300)` for `local_window_attn` and `n_way_sparse_router`
- Outcomes:
  - `local_window_attn`: generate pass, compile pass, forward pass, rapid fail (`No learning after 300 steps: init=10.640 final=10.556`)
  - `n_way_sparse_router`: generate pass, compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9855114465795735`)

### Learning-quality push: simplified forced builders
- Change: `research/tools/explore_under_observed.py` now forces direct residual builders for `local_window_attn` and `n_way_sparse_router`
- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, rapid_steps=300, s1_steps=300)` after the builder change
- Outcomes:
  - `local_window_attn` forced graph: `rmsnorm -> local_window_attn(window_size=16) -> linear_proj -> add`; compile pass, forward pass, rapid fail (`init=10.648 final=10.458`)
  - `n_way_sparse_router` forced graph: `rmsnorm -> n_way_sparse_router -> rmsnorm -> add`; compile pass, forward pass, rapid fail (`init=10.807 final=10.796`)
- Interpretation: simpler forced graphs did not improve learning quality in this evaluation path; for `n_way_sparse_router`, the earlier richer forced graph had better rapid-stage behavior

### Silent-catch cleanup pass
- Command: `rg -n "except .*:|\\bpass\\b" research/synthesis/grammar.py research/synthesis/validator.py research/synthesis/templates.py`
- Outcome:
  - bare `pass` sites previously cleaned in `grammar.py`, `validator.py`, and two high-signal sites in `templates.py`
  - many non-`pass` `except ValueError` / `except KeyError` fallbacks still remain in `templates.py` and require case-by-case review rather than a blind sweep

### Forced-wrapper family broadening: local_window_attn / n_way_sparse_router
- Change: `research/tools/explore_under_observed.py` now uses seed-selected curated wrapper families for `local_window_attn` and `n_way_sparse_router` instead of one direct residual graph per op
- Command: `pytest -q research/tests/test_under_observed_coverage.py -k 'forced_generation_covers_spiking_threshold_and_stdp or forced_generation_uses_multiple_valid_wrapper_variants or exploration_config_creation or exploration_config_custom_boost'`
- Result: `6 passed, 10 deselected in 1.22s`
- Interpretation: forced generation now preserves multiple valid wrapper variants for both targeted ops

- Command: `pytest -q research/tests/test_context_rules.py research/tests/test_math_space_rules.py research/tests/test_n_way_sparse_router_regression.py`
- Result: `24 passed in 3.30s`
- Interpretation: the broader forced-wrapper families did not regress the accepted rule or router-fix coverage

- Command: inline `generate_forced_graph(...)` + `evaluate_graph(..., device='cuda', run_s1=True, rapid_steps=150, s1_steps=150)` sample across seeds `42, 43, 44` for `local_window_attn` / `n_way_sparse_router`
- Outcome: not accepted as decision evidence; the probe emitted heavy Triton `kernel_fallback` spam for `local_window_attn` and did not produce a clean coordinator-grade comparison

## Rerun Tool Verification

- Command: `python -m research.tools.explore_under_observed --help`
  Outcome: available. Supports `--mode {weighted,forced}`, `--threshold`, `--graphs-per-op`, `--rapid-steps`, optional `--no-s1`, `--s1-steps`, `--device`, and `--record`.

- Planned coordinator rerun command:
  `python -m research.tools.explore_under_observed --mode=forced --threshold=50 --graphs-per-op=10 --rapid-steps=500 --n-graphs=500 --device=cuda --record`
  Outcome: planned, not yet executed.

## Broad recorded rerun campaign (2026-03-21)

- Command: `python -m research.tools.explore_under_observed --mode=forced --threshold=50 --graphs-per-op=10 --rapid-steps=500 --n-graphs=500 --device=cuda --record`
- Outcome:
  - Experiment id: `9df0473d-c96`
  - Process completed and wrote reports:
    - `research/reports/exploration_20260321_145219.md`
    - `research/reports/exploration_20260321_145219.json`
  - DB recording complete: `430` results
  - Final summary from report: `44/46` covered, `44` compile-pass, `44` forward-pass, `41` rapid-pass, `1` S1-pass
  - Final DB stage totals: `402` stage0 pass, `417` stage0.5 pass, `1` stage1 pass
  - Error-type breakdown in `program_results` for this experiment:
    - blank / no explicit error type: `228`
    - `rapid_screening_error`: `174`
    - `forward_error`: `28`
  - Uncovered ops: `sparse_threshold`, `stdp_attention`
  - Sole S1-passing op in this campaign: `hyp_distance`
  - Rapid-pass but still S1-fail examples relevant to current triage: `embedding_lookup`, `log`, `sqrt`
  - Rapid-fail examples relevant to current triage: `tropical_matmul`, `sum_last`, `token_merge`
