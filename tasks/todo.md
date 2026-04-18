# Task Plan

## 2026-04-17 — Open-depth audit for rule design

- Run the controlled `open_depth` audit described in [tasks/open_depth_audit_tomorrow.md](/home/tim/Projects/LLM/tasks/open_depth_audit_tomorrow.md).
- Keep production depth/op caps unchanged; this is a test-only experiment to learn which failures should become better op, slot, template, and assembly rules.
- Deliver structured failure distributions, deep-survivor inventory, and proposed local-rule upgrades based on observed breakage rather than intuition.

## 2026-04-16 — claude-opus — Capability-first discovery: beat GPT-2 on ppl + binding + induction + ar + hs simultaneously

### What we already have (landed by codex, do not redo)
- Role-slot taxonomy — `research/synthesis/_template_role_slots.py` (8 roles: `trunk_compression`, `local_mixing`, `global_retrieval`, `binding_write`, `binding_read`, `controller`, `merge_policy`, `stabilizer`).
- Three trunk+sidecar templates — `research/synthesis/_templates_role_slots.py`:
  - `typed_slot_memory_block` — conv trunk + typed-entropy gate + matmul/gather_topk sidecar → gated add.
  - `sparse_relation_graph_block` — conv trunk + route_topk relation edges + matmul message pass → gated add.
  - `token_program_interpreter_block` — n-way sparse router + token-merge memory + matmul/gather read → gated add.
- Registered in `templates.py` with weights 4.5 / 4.0 / 4.25 (`DEFAULT_TEMPLATE_WEIGHTS` L769-771) and in `COMPONENT_GRAPH_EXEMPT_TEMPLATES` (L152-154).
- Role-slot telemetry piggybacks on existing `template_slot_usage` channel — visible in observability without schema change.

### Why this alone will NOT close the gap
The templates open a new search surface but four downstream pressures still favor ppl-only winners:
1. **Scoring is ppl-dominant** (ppl=40, binding=5, induction=3, ar=2). A zero-binding graph still wins composite.
2. **Screening has no binding gate** — SSM/conv-only graphs advance to investigation and burn deep-validation compute.
3. **Grammar's `routing_first` preset** does not include the new role-slot templates in `ROUTING_TEMPLATES` (verify).
4. **Op-pair priors missing** — a `global_retrieval` slot picking bare `matmul` is not the same as `matmul + softmax` or `cosine + gather_topk`.

### Steps (ordered by leverage; verify each before moving on)

**Step 1 — Wire role-slot templates into `routing_first` preset.**
- Edit `research/synthesis/grammar.py` around L292-347 so that `routing_first` treats `typed_slot_memory_block`, `sparse_relation_graph_block`, `token_program_interpreter_block` as first-class routing templates (weight 5.0, others zeroed). Alternatively add a sibling preset `capability_first` with these three + a compatible low-ppl trunk (e.g. `conv_ssm_block`).
- Rationale: codex left the templates registered but not promoted. Without this the sampler will pick them at ~4.5 weight while other templates sit at 1.0 — OK but not pressure.

**Step 2 — Add screening gate `gate8_retrieval_dead`.**
- File: `research/scientist/runner/execution_screening_graphs.py`.
- Condition: when `routing_mandatory=True` AND graph contains NO op from `CONTENT_ADDRESSED_OPS` AND NO `matmul`/`outer_product`/`gather_topk`/`cosine_similarity` AND graph has no `binding_range_class in {"full","medium"}` mixer → fail.
- Add a tiny 32-step binding probe as secondary gate (optional, gated by config flag, cheap on CPU).
- Rationale: stops conv/SSM-only retrieval-dead graphs at screening instead of burning investigation compute.

**Step 3 — Rebalance composite: binding as multiplier, not additive.**
- File: `research/scientist/leaderboard_scoring.py` (composite_v8 around L870-980, `binding_penalty` L773-810).
- Change the "all three binding signals below threshold" penalty from `×0.80` to `×0.50` (or promote to a named v8.1 dispatcher so we can A/B).
- Additionally: when `binding_composite >= 0.05`, apply a `×1.15` composite boost so non-ppl dimensions can pay for ppl loss.
- Rationale: makes a ppl=11 graph with zero binding score lower than a ppl=13 graph with real binding — disciplines the search toward the tuple, not the marginal.

**Step 4 — Op-pair priors for `global_retrieval` slot.**
- File: `research/synthesis/primitives.py` (`OP_WIRING_RULES` around L1460-1520).
- When the producer of an op's input is flagged `role:global_retrieval`, bias the consumer pool toward paired ops: `matmul → softmax → mul`, `cosine_similarity → gather_topk`, `outer_product → reduce_sum`. Keep legal-bare matmul as a minority path (maybe 20%).
- Rationale: today matmul in a retrieval context is treated identically to matmul as a generic binary op. Pairing enforces real content-addressed retrieval.

**Step 5 — Seed breeding runs from known donors.**
- Two targeted runs (script under `research/tools/seed_breed.py`, or extend existing `register_references.py`):
  - (a) `bb120386-3bc` mutation children — keep its `token_type_classifier → matmul → entropy_score → mul` retrieval substring intact, mutate the surrounding trunk. Constrain mutation ops to the ppl-helper family (conv1d_seq, selective_scan, swiglu_mlp, token_merge, nm_sparse_linear).
  - (b) `903157e5-219` × low-ppl-trunk hybrids — graft its retrieval module onto the trunk of `e8ab9e69-b38` / `a553c42b-11b` / `c9e0476e-210`. This is the explicit "retrieval donor + ppl donor" test.
- Rationale: gives the next 200-candidate batch concrete starting points instead of random priors.

**Step 6 — Role-slot v2 variants of the top three legacy templates (codex's suggested follow-up).**
- Pick the three highest-mean-ppl legacy templates from the leaderboard. Preserve their trunks; swap hardcoded attention/FFN internals for `global_retrieval` / `binding_read` role slots via `pick_role_motif()`.
- Register as `<name>_v2` so the frontier stays comparable.
- Rationale: widens the search space to let the role taxonomy cross-pollinate proven trunks. Safe — v2 suffix keeps A/B clean.

**Step 7 — Observability verification.**
- Query `template_slot_usage` to confirm `role:global_retrieval`, `role:binding_read`, `role:merge_policy` entries appear for candidates using the new templates.
- Dashboard: extend `TemplateSlotObservability.js` to surface a role-slot rollup (counts of selected motifs per role).
- Rationale: we need to see whether the new templates actually exercise the retrieval sidecar or collapse it during simplification.

### Verification gates (run after each step)
- `pytest tests/test_templates.py tests/test_grammar.py -x --tb=short` for steps 1, 6.
- `pytest tests/test_screening*.py -x --tb=short` for step 2.
- `pytest tests/test_scoring*.py tests/test_leaderboard*.py -x --tb=short` for step 3.
- `pytest tests/test_primitives*.py tests/test_wiring*.py -x --tb=short` for step 4.
- End-to-end smoke: `python -m research.tools.register_references --arch all --device cpu` before declaring each step done.

### Explicit non-goals
- No new primitives. The registry has everything we need; the bottleneck is topology + priors + scoring.
- No new GLA variants. GLA is a ppl helper, not a binding champion — data says don't waste search budget there.
- No retrofitting legacy template slot definitions in place (per codex's comparability argument). Use `_v2` suffixes for step 6.
- No destructive leaderboard cleanup while this lands — keep historical runs for before/after comparison.

### Success criteria
After steps 1-4 land and 200+ candidates run through the new pipeline, at least one graph on the candidate-comparable frontier must satisfy:
- `wikitext_perplexity < 13.0` AND
- `binding_auc > 0.10` AND
- `induction_auc > 0.05` AND
- `ar_auc > 0.03` AND
- `hellaswag_acc > 0.29`

If that tuple remains empty after steps 1-4, the topology prior was necessary but not sufficient — revisit scoring/probe weights before adding more templates.

### 2026-04-16 — Execution summary (steps 1-7 all landed, all tests green)

**Files changed:**
- `research/synthesis/_templates_routing.py` — added role-slot templates to `ROUTING_TEMPLATES`, new `CAPABILITY_FIRST_TEMPLATES` frozenset.
- `research/synthesis/grammar.py` — added `GrammarConfig.capability_first()` preset + `binding_capable_required` field.
- `research/synthesis/templates.py` — re-exported `CAPABILITY_FIRST_TEMPLATES`, registered 3 v2 templates in `TEMPLATES`, `DEFAULT_TEMPLATE_WEIGHTS`, `COMPONENT_GRAPH_EXEMPT_TEMPLATES`.
- `research/synthesis/_templates_role_slot_v2.py` — NEW — `conv_residual_retrieval_v2`, `state_space_retrieval_v2`, `latent_attn_retrieval_v2` plus shape-safe `_retrieval_sidecar` helper.
- `research/synthesis/_template_role_slots.py` — added `_retrieval_biased_weights()` biasing `global_retrieval` / `binding_read` / `binding_write` motif selection toward matmul/outer/cosine/gather_topk.
- `research/synthesis/primitives.py` — added `preferred_pair_consumer` hints on matmul/outer_product/cosine_similarity/gather_topk + `retrieval_pair_for()` / `RETRIEVAL_PAIR_OPS`.
- `research/scientist/runner/execution_screening_graphs.py` — expanded `CONTENT_ADDRESSED_OPS` to include bare retrieval ops; added `gate8_retrieval_dead` (opt-in via `binding_capable_required`).
- `research/scientist/runner/execution_screening.py` — threaded `binding_capable_required` from config to gate.
- `research/scientist/leaderboard_scoring.py` — added `_V8_1_CONFIG`, `compute_composite_v8_1()`, env-var-aware `SCORING_VERSION` dispatcher; threaded `cfg` into `_apply_scoring_penalties` so the penalty/boost multipliers come from config.
- `research/tools/seed_breed.py` — NEW — donor-driven mutation breeder under `capability_first` grammar.
- `research/tests/test_role_slot_telemetry.py` — NEW regression for role-slot telemetry + retrieval op presence.

**Verified:**
- v8.1 scoring flips ordering: ppl-only graph 43.53 → 27.21 (×0.625), binding graph 59.06 → 67.92 (×1.15).
- All three v2 templates compile + forward cleanly (n_ops=10-11, matmul in every graph).
- `gate8_retrieval_dead` rejects CONTENT_ADDRESSED-free graphs when `binding_capable_required=True`, stays inert otherwise.
- `capability_first` preset promotes the 6 role-slot templates to weight 6.0, keeps 5 ppl trunks at 1.0-1.5, zeros everything else.
- 349/349 tests pass across synthesis, scoring, screening, wiring, templates, telemetry.

**Known limitations / follow-ups:**
- v2 sidecar currently omits the typed-entropy gate because `linear_proj_up` misbehaves when input has D=1 (entropy_score output shape). Op-level fix to `linear_proj_up` would let us reinstate the gated variant.
- `seed_breed.py` is a CLI that prints mutants to JSON; direct injection into the experiment queue is a future step once the runner exposes a programmatic seed API.
- `depth_weighted_proj` Rust backward kernel — optional perf upgrade for later; current PyTorch fallback is correct but slower.

**Follow-ups closed 2026-04-16 evening:**
- ✅ Scoring version selector now runtime-switchable via `/api/scoring/version` GET/POST + Advanced panel dropdown. Historical rows stay on prior version.
- ✅ Role-slot rollup panel in `TemplateSlotObservability.js` aggregates `role:*` telemetry by capability slot.
- ✅ Notebook write pipeline unstuck: aria-db Rust layer was enforcing `PRAGMA foreign_keys=ON` (Python sqlite3 default is OFF), causing silent `FOREIGN KEY constraint failed` drops of `program_results` rows for ~16 hours. Fixed in `research/runtime/native/rust/aria-db/src/lib.rs:219` + regression test + louder stderr logging on writer errors.
- ✅ `depth_weighted_proj` and 5 aliased ops (`gated_lane_blend`, `route_lanes`, `depth_gated_transform`, `route_recursion`, `adaptive_recursion`) excluded from native C kernel dispatch because forward has no backward kernel — forced PyTorch path; regression test pins the exclusion until the Rust backward lands.
