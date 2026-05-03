# Deep dive: slot empirics → template cull / convert / weight

You are picking up a template-narrowing initiative for the Aria neural
architecture search project. The goal is to finish the pool-pruning pass:
keep templates whose slot machinery empirically produces dominant mixing,
convert templates whose *exotic* structures show favorable signals into
new first-class templates, and cull the rest. This is a synthesis-affecting
change — measure twice, cut once.

Cold-start safe. No prior conversation context required.

---

## 0. Mandatory pre-work (project gate)

Before ANY Edit/Write call:

```
mcp__code-review-graph__semantic_search_nodes_tool
mcp__code-review-graph__query_graph_tool
mcp__code-review-graph__detect_changes_tool
mcp__code-review-graph__get_review_context_tool
```

Skipping this caused the 2026-04-29 partial-data S1 incident. Use the graph
to find duplicates / callers / dependents before writing any new template
function. Fall back to Grep/Read only when the graph lacks coverage.

Activate venv: `source /home/tim/venvs/llm/bin/activate`.

---

## 1. Source of truth

- DB (read-only): `/home/tim/Projects/LLM/research/lab_notebook.db`
  (open with `file:...?mode=ro&immutable=0`).
- Pass/fail signal: `program_results.controlled_lang_s05_sa_score`
  (S0.5 controlled-language synthetic-association probe, full-DB backfill
  completed 2026-05-03; n≈7,809 non-reference graphs).
  - Pass cohort: `>= 0.95`
  - Fail cohort: `< 0.30`
  - Filter `COALESCE(leaderboard.is_reference, 0) = 0` to exclude pinned refs.
- Companion signal: `controlled_lang_s05_nb_order_acc` (order grammaticality —
  primary discriminating sub-test of nano_blimp).
- Graph features: `program_graph_features` (template_name, slot_usage_json,
  motifs_json, op_count, pair_count). Join via `graph_fingerprint`.
- Full graph topology: `program_results.graph_json` (use this when slot_usage_json
  is too coarse — extract `nodes` / `edges` directly).

## 2. Code surface

| Path | Role |
|---|---|
| `research/synthesis/templates.py:319` | `DEFAULT_TEMPLATE_WEIGHTS` static priors |
| `research/synthesis/templates.py` (TEMPLATES dict ~ line 280) | Template function registry |
| `research/synthesis/_template_role_slot_manifest.py` | Role-slot template registration surface |
| `research/synthesis/_templates_role_slots.py` | Role-slot v1 template implementations |
| `research/synthesis/_templates_role_slot_v2.py` | Role-slot v2 (trunk + retrieval sidecar) |
| `research/synthesis/grammar_support.py:543` `_build_db_template_weights` | Runtime blender (currently uses `s1_rate` only) |
| `research/synthesis/grammar.py` | Grammar config + sampling |
| `research/synthesis/component_catalog.py`, `component_registry.py` | Op/component metadata used during slot fill |

Grammar weights are blended at runtime: `weight = static × (s1_rate/mean)² × (1+novelty)`,
clamped `[0.5, 8.0]`. The blender does NOT currently weigh `controlled_lang_s05_sa`.
Adding that signal to the blender is one of the deliverables.

## 3. Established empirical priors (2026-05-03)

Use these as the starting point — re-verify with your own queries, do not
parrot:

- 96% of fail-cohort graphs contain **no mixer at all**
  (no attention, ssm, or conv op). 77% of pass-cohort graphs contain at least one.
- Reliable pass-rate ops (n≥30): softmax_attention 96%, linear_attention 87%,
  adjacent_token_merge 78%, conv1d_seq 80%, selective_scan 72%,
  rope_rotate 86%, swiglu_mlp 77%, rwkv_time_mixing 80%, rwkv_channel 75%.
- Reliable fail-rate ops: padic_residual (3.5% pass, n=85),
  compression_mixture_experts (3.4%, n=59), embedding_lookup (4.4%, n=159),
  exp_map (8.5%, n=71), basis_expansion (9.3%, n=75), sparse_threshold (11%, n=97).
- **Exotic-but-passing** ops to study (these are the rescue candidates):
  tropical_attention (71% pass, n=52), tropical_matmul (71%, n=38),
  clifford_attention (60%, n=70), associative_memory (61%, n=38),
  graph_attention (66%, n=212), feature_sparsity (70%, n=77),
  depth_gated_transform (71%, n=94), progressive_compression_gate (71%, n=68).
- Cull-candidate templates (32 with n≥20, mean s05_sa<0.55), spanning ~5,620
  rows (~65% of corpus). Highest population: `latent_attn_sparse_ffn` (n=1638),
  `conditional_compute` (n=909), `latent_attn_moe` (n=566), `routed_bottleneck`
  (n=535), `three_way_split` (n=483, 0% pass).
- Templates that fail because slot resolution picks non-mixers (verified by
  sampling 50 graphs of `latent_attn_sparse_ffn` — fail cohort fills with
  ternary_projection / spectral_filter / nm_sparse_linear; pass cohort fills
  the same template with conv1d_seq + swiglu_mlp).

Priors that need **re-confirmation** before acting on them:
- The "exotic-but-passing" ops above may be passing because they co-occur with
  a real mixer, not because the exotic op itself contributes. Phase 1 must
  isolate this.

---

## Phase 1 — Slot-level statistical analysis

Goal: every slot in every active template gets an empirical realization
table. A slot earns the label "dominant mixing" only if, conditioned on the
*op that fills it*, the marginal pass rate is materially better than chance
floor.

### 1.1 Inventory slots

For every template in `TEMPLATES` (`templates.py`) and every role-slot template
(`_template_role_slot_manifest.py`), enumerate:

- Template function name
- Slot identifiers (mixer_slot, ffn_slot, norm_slot, retrieval_slot, etc.)
- Allowed op set per slot (from component_catalog / component_registry / the
  template body)
- Number of ops per slot

Output: `research/reports/slot_inventory.json`. Schema:
```json
{
  "template_name": {
    "slots": [
      {"name": "mixer_slot", "allowed_ops": ["softmax_attention", ...],
       "n_allowed": 12, "is_required": true}
    ]
  }
}
```

### 1.2 Realization frequency (slot → op → outcome)

For each (template_name, slot_name, filling_op), compute over the corpus:
- `n` (sample count, require ≥20 to publish)
- `pass_rate` (sa ≥ 0.95)
- `fail_rate` (sa < 0.30)
- `mean_sa`, `mean_order_acc`
- 95% Wilson CI on pass_rate

Use `program_graph_features.slot_usage_json` if it carries slot→op mapping.
If it does not (verify by sampling 5 rows), fall back to inferring slot fill
from `graph_json` topology — match node ops against the template's allowed-op
set per slot. If you cannot determine slot-fill unambiguously for a template,
flag it as "slot-opaque" and skip ranking it.

Output: `research/reports/slot_realization.parquet` (or csv if pyarrow not
installed) — one row per (template, slot, op).

### 1.3 Marginal mixer credit

To distinguish "the slot itself produces mixing" from "the graph contains a
mixer elsewhere", compute for each slot:

- `pass_rate | slot_op ∈ MIXER_SET, no_other_mixer_in_graph`
- `pass_rate | slot_op ∉ MIXER_SET, has_other_mixer_in_graph`
- `pass_rate | slot_op ∉ MIXER_SET, no_other_mixer_in_graph`

Where `MIXER_SET = {softmax_attention, linear_attention, diff_attention,
graph_attention, tropical_attention, clifford_attention, multiquery_attention,
grouped_query_attention, conv1d_seq, dilated_conv, separable_conv,
selective_scan, mamba_block, gla, hyena_op, rwkv_time_mixing, rwkv_channel,
retention, associative_memory}`.

A slot earns "dominant mixing" iff condition (a) shows pass_rate ≥ 0.60 with
n ≥ 30. A slot fails this test if the only way it passes is via an exogenous
mixer.

Output: `research/reports/slot_mixer_credit.csv`.

### 1.4 Same exercise on ops, marginalized over slots

Sanity check: per-op, condition on `(no_other_mixer, n≥30)`. Use this to
publish a **definitive** mixer / non-mixer / exotic-functional list. Save to
`research/reports/op_mixer_certification.csv`.

---

## Phase 2 — Template classification

Each active template lands in exactly one bucket:

### Bucket A — KEEP (raise weight)
Criteria (all of):
- mean s05_sa ≥ 0.80, n ≥ 20
- Has at least one slot certified "dominant mixing" in Phase 1.3
- Pass rate stable across (s05_sa ≥ 0.95) and (s05_nb_order_acc ≥ 0.90)

### Bucket B — CULL (zero weight, pending removal)
Criteria (any of):
- mean s05_sa < 0.40 with n ≥ 30 AND no slot earns dominant-mixing credit
- 95% Wilson CI upper bound on pass_rate < 0.20

Examples currently expected to land here: `three_way_split`,
`local_attn_routing`, `recursive_moe_attn`, `dual_routing_deep`,
`reduce_attend`, `local_attn_moe`. Verify with your data, do not rubber-stamp.

### Bucket C — RESCUE (reframe slot constraints)
Criteria:
- Same template name appears in *both* pass cohort (n ≥ 30, sa ≥ 0.95) and
  fail cohort (n ≥ 30, sa < 0.30), AND the difference correlates with
  *which op fills a specific slot*.

For each rescue candidate, propose tightened slot constraints — restrict the
slot's allowed_ops to the empirical pass-cohort fill set with n ≥ 5 and
slot-conditioned pass_rate ≥ 0.65. Document the constraint diff in a JSON
report.

Examples expected: `latent_attn_sparse_ffn` (rescue if mixer_slot constrained
to mixers; cull otherwise), `latent_attn_moe`, `routed_bottleneck`.

### Bucket D — MINE (extract sub-pattern as new template)
Criteria:
- A failing parent template contains a topological sub-pattern (≥4 nodes)
  that, when extracted across the pass cohort, has pass_rate ≥ 0.70 with n ≥ 30.
- The sub-pattern uses at least one *exotic* op (tropical, clifford,
  hyperbolic, MoE, sparse) but pairs it with a certified mixer.

Use `program_graph_features.motifs_json` if it carries sub-graph motifs;
otherwise extract motifs by walking edges in `graph_json` and grouping by
(source_op, edge_type, target_op) triples or longer paths. Limit motif size
to ≤8 nodes for tractability.

For each MINE candidate, write a new template function in either
`templates.py` (if it is a basic block) or `_templates_role_slots.py` (if it
uses typed slots). Naming convention: `<exotic>_<mixer>_<sink>`, e.g.
`tropical_attn_swiglu_block`, `clifford_attn_conv_hybrid`.

### Bucket E — INSUFFICIENT DATA
n < 20 or slot-opaque. Leave alone, log to a "needs more samples" file
so the next backfill targets them.

Output: `research/reports/template_classification.csv` with columns
`template_name, bucket, n, mean_sa, primary_reason, action_summary`.

---

## Phase 3 — Implement

### 3.1 New templates (Bucket D)

Add new template functions where they belong. Update `TEMPLATES`,
`DEFAULT_TEMPLATE_WEIGHTS` (start at 3.0), and any role-slot manifest. Each
new template must:

- Use only ops that are registered in `component_registry.py`
- Have a corresponding entry in `pick_template`'s allowed list (auto via
  `TEMPLATES.keys()`)
- Compile end-to-end: `python -m py_compile research/synthesis/templates.py`

Add a smoke test in `research/tests/test_template_smoke.py` (create if absent)
that just instantiates each new template via `pick_template` with a fixed RNG
seed and confirms the resulting graph is `validate_graph()`-clean.

### 3.2 Slot constraint tightening (Bucket C)

For rescue templates, edit the template body so the affected slot draws from
the tightened op set. Do NOT delete ops globally — these are template-local
constraints. Document the diff in the template docstring.

### 3.3 Static weight changes (Bucket B + Bucket A boost)

Edit `DEFAULT_TEMPLATE_WEIGHTS` in `templates.py:319`:

- Bucket B (cull): set weight to **0.5** (the clamp floor, effectively
  near-zero after blending).
- Bucket A (keep): cap any boost at +50% over current; do not exceed clamp
  ceiling 8.0.

Do NOT delete cull-bucket templates outright in this pass — leave them
weighted near-floor so we can resurrect if the blender ever swings them back.

### 3.4 Runtime blender extension (`grammar_support.py:543`)

Extend `_build_db_template_weights` to fold in `controlled_lang_s05_sa_score`:

```python
# Pseudocode — current shape:
weight = static × (s1_rate/mean_s1_rate)² × (1+novelty)

# New shape (additive multiplier, conservative floor at 0.1×):
sa_pass_rate = pass_count / total_count   # for that template, n>=20
sa_factor = max(0.1, sa_pass_rate / 0.40) # 0.40 normalizes mid-cohort
weight = static × (s1_rate/mean_s1_rate)² × (1+novelty) × sa_factor
```

Use a fresh DB query inside the existing `_fetch_template_weight_rows`
helper — do not introduce a second SQL pass. Keep the clamp `[0.5, 8.0]` at
the end. Templates with n<20 use `sa_factor = 1.0` (no penalty, no boost).

Add a unit test in `research/tests/test_grammar_support.py` that constructs a
mock row set and verifies the multiplicative composition.

---

## Acceptance criteria (verify before declaring done)

1. `slot_inventory.json` enumerates every template and every slot.
2. `slot_realization.parquet` covers ≥80% of templates with n≥20; opaque
   templates explicitly listed in a "slot_opaque.txt" exclusion file.
3. Every Bucket-A template has at least one Phase-1.3-certified
   dominant-mixing slot. No exceptions.
4. Every Bucket-B template has Wilson upper-bound CI on pass_rate < 0.20.
5. Every Bucket-D new template:
   - Uses an exotic op + a certified mixer in the same block
   - Compiles
   - Passes the smoke test
6. `python -m pytest research/tests/test_template_smoke.py
   research/tests/test_grammar_support.py -x -q` passes.
7. Pre-existing tests still pass:
   `python -m pytest research/tests/ -m "unit and not slow" -n auto -q`.
   No regressions.
8. The runtime blender's `sa_factor` is observable: log it once at synth-init
   per template name to `research/runtime_events/template_weights.ndjson`.
9. Markdown summary at `research/reports/template_pass_2026XXXX_summary.md`
   listing every classification decision with the empirical justification
   (n, mean, CI bound) — one line per template.

## Constraints

- **Do not** delete templates outright in this pass; weight-floor only.
- **Do not** modify `is_reference`-flagged rows or any pinned reference
  architectures (GPT-2, Mamba, RWKV, RAG).
- **Do not** broaden the corpus during analysis — work only with rows that
  already have `controlled_lang_s05_sa_score`.
- **Do not** introduce new ops; this pass is template/slot-level only.
- Per-template static weight clamp: `[0.5, 8.0]`. Do not bypass.
- Failure cases: if Phase 1.3 produces fewer than 5 certified
  dominant-mixing slots in total, **stop and report** before touching
  weights — the test set is too noisy to act on.

## Coordination

- Read `.current_work.md` before editing any of:
  `research/synthesis/templates.py`, `_templates_role_slots.py`,
  `_templates_role_slot_v2.py`, `grammar_support.py`. Claim each file with a
  timestamped entry; release when done.
- The dashboard (port 5000) does not need restart for these edits to take
  effect on the next synthesis run.
- After landing changes, write a `project_template_pass_<date>.md` memory
  entry summarizing: cull count, rescue count, new-template count, and the
  blender extension landing. Update `MEMORY.md` index.

## Out of scope (do not do in this pass)

- New op kernels
- Scoring weight changes (`leaderboard_scoring.py`)
- Anchor recalibration
- Reference-architecture changes
- Backfilling new probes
- Dashboard UI work

If you find evidence that any of these are needed, write a
`tasks/proposal_<topic>.md` and stop — do not silently expand scope.
