# Findings

## Confirmed
- `AGENTS.md` is absent from this workspace. That is a repo-input gap, not a workflow decision.
- The low-S1 audit supports one real component-code fix: `local_window_attn`.
- Most other low-S1 cases are not broken-component evidence. They are stale evidence, structural misuse, invalid placement, niche placement, screening artifacts, or valid-but-weak behavior.
- Active rule ownership is now wired through [context_rules.py](/home/tim/Projects/LLM/research/synthesis/context_rules.py), [grammar.py](/home/tim/Projects/LLM/research/synthesis/grammar.py), and [validator.py](/home/tim/Projects/LLM/research/synthesis/validator.py).
- Targeted local tests passed for:
  - valid and invalid `local_window_attn` placement
  - structural `identity` misuse
  - fresh generation respecting the active rule layer
  - mutation generation respecting the active rule layer
- `templates.py` no longer contains the duplicate context-policy tables or validator helpers from the earlier audit snapshot.
  Evidence:
  - `rg` over `research/synthesis/templates.py` finds no `CONTEXT_CLASS_*`, `_OP_CONTEXT_CLASS`, `_MOTIF_TEMPLATE_ALLOWLIST`, `apply_context_rule_priors`, or `find_graph_context_violations` definitions
  - template motif gating now imports the canonical helper from `context_rules.py`
- Silent catch-and-drop still exists in active synthesis code paths.
  Evidence:
  - `research/synthesis/grammar.py:582-583` and `research/synthesis/grammar.py:603-604` use `except ValueError: pass`
  - `research/synthesis/validator.py:283-284` uses `except KeyError: pass`
  - `research/synthesis/templates.py:574-575` and `research/synthesis/templates.py:977-978` use `except ValueError: pass`
- No reviewed file showed stub markers such as `TODO`, `NotImplemented`, or `raise NotImplementedError`.

## Still Open
- `local_window_attn` still has a real default-search shared-memory failure mode. This is a config/hardware constraint issue, not a generic predecessor/successor rule issue.
- Spiking, tropical, and hyperbolic niche clusters still need `MATH_SPACE_RULES`-style enforcement in `motifs.py`. The active generic context layer does not replace those domain-specific constraints.
- `n_way_sparse_router` had a real placement-rule gap: validation previously allowed direct residual `add` without an immediate `rmsnorm` / `layernorm` / `linear_proj` successor.
  Evidence:
  - before fix, `rmsnorm -> n_way_sparse_router -> add` passed `_validate_graph(...)`
  - after fix, that graph is rejected with `n_way_sparse_router must feed rmsnorm/layernorm/linear_proj, not stand alone`
- The earlier divisibility/init suspicion for `n_way_sparse_router` did not reproduce in coordinator-local compiler spot checks across non-divisible `(D, n_ways)` combinations.
- Fresh targeted forced reruns for `geometric_product`, `tropical_matmul`, `sign_ste`, `log`, and `sqrt` all compiled and forwarded in valid graphs but still failed rapid learning.
  Interpretation:
  - these are no longer “obviously broken compile path” cases in the tested valid contexts
  - `geometric_product` / `tropical_matmul` remain niche-placement / weak-learning candidates
  - `sign_ste` / `log` / `sqrt` still look numerically weak rather than placement-broken
- Fresh targeted forced rerun for `n_way_sparse_router` reproduced a real bf16/autocast forward bug in `compiler.py`.
  Evidence:
  - `safe_eval(...)` failed with `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`
  - the failing site was `research/synthesis/compiler.py:849` inside `_op_n_way_sparse_router`
  - after casting the scatter source to the accumulator dtype, the same forced graph compiled and forwarded successfully, then failed only at rapid learning
- Fresh targeted forced rerun for `embedding_lookup` passed compile, forward, and rapid screening in a valid forced graph, but still failed S1 micro-train (`loss_ratio=0.9867` over 200 steps).
  Interpretation: `embedding_lookup` is not currently supported by fresh evidence as a broken 0%-pass op; the stale table row needs replacement with fresh evidence.
- Fresh targeted forced rerun for `mod_topk` passed compile and forward in a valid forced graph but still failed rapid learning.
  Interpretation: this currently looks like a weak-learning routing case, not a fresh compile-path bug.
- `norm_last`, `sum_last`, `mean_last`, and `max_last` remain on the structural/reporting path.
  Evidence:
  - they are in `S1_EXEMPT_OPS` at `research/synthesis/context_rules.py`
  - targeted attribution tests already exist in `research/tests/test_structural_s1_exemption.py`
- Broad recorded forced rerun campaign finished with `44/46` coverage, `41` rapid-pass ops, and `1` S1-passing op.
  Evidence:
  - report: `research/reports/exploration_20260321_145219.md`
  - JSON: `research/reports/exploration_20260321_145219.json`
- The broad campaign materially weakens the stale “broken op” interpretation for several rows.
  Evidence from fresh valid-context coverage:
  - `embedding_lookup`: `10/10` inserted, `10` compile, `10` forward, `7` rapid, `0` S1
  - `log`: `13/10` inserted, `12` forward, `11` rapid, `0` S1
  - `sqrt`: `10/10` inserted, `10` forward, `10` rapid, `0` S1
  - `n_way_sparse_router`: `10/10` inserted, `10` compile, `10` forward, `4` rapid, `0` S1 after the dtype fix
  - `tropical_matmul`: `10/10` inserted, `10` compile, `10` forward, `0` rapid, `0` S1
- The broad rerun campaign’s spiking coverage gap was a generation reachability bug, not an execution-path failure.
  Evidence:
  - the motif `spiking_threshold_stdp` previously started with `sparse_threshold`, which violated the accepted predecessor rules
  - after changing the motif to `lif_neuron -> sparse_threshold -> stdp_attention -> linear_proj` and adding a direct forced-graph path, both ops generate successfully
- Fresh targeted spiking reruns now cover both previously missing spiking ops.
  Evidence:
  - `sparse_threshold`: generate pass, compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9851`)
  - `stdp_attention`: generate pass, compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9981`)
- `hyp_distance` was the sole S1-passing op in the recorded campaign (`1/46` targets).
- Fresh post-fix reruns show `n_way_sparse_router` now has stronger evidence than `local_window_attn`.
  Evidence:
  - `n_way_sparse_router`: generate pass, compile pass, forward pass, rapid pass, S1 fail (`loss_ratio=0.9855` in a 300-step targeted rerun)
  - `local_window_attn`: generate pass, compile pass, forward pass, rapid fail (`No learning after 300 steps: init=10.640 final=10.556`)
- Attempting to push learning quality with simpler direct forced builders did not improve results.
  Evidence:
  - `local_window_attn` remained rapid-fail in the direct residual graph
  - `n_way_sparse_router` regressed from rapid-pass to rapid-fail when reduced to the simplest residual graph
- The right next exploration shape is better surrounding context, not smaller graph space.
  Evidence:
  - `research/tools/explore_under_observed.py` now builds a small seed-selected family of valid wrappers for `local_window_attn` and `n_way_sparse_router` instead of a single direct residual block
  - `research/tests/test_under_observed_coverage.py` now proves both ops generate in multiple valid forced-wrapper variants
- The latest CUDA learning-quality probe for the new wrapper families did not yield clean coordinator-grade evidence.
  Evidence:
  - the ad hoc `evaluate_graph(..., device='cuda', run_s1=True)` probe emitted heavy Triton `kernel_fallback` spam for `local_window_attn`
  - that run is not being used as merge or decision evidence; only the builder/test change is accepted at this stage
- Silent exception cleanup is only partially complete.
  Evidence:
  - bare `pass` sites were removed from `grammar.py`, `validator.py`, and two high-signal `templates.py` helpers
  - `templates.py` still contains many explicit exception fallbacks that need case-by-case review before any broader cleanup can be trusted
- Structural ops are still a scoring/attribution problem in per-op S1 reporting until the runner/reporting subsystem is updated.
- The earlier split-brain policy-owner issue between `templates.py` and `context_rules.py` is no longer active.

## Reclassification
- `local_window_attn`: restricted-use / rehab-style handling until default-search evidence is clean.
- Structural-only class: masks, splits/concat family, identity, and reduction-style structural carriers should not be treated as standalone learning carriers.
- Valid-but-weak ops remain valid-but-weak. They should not be turned into forbidden ops without new evidence.

## Rerun-Relevant Clusters
- Polluted-history attention cluster: `graph_attention`, `linear_attention`, `softmax_attention`
- Polluted-history sequence/state cluster: `state_space`
- `local_window_attn` in default-search context after the T1 fix
- Other stale bf16/runtime dominated clusters already marked rerun-needed in the audit artifacts

---

## T2/T3/T5 Implementation (2026-03-21, Volta terminal)

### Changes made
**File: `research/synthesis/motifs.py` (MATH_SPACE_RULES dict)**

Added 7 new entries (was 8, now 15):

| Op | Rule | Rationale |
|---|---|---|
| `tropical_center` | `must_follow: {"tropical_attention", "tropical_gate"}, must_follow_with: {"linear_proj", "linear_proj_down", "tropical_gate"}` | Was only `must_precede: {"rmsnorm", "layernorm"}` — too permissive. Tightened to require tropical predecessor. |
| `tropical_matmul` | `must_precede: {"rmsnorm", "layernorm"}, must_follow_with: {"linear_proj", "linear_proj_down"}` | New entry. Follows existing pattern for tropical_gate/tropical_attention. |
| `lif_neuron` | `must_follow_with: {"spike_rate_code", "sparse_threshold", "stdp_attention"}` | New entry. Enforces spiking successor chain. |
| `sparse_threshold` | `must_follow: {"lif_neuron", "spike_rate_code"}` | New entry. Prevents placement after non-spiking ops. |
| `stdp_attention` | `must_follow: {"sparse_threshold", "spike_rate_code", "lif_neuron"}` | New entry. Prevents placement after non-spiking ops. |
| `hyp_linear` | `must_follow: {"exp_map"}` | New entry. Enforces Poincaré ball bridge. |
| `hyp_tangent_nonlinear` | `must_follow: {"hyp_linear"}, must_follow_with: {"log_map", "linear_proj"}` | New entry. Enforces bridge return path. |

Also updated existing entries:
- `tropical_gate.must_follow_with`: added `"tropical_center"` (valid consumer in tropical chain)
- `tropical_attention.must_follow_with`: added `"tropical_center"` (valid consumer in tropical chain)

**File: `research/tests/test_math_space_rules.py` (new, 16 tests)**
- 7 acceptance tests (valid chains pass `_validate_graph`)
- 9 rejection tests (invalid placements raise ValueError)

### Test evidence
```
$ pytest research/tests/test_math_space_rules.py -q
16 passed in 1.47s

$ pytest research/tests/test_context_rules.py research/tests/test_synthesis_integration.py research/tests/test_ir_roundtrip.py -q
29 passed, 1 warning in 2.43s

$ pytest research/tests/test_routing_ops.py research/tests/test_causal_ops_regression.py -q
34 passed in 2.72s
```

### Files touched
- `research/synthesis/motifs.py` — MATH_SPACE_RULES dict only
- `research/tests/test_math_space_rules.py` — new test file
- No changes to `context_rules.py`, `grammar.py`, `templates.py`, `validator.py`, or scoring code
