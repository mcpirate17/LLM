# Attention Template Failure Audit

Audit scope:
- Notebook corpus: `17150` graph-bearing `program_results` rows, corpus fail rate `77.9%`
- Target families: the 18 templates supplied in the request
- Live code paths checked: template generation, slot constraints, validator, compiler, forward/backward runtime, notebook observability, failure provenance, and backfill/search weighting

## actually broken

No primitive component met the bar for "broken" from live implementation evidence.

The concrete breakage was in template lowering semantics:

| rank | item | kind | raw failures | exposure-normalized failure rate | dominant slot | dominant failure mode | confidence |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | `attn_hyperbolic` | broken template semantics | 50 | 100.0% | `slot1` attention core | `inflight_no_progress` / invalid `hyp_distance -> linear_proj` lowering in live code | high |
| 2 | `attn_bottleneck_hybrid` | broken template semantics + toxic slot usage | 15 | 100.0% | `slot2` compressed sparse path | `insufficient_learning` plus live invalid projection-chain lowering | high |

Exact source files/functions:
- `research/synthesis/_templates_attention_tail.py::tpl_attn_hyperbolic`
- `research/synthesis/_templates_attention.py::tpl_attn_bottleneck_hybrid`
- `research/synthesis/_template_helpers.py::_SLOT_MOTIF_ALLOWLIST`

Concrete evidence:
- `tpl_attn_hyperbolic` was restoring the scalar `hyp_distance` output with `_fix_dim(...)`, which lowers to `linear_proj`; current context rules explicitly forbid `hyp_distance -> linear_proj`.
- `tpl_attn_bottleneck_hybrid` could emit `linear_proj_up -> linear_proj_up` or `linear_proj_up -> linear_proj` on the compressed tail, depending on the sampled motif. Both pairs are rejected by the live validator.

Matched non-failing usages exist for the underlying components, so the blame does not land on the primitives:

| op | exposures | failures | S1 successes | fail rate |
| --- | ---: | ---: | ---: | ---: |
| `hyp_distance` | 106 | 87 | 19 | 82.1% |
| `poincare_add` | 50 | 39 | 11 | 78.0% |
| `graph_attention` | 571 | 446 | 125 | 78.1% |
| `rwkv_channel` | 517 | 410 | 107 | 79.3% |

## not broken but overrepresented

These families are weak in the current screening regime, but the evidence is "underperforming" rather than "broken":

| rank | template | raw failures | exposure-normalized failure rate | dominant template signal | dominant slot | confidence |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1 | `attn_rwkv_hybrid` | 77 | 100.0% | low-loss but never converts; mostly `insufficient_learning` | `slot2` post-FFN norm slot | medium |
| 2 | `attn_gated_product` | 50 | 100.0% | fast-lane probes positive despite no S1 | `slot1` attention core | medium |
| 3 | `attn_hyperbolic` | 50 | 100.0% | historically stalled; live semantic bug also existed | `slot1` attention core | high |
| 4 | `depth_token_mask_block` | 49 | 100.0% | opaque failures, no S1 successes anywhere | `slot1` auxiliary path | medium |
| 5 | `attn_softmax_matmul_sparse_tail` | 28 | 100.0% | low validation loss but no learning conversion | `slot2` sparse tail | medium |
| 6 | `linear_attn_sparse_ffn` | 28 | 100.0% | gets some downstream metrics but no S1 | `slot2` tail norm path | medium |
| 7 | `poincare_add_bridge` | 22 | 100.0% | mostly unlabeled/opaque failures | `slot1` post-bridge motif | medium |
| 8 | `graph_attn_sparse_ffn` | 21 | 100.0% | weak but at least not obviously invalid | `slot1` refine norm path | medium |

Interpretation:
- `attn_linear_no_matmul_ffn_v2`, `attn_softmax_normalized_matmul_v2`, and `attn_softmax_router_sidecar` are already downweighted in `research/synthesis/templates.py` at `0.25`, so they are search-space noise, not a current correctness bug.
- `compact_ffn`, `fixed_tail_norm`, `dense_tail`, and `direct_recovery` variants are sparse-data templates. They should be constrained or backfilled, not deleted.
- `attn_gated_product` is a good example of "weak, not broken": historical S1 is zero, but fast-lane probes were positive, so the right action is guardrails and weight discipline, not removal.

## toxic slot/template interactions

These are placements where otherwise valid motifs/components are structurally misused:

| rank | slot/component interaction | raw failing exposures | normalized failure rate | dominant template | action | confidence |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1 | `attn_bottleneck_hybrid.slot2` + routing/gate motifs (`route_cascade`, `gate_progressive`) | 6 | 100.0% | `attn_bottleneck_hybrid` | constrained to bottleneck-aware sparse motif only | high |
| 2 | `attn_bottleneck_hybrid.slot2` + `bottleneck_sparse` + unconditional width restore | validator-invalid in live code | 100.0% | `attn_bottleneck_hybrid` | fixed residual bridge before width restore | high |
| 3 | `attn_hyperbolic` scalar distance tail + `_fix_dim` | validator-invalid in live code | 100.0% | `attn_hyperbolic` | switched to `linear_proj_up` restoration | high |
| 4 | `depth_token_mask_block.slot1` mixed auxiliary motifs after `depth_token_mask` | 19 observed mixed placements | 100.0% | `depth_token_mask_block` | watchlist only; no live correctness bug proven | medium |

Exact generation paths:
- `attn_bottleneck_hybrid.slot2` is selected inside `research/synthesis/_templates_attention.py::tpl_attn_bottleneck_hybrid`.
- Slot rules are exported from `research/synthesis/_template_helpers.py::get_slot_rule_summary`.
- Historical slot usage comes from graph metadata `template_slot_usage`, attached during template lowering in `research/synthesis/_template_helpers.py`.

Why these are toxic rather than globally broken:
- `route_cascade` and `gate_progressive` are legitimate motifs in routing-style contexts, but they do not belong inside a reduced-rank post-attention bottleneck that is supposed to learn a sparse transform.
- `hyp_distance` is valid elsewhere; the bad behavior came from the template restoring its scalar output with the wrong projection op.

## probable false alarms from baseline op frequency

Raw frequency badly overstates blame for ubiquitous baseline ops. Their failure rates are near the corpus fail rate and they also have thousands of successful S1 exposures.

| op | raw failing exposures | total exposures | fail rate | corpus fail rate | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `add` | 13242 | 17008 | 77.9% | 77.9% | innocent baseline |
| `linear_proj` | 9696 | 12511 | 77.5% | 77.9% | innocent baseline |
| `rmsnorm` | 9400 | 12294 | 76.5% | 77.9% | innocent baseline |
| `layernorm` | 7701 | 9928 | 77.6% | 77.9% | innocent baseline |

High-frequency but still suspicious, because their normalized rates are materially worse than baseline:

| op | raw failing exposures | total exposures | fail rate | dominant template | confidence |
| --- | ---: | ---: | ---: | --- | --- |
| `depth_token_mask` | 50 | 50 | 100.0% | `depth_token_mask_block` | medium |
| `softmax_attention` | 983 | 1051 | 93.5% | spread across many weak attention families | medium |
| `linear_attention` | 892 | 984 | 90.7% | spread across many weak linear-attention families | medium |
| `score_depth_blend` | 91 | 99 | 91.9% | `depth_token_mask_block` | medium |

The first table is the key anti-scapegoating result: `add`, `linear_proj`, `rmsnorm`, and `layernorm` are common because they are everywhere, not because they are broken.

## fixes implemented

### correctness fixes

1. `attn_hyperbolic` now restores the `hyp_distance` scalar with `linear_proj_up` instead of `_fix_dim(...)`.
   - file: `research/synthesis/_templates_attention_tail.py`
   - function: `tpl_attn_hyperbolic`
   - effect: removes the live-invalid `hyp_distance -> linear_proj` edge

2. `attn_bottleneck_hybrid` now uses bottleneck-safe motif classes and inserts a normalization bridge before restoring width when the sampled motif ends in `linear_proj_up`.
   - file: `research/synthesis/_templates_attention.py`
   - function: `tpl_attn_bottleneck_hybrid`
   - effect: removes both forbidden projection chains while preserving the bottleneck structure

### compatibility constraints

3. Added an explicit allowlist for `attn_bottleneck_hybrid.slot2` so the compressed sparse slot only samples `bottleneck_sparse`.
   - file: `research/synthesis/_template_helpers.py`
   - symbol: `_SLOT_MOTIF_ALLOWLIST`
   - effect: blocks routing/gating motifs from a slot where they were structurally toxic

### tests added/updated

4. Extended seed-slice validation coverage to the repaired templates.
   - file: `research/tests/test_slot_template_wiring.py`
   - test: `test_audited_templates_validate_across_seed_slice`

5. Extended slot-rule export coverage for the new bottleneck slot constraint.
   - file: `research/tests/test_slot_template_wiring.py`
   - test: `test_slot_rule_summary_exports_current_template_constraints`

Verification run:
- `pytest -q research/tests/test_slot_template_wiring.py -k "attn_bottleneck_hybrid or attn_hyperbolic or slot_rule_summary_exports_current_template_constraints"` -> `3 passed`
- 25-seed validator sweep for `attn_bottleneck_hybrid` and `attn_hyperbolic` -> `0` validation failures each
- live compile/forward/backward check for both repaired templates -> pass

## ranked tables

### raw error counts by target template

| rank | template | raw failures | exposures | fail rate | best validation loss | dominant error label | confidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `attn_rwkv_hybrid` | 77 | 77 | 100.0% | 0.617 | `insufficient_learning` | medium |
| 2 | `attn_gated_product` | 50 | 50 | 100.0% | 0.630 | `insufficient_learning` / `inflight_no_progress` | medium |
| 3 | `attn_hyperbolic` | 50 | 50 | 100.0% | 0.780 | `inflight_no_progress` | high |
| 4 | `depth_token_mask_block` | 49 | 49 | 100.0% | n/a | `unknown` | medium |
| 5 | `attn_softmax_matmul_sparse_tail` | 28 | 28 | 100.0% | 0.685 | `insufficient_learning` | medium |
| 6 | `linear_attn_sparse_ffn` | 28 | 28 | 100.0% | 0.798 | `insufficient_learning` | medium |
| 7 | `poincare_add_bridge` | 22 | 22 | 100.0% | n/a | `unknown` | medium |
| 8 | `graph_attn_sparse_ffn` | 21 | 21 | 100.0% | 0.745 | `insufficient_learning` | medium |

### exposure-normalized op rates for high-signal suspects

| rank | op | raw failures | exposures | fail rate | dominant templates | confidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `depth_token_mask` | 50 | 50 | 100.0% | `depth_token_mask_block` | medium |
| 2 | `softmax_attention` | 983 | 1051 | 93.5% | many weak attention templates | medium |
| 3 | `score_depth_blend` | 91 | 99 | 91.9% | `depth_token_mask_block` | medium |
| 4 | `linear_attention` | 892 | 984 | 90.7% | many weak linear-attention templates | medium |
| 5 | `hyp_distance` | 87 | 106 | 82.1% | `attn_hyperbolic`, `hyp_distance_scoring` | medium |

### dominant slots for repaired or suspicious templates

| template | dominant slot | dominant slot components seen historically | action |
| --- | --- | --- | --- |
| `attn_bottleneck_hybrid` | `slot2` | `bottleneck_sparse`, `route_cascade`, `gate_progressive`, `routed_ternary` | constrained to `bottleneck_sparse` only |
| `attn_hyperbolic` | `slot1` | `attn_latent_compress`, `attn_diff`, `attn_linear`, `attn_softmax` | no slot block added; fixed tail semantics instead |
| `depth_token_mask_block` | `slot1` | mixed auxiliary motifs after `depth_token_mask` | watchlist; needs better failure labeling before hard constraints |
| `poincare_add_bridge` | `slot1` | mixed post-bridge motifs | watchlist; no live correctness bug proven |

## summary judgment

- Real breakage was template-local, not primitive-global.
- Baseline ops are overrepresented, not broken.
- The strongest fix was to repair invalid lowering and constrain one toxic compressed slot.
- `depth_token_mask_block` and `poincare_add_bridge` remain the highest-value follow-up audits because the current notebook labels are still too opaque to prove a correctness bug.
