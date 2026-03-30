---
status: active
created: 2026-03-29
author: claude-opus
---

# Task: Fix Context Rules for 4 High-Value Ops

## What You're Fixing

Four ops produce the **lowest loss in the entire system** when used correctly, but have <5% S1 pass rates because the grammar places them without proper scaffolding. You need to add/tighten context rules so these ops only appear in contexts where they can succeed.

**IMPORTANT**: There is a pre-existing import error. `_templates_core.py` imports `MOTIF_CLASS_NORM` from `_template_helpers.py` but it doesn't exist there (probably renamed). Fix this first or the synthesis module won't import.

```bash
# Find it
grep -n "MOTIF_CLASS_NORM" research/synthesis/_template_helpers.py research/synthesis/_templates_core.py
```

## File to edit

`research/synthesis/context_rules.py` — all rules live in the `CONTEXT_RULES` dict (line ~210+).

## Op 1: `token_merge` — Currently has NO context rules

**Problem**: 89% S0 failure. 660/719 failures are `Strict Causality Gate Failed: Model looks ahead at future tokens`. Token_merge reorders tokens which breaks causality for downstream causal ops.

**Evidence from data**:
- `token_merge_block` template: **75% S1 success**, mean loss 0.342
- `sparse_ffn` template: **100% S1 success** (6/6), mean loss 0.273
- `sequential` template: **8% S1 success** — random placement kills it
- Succeeds WITH: `rmsnorm` (59%), `nm_sparse_linear` (69%)
- Fails WITH: `linear_proj` (7%), `selective_scan` (13%)

**What to add** — new entry in CONTEXT_RULES:
```python
"token_merge": ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS | frozenset({
        "split2", "split3",  # dim mismatch after split
    }),
    forbidden_successors=_CAUSAL_SENSITIVE_OPS | frozenset({
        "output_head",
        "softmax_attention",  # attention needs original token order
        "linear_attention",
        "selective_scan",  # SSM needs causal token ordering
        "state_space",
        "conv1d_seq",  # conv assumes original sequence order
    }),
    requires_residual_context=True,  # merged output needs skip connection to preserve info
),
```

The key insight: token_merge destroys token ordering, so anything downstream that depends on sequential/causal structure will fail. It must be inside a residual block.

## Op 2: `selective_scan` — Needs predecessor preference

**Problem**: 54% S0 failure (causality gate), 90% S1 failure when S0 passes (insufficient learning).

**Evidence from data**:
- **Every success with loss < 0.05** has the pattern: `norm → conv1d_seq → silu → selective_scan → add`
- `recursive_depth_router`: 19% S1 success
- `transformer_block`: 41% S1 success
- `parallel_split`: 0% S1 (100% fail) — splitting dims breaks SSM
- `depth_gated_block`: 0% S1 (100% fail)

`selective_scan` already appears in `_CAUSAL_SENSITIVE_OPS` and `_FULL_DIM_OPS`, but has **no dedicated ContextRule entry**. It needs one:

```python
"selective_scan": ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS | _STRUCTURAL_SPLIT_OPS | frozenset({
        "identity",  # strips causal context
        "token_merge",  # destroys token ordering SSM depends on
        "transpose_sd",  # transposes (S,D) axes, breaks SSM state shape
    }),
    forbidden_successors=frozenset({
        "output_head",
        "selective_scan",  # SSM→SSM chaining 96% fail
        "state_space",  # same failure mode
    }),
    requires_residual_context=True,  # SSM output needs residual path
),
```

## Op 3: `softmax_attention` — Tighten existing rule

**Problem**: 95% S0 pass (code is fine), but 97% S1 failure. 830/850 failures are `insufficient_learning`. Attention needs QKV projections + causal masking + positional encoding to function — but the grammar drops it into random positions.

**Current rule** (line ~538):
```python
"softmax_attention": ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=frozenset({"causal_mask"}),
    forbidden_successors=frozenset({"output_head", "linear_proj"}),
    requires_residual_context=True,
),
```

**Evidence from data**:
- `signal_routed_compression`: **89% S1 success** (8/9)
- `rope_attention_block`: **7% S1 success** (10/137) — even the dedicated template struggles
- `parallel_split`: **0% S1** (60 failures) — halved dim breaks multi-head structure
- `mixed_recursion`: **0% S1** (29 failures)
- `conditional_compute`: **0% S1** (28 failures)
- `hybrid_parallel`: **0% S1** (15 failures)

**Tighten to**:
```python
"softmax_attention": ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=frozenset({
        "causal_mask",  # 100% fail (58/58) — mask tensor fed as data input
        "split2", "split3",  # halved dim breaks multi-head structure
        "linear_proj_down",  # reduced dim breaks head dimension
        "token_merge",  # destroyed token order breaks attention
        "transpose_sd",  # wrong axis orientation
    }) | _REDUCE_OPS,
    forbidden_successors=frozenset({
        "output_head",
        "linear_proj",  # 100% fail (13/13) — raw attention output needs norm first
        "identity",  # strips causal context
        "softmax_attention",  # stacking raw attention 100% fail
    }),
    requires_residual_context=True,
),
```

## Op 4: `transpose_sd` — Add context rule (currently none)

**Problem**: 95% S1 failure. Only works inside dedicated cross-dimension templates (9% in `cross_dim_mixer`, 5% in `dual_axis_block`). Transposing the (Sequence, Dimension) axes is a very specialized operation that breaks most downstream ops expecting (B, S, D) layout.

**Evidence**: `residual_block`: 0% (0/27). `recursive_depth_router`: 0% (0/7). Every template that doesn't explicitly handle the transposed layout fails.

```python
"transpose_sd": ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS | frozenset({
        "transpose_sd",  # 100% fail (10/10) — double transpose = noop or wrong
    }),
    forbidden_successors=_CAUSAL_SENSITIVE_OPS | _FULL_DIM_OPS | frozenset({
        "output_head",
        "rmsnorm", "layernorm",  # norms along wrong axis after transpose
        "concat",  # dim mismatch
        "split2", "split3",  # splitting transposed tensor
    }),
    requires_residual_context=True,  # transposed output MUST rejoin through residual
),
```

## Verification

After making changes, run:

```bash
source /home/tim/venvs/llm/bin/activate

# Compile check
python -m py_compile research/synthesis/context_rules.py

# Run context rule tests
python -m pytest research/tests/test_low_s1_context_rules.py -x -q

# Verify the rules load and count
python -c "
from research.synthesis.context_rules import CONTEXT_RULES
for op in ['token_merge', 'selective_scan', 'softmax_attention', 'transpose_sd']:
    rule = CONTEXT_RULES.get(op)
    if rule:
        n_fp = len(rule.forbidden_predecessors)
        n_fs = len(rule.forbidden_successors)
        print(f'{op}: {n_fp} forbidden preds, {n_fs} forbidden succs, residual={rule.requires_residual_context}')
    else:
        print(f'{op}: MISSING — add it!')
"
```

## What NOT to do

- Do NOT penalize or lower op weights — we're fixing placement, not banning ops
- Do NOT modify the Bayesian tracker or ML models — the backfill after your fix will retrain them
- Do NOT modify `grammar.py` or `templates.py` — only `context_rules.py`
- Do NOT add overly broad bans — each forbidden pair should have data backing (noted in comments)
- Keep the existing `CONTEXT_RULES` entry for `softmax_attention` and update it in place
