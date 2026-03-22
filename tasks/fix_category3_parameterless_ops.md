# Category 3: Parameter-Free Ops Need Richer Templates — minimum, sub, geometric_product, tropical_matmul, cumprod_safe, early_exit, cascade

## Problem

Seven ops are at 0% S1 because their templates don't provide enough learnable capacity. These ops have **zero or minimal learnable parameters** — they're pure mathematical transforms (element-wise min, subtraction, Clifford product, tropical matmul, cumulative product). Their templates surround them with 1-2 linear projections, which isn't enough to compensate.

The fix is to embed each op inside a **richer architectural context** with sufficient downstream capacity — following the pattern that worked for `tropical_center` (lr=0.079 when placed inside a full tropical stack) and `spike_rate_code` (lr=0.007 when paired with tropical_moe).

## Evidence from the database

| Op | Params | N trained | Best LR | Current template | Template structure |
|---|---|---|---|---|---|
| minimum | 0 | 14 | 0.963 | gated_minimum | norm → proj_a → proj_b → minimum(a,b) → proj → add |
| sub | 0 | 2 | 0.986 | residual_difference | norm → proj_a → proj_b → sub(a,b) → proj → add |
| geometric_product | 0 | 8 | 0.968 | geometric_product_block | norm → rotor_a → rotor_b → geom_product(a,b) → grade_select → proj → add |
| tropical_matmul | 0 | 5 | 0.961 | tropical_matmul_block | norm → proj_a → proj_b → tropical_matmul(a,b) → proj → add |
| cumprod_safe | 0 | 2 | 0.993 | decay_sequence | norm → sigmoid → cumprod_safe → proj → add |
| early_exit | D | 2 | 0.979 | cascaded_early_exit | norm → linear_attention → proj → early_exit → add → norm → sigmoid → log → proj → cascade → add |
| cascade | D | 2 | 0.654 | cascaded_early_exit | (same template as early_exit) |

All of these follow the same pattern: norm → [1-2 projections] → param-free op → [1 projection] → residual. The surrounding projections have D*D parameters each, giving ~2-3*D*D total capacity. But the param-free ops themselves don't contribute to gradient-based learning — they just transform the signal.

## Root cause analysis

The param-free binary ops (minimum, sub, tropical_matmul) compute `f(proj_a(x), proj_b(x))` where `f` is a fixed function. The two projections CAN learn, but:

1. **minimum(a, b)** kills gradients on the non-minimum element at each position — half the gradient signal is zeroed
2. **sub(a, b)** works if the projections learn to produce complementary features, but the template provides no incentive for this
3. **tropical_matmul(a, b)** is min-plus matrix multiply — sparse gradients through the min operation
4. **geometric_product** operates in Clifford algebra space — wrong inductive bias for token sequences
5. **cumprod_safe** multiplies along the sequence — exponential decay/growth even with sigmoid bounding

## Tasks

### 1. Add FFN capacity after binary ops

The core issue is that a single `linear_proj` after the param-free op can't compensate. Add a full FFN (norm → linear_proj_up → gelu → linear_proj_down) after the op to give the network capacity to interpret the transform's output.

File: `research/synthesis/templates.py`

For each of: `tpl_gated_minimum`, `tpl_residual_difference`, `tpl_tropical_matmul`:

Change the pattern from:
```
norm → proj_a → proj_b → OP(a,b) → linear_proj → add
```
to:
```
norm → proj_a → proj_b → OP(a,b) → linear_proj → [FFN motif] → add
```

Use `_pick_compatible_motif_from_classes(graph, projected, rng, list(_FFN_CLASSES), weights)` to add an FFN motif after the projection. Look at `tpl_state_space_block` or `tpl_integral_kernel_block` for examples of how to do this — they both add an FFN motif after their main op.

The relevant motif classes for FFN are defined near the top of templates.py:
```python
_FFN_CLASSES = (MOTIF_CLASS_FFN, MOTIF_CLASS_GATED_FFN)
```

Then use `_fix_dim(graph, processed)` to ensure the output matches `model_dim` before the residual add.

### 2. Fix cumprod_safe template

The current `decay_sequence` template does:
```
norm → sigmoid → cumprod_safe → proj → add
```

The sigmoid bounds values to (0,1), and cumulative product of values in (0,1) gives exponentially decaying signal. After a few positions, the signal is essentially zero.

Fix: Use `cumprod_safe` as a **decay weight** for an attention-like mechanism, not as the primary signal:
```
norm → proj_value → cumprod_safe(sigmoid(proj_decay(normed))) → mul(value, decay) → proj → add
```

This applies exponential decay weighting to projected values — similar to how RWKV uses exponential decay for time mixing.

### 3. Fix early_exit / cascade templates

`cascade` has 2 S1 passes (lr=0.654) — marginal but real. The `cascaded_early_exit` template is overly complex (12 ops including log, sigmoid, entropy). Simplify to the proven cascade pattern:

Check what ops co-occur with cascade in its 2 S1-passing programs and build a template matching that pattern. The current template tries to do too much — routing, scoring, exit detection, cascade all in one block.

### 4. Consider structural exemption for truly-structural ops

Some of these ops (minimum, sub) are fundamentally **structural transforms** that will never carry learning on their own. They're analogous to `concat`, `split2`, `identity` which are already in `S1_EXEMPT_OPS`.

File: `research/synthesis/context_rules.py`

Consider adding to `S1_EXEMPT_OPS`:
```python
S1_EXEMPT_OPS: FrozenSet[str] = frozenset({
    "identity", "split2", "split3", "concat",
    "causal_mask", "sliding_window_mask",
    "norm_last", "sum_last", "mean_last", "max_last",
    # Parameter-free elementwise ops that can't carry learning standalone
    "minimum", "maximum", "sub",
    # Parameter-free sequence ops
    "cumprod_safe", "cumsum",
})
```

This won't fix the ops, but it will stop them from being blamed in the Component Health Grid for low S1 rates. They'll show as "structural" instead of "degraded/broken", which is accurate.

Only exempt ops that truly have zero learnable parameters and are pure mathematical transforms. Do NOT exempt `geometric_product` (it's in a math space with specific algebraic semantics) or `early_exit`/`cascade` (they have learnable parameters).

## Verification

```bash
# Test each updated template produces valid graphs
python -c "
import random, torch
from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_model
from research.synthesis.templates import (
    tpl_gated_minimum, tpl_residual_difference, tpl_tropical_matmul
)

for name, fn in [('gated_minimum', tpl_gated_minimum),
                 ('residual_difference', tpl_residual_difference),
                 ('tropical_matmul', tpl_tropical_matmul)]:
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    out = fn(g, inp, random.Random(42))
    g.set_output(out)
    ops = [n.op_name for n in g.nodes.values() if not n.is_input]
    model = compile_model([g], vocab_size=100, max_seq_len=64)
    x = torch.randint(0, 100, (2, 32))
    y = model(x)
    y.sum().backward()
    print(f'{name}: ops={ops} shape={y.shape} OK')
"

# Test suite
python -m pytest research/tests/test_synthesis_integration.py research/tests/test_slot_template_wiring.py research/tests/test_context_rules.py -x --tb=short -q

# Run exploration for the most promising ops
python -m research.tools.explore_under_observed \
  --mode=forced --graphs-per-op=10 --device=cuda --record \
  --ops minimum,sub,tropical_matmul
```

## Priority

1. **High**: Add FFN capacity to minimum/sub/tropical_matmul templates — this is the same fix pattern as adding tropical_attention to tropical_center, and it's mechanical
2. **Medium**: Fix cumprod_safe template to use decay-weighting pattern
3. **Medium**: Add structural exemptions for pure-transform ops
4. **Low**: Simplify cascaded_early_exit — cascade already has 2 marginal passes, not worth major refactoring
