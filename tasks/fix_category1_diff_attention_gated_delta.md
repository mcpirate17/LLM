# Category 1: Pipeline/Template Gaps — diff_attention, gated_delta, fused_linear_gelu

## Problem

Three ops with learnable parameters (D*D to D*D*4) are at 0% S1 not because the ops are broken, but because they lack proper template routing or have evaluation pipeline issues preventing them from being trained.

## Evidence from the database

### diff_attention (Microsoft Differential Attention, ICLR 2025)
- **48 programs generated, 47 passed S0, 0 have loss_ratio** — every single program dies in the screening/S1 pipeline before full training runs
- **No template routing exists** — there is no `_OP_TO_TEMPLATE` entry in `grammar.py` for diff_attention
- **No MATH_SPACE_RULES** — freeform placement only
- The op itself is well-implemented (`compiler.py:421-454`): dual softmax attention maps subtracted with learned lambda, output projected via o_proj
- Regular `softmax_attention` has hundreds of S1 passes, and diff_attention is strictly more expressive
- Has context rule: `SearchMode.GENERAL`, forbidden_pred=reduce_ops, forbidden_succ=split_ops, requires_residual_context=True

### gated_delta (NVIDIA GatedDeltaNet, ICLR 2025)
- **Only 2 programs reached loss_ratio** out of 49 attempts — under-sampled
- Has D*D*4 parameters (5 projections + output projection) — plenty of capacity
- Template `recurrent_delta_block` produces: `layernorm → gated_delta → linear_proj → linear_proj_down → relu → linear_proj_up → add`
- Best loss_ratio is 0.968 — learns slightly but not enough for S1 (threshold 0.85)
- The op was recently optimized (multi-head chunked scan, 4.8x speedup) so it may now train faster and converge better

### fused_linear_gelu
- **7 programs trained, all with loss_ratio 0.986-0.997** — barely learns
- Regular `gelu` appears in 455 programs with 219 S1 passes — so linear+gelu clearly works
- Template `fused_gelu_ffn` produces: `layernorm → fused_linear_gelu → linear_proj_down → add`
- The issue: `fused_linear_gelu` projects D→4D (up-projection with GELU activation), then `linear_proj_down` projects 4D→D. This is a standard SwiGLU/FFN pattern but the template is missing the **gate branch** that makes SwiGLU work vs plain FFN
- Compare with `swiglu_mlp` which does: `silu(gate_proj(x)) * up_proj(x)` → down_proj — the gating is what makes it learn

## Tasks

### 1. Create diff_attention template and routing

File: `research/synthesis/templates.py`

Create `tpl_diff_attention_block` following the pattern of other attention templates:
```
norm → diff_attention → linear_proj → [optional FFN motif] → residual_add
```

The `diff_attention` op handles Q/K/V projection and the dual-softmax internally (it has q_proj, k_proj, v_proj, o_proj, lambda_param). The template just needs to provide it with normalized input and a residual connection.

Look at `tpl_local_attention_block` or `tpl_graph_attention_block` as models — they follow the same pattern for attention ops that have internal projections.

File: `research/synthesis/grammar.py`

Add to `_OP_TO_TEMPLATE`:
```python
"diff_attention": "diff_attention_block",
```

File: `research/synthesis/templates.py`

Register in `TEMPLATES` dict and `DEFAULT_TEMPLATE_WEIGHTS` (weight 3.0 — it's a proven attention mechanism).

### 2. Fix fused_linear_gelu template

File: `research/synthesis/templates.py`

The current `tpl_fused_gelu_ffn` template is:
```
layernorm → fused_linear_gelu → linear_proj_down → add
```

This is a plain FFN (up-project + activation + down-project). The problem is there's no **gating branch**. Compare with the proven swiglu pattern: `silu(gate(x)) * up(x) → down`.

Fix by adding a sigmoid gate branch:
```
norm → fused_linear_gelu(x) → sigmoid(linear_proj(x)) → mul → linear_proj_down → add
```

Or simpler: just add a second linear_proj as a gate:
```
norm → fused_linear_gelu → mul(fused_out, sigmoid(linear_proj(normed))) → linear_proj_down → add
```

This gives the network a way to selectively suppress features, which is what makes gated FFNs work.

### 3. Investigate gated_delta under-sampling

The op only had 2 training attempts reach loss_ratio. Run:
```bash
python -m research.tools.explore_under_observed \
  --mode=forced --graphs-per-op=20 --device=cuda --record \
  --ops gated_delta
```

If it still doesn't pass S1 after 20 attempts with the new multi-head scan optimization, the issue is likely that the `recurrent_delta_block` template needs a richer surrounding context (e.g., an FFN after the delta recurrence, or a different norm placement).

## Verification

After changes:
```bash
# Compile check
python -c "from research.synthesis.templates import TEMPLATES; print(len(TEMPLATES))"

# Placement check
python -c "
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
for op in ['diff_attention', 'fused_linear_gelu', 'gated_delta']:
    cfg = GrammarConfig.exploration(target_ops=frozenset({op}), model_dim=64, boost_factor=50.0)
    ok = sum(1 for s in range(100) if _try(cfg, s, op))
    print(f'{op}: {ok}/100 placement')
"

# Test suite
python -m pytest research/tests/test_synthesis_integration.py research/tests/test_slot_template_wiring.py research/tests/test_context_rules.py -x --tb=short -q
```
