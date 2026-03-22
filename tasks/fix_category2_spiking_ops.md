# Category 2: Wrong Template Composition — Spiking Ops (lif_neuron, sparse_threshold, stdp_attention)

## Problem

The spiking neural network ops (`lif_neuron`, `sparse_threshold`, `stdp_attention`) are all at 0% S1, but the closely related `spike_rate_code` op has **2 S1 passes with loss_ratio 0.007** (excellent). The winning architecture for `spike_rate_code` does NOT use the `spiking_residual_block` template — it uses a completely different composition that pairs with `tropical_moe`.

This is the same class of bug as `tropical_center`: the template doesn't match the proven winning architecture from the database.

## Evidence from the database

### spike_rate_code winners (the only spiking ops that pass S1)
```
id=0 op=input
id=6 op=spike_rate_code inputs=[0]      ← direct from input, no norm
id=11 op=tropical_moe inputs=[6]        ← tropical MoE processes the spike-coded signal
id=12 op=add inputs=[0, 11]             ← residual
loss_ratio=0.0070
```

This is a radically different architecture from `spiking_residual_block`. The winning pattern:
1. Skip normalization (spike_rate_code normalizes internally via firing rate encoding)
2. Feed spike-coded signal directly to `tropical_moe` (min-plus routing with learned experts)
3. Residual connection

### Current spiking_residual_block template (what the spiking ops actually use)
```
norm → lif_neuron → spike_rate_code → [optional sparse_threshold → stdp_attention] → linear_proj → residual
```

This template chains 3-4 spiking ops sequentially, compounding gradient sparsity. Each spike operation produces binary/near-binary signals, and stacking them destroys gradient flow.

### Op-level stats
| Op | Total | S1 passes | Best LR | Template |
|---|---|---|---|---|
| spike_rate_code | 26 | 2 | 0.007 | spiking_residual_block (BUT winners don't use it) |
| lif_neuron | 45 | 0 | 0.964 | spiking_residual_block |
| sparse_threshold | 31 | 0 | 0.979 | spiking_residual_block |
| stdp_attention | 31 | 0 | 0.979 | spiking_residual_block |

### Why the current template fails
- **Gradient sparsity compounding**: lif_neuron produces binary spikes → spike_rate_code averages them → sparse_threshold re-binarizes → stdp_attention applies timing-based plasticity. Each stage discards more gradient signal.
- **No capacity for learning**: The spiking ops have 0-1 learnable parameters. The template's only learnable capacity is the linear_proj at the end, which can't compensate.
- **No routing/mixing**: The winning `spike_rate_code + tropical_moe` pattern works because `tropical_moe` has D*D*4 parameters spread across multiple experts, providing massive downstream capacity to interpret the spike-coded signal.

## Tasks

### 1. Create a `spiking_moe_block` template that matches the winning pattern

File: `research/synthesis/templates.py`

The winning architecture is: `[spiking_op] → tropical_moe → residual_add`

But `tropical_moe` is dispatched via `execute_fn` (mathspace registry), not `_OP_DISPATCH`. Use `tropical_gate` as the routing component instead (it IS in dispatch and has D*D params):

```python
def tpl_spiking_moe_block(graph, input_id, rng, weights=None):
    """[spiking_op] → tropical_gate → linear_proj → residual.

    Proven pattern: spiking encoding → tropical routing → projection.
    spike_rate_code + tropical_moe achieved lr=0.007 (best in project).
    Spiking ops normalize internally, so no norm predecessor needed.
    """
    D = graph.model_dim
    # Pick a spiking op (rotate between lif_neuron, spike_rate_code, sparse_threshold)
    spiking_ops = ["lif_neuron", "spike_rate_code"]
    spike_op = rng.choice(spiking_ops)

    try:
        spiked = graph.add_op(spike_op, [input_id])
        # Tropical routing provides the learnable capacity
        gated = graph.add_op("tropical_gate", [spiked])
        out = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out
```

**Important**: Check the MATH_SPACE_RULES for `tropical_gate` — it has `must_precede: {rmsnorm, layernorm, tropical_attention}`. You'll need to add the spiking ops to this set, since the winning architecture skips normalization:

File: `research/synthesis/motifs.py`

In `MATH_SPACE_RULES["tropical_gate"]["must_precede"]`, add `"lif_neuron"`, `"spike_rate_code"`, `"sparse_threshold"`.

### 2. Update grammar routing for spiking ops

File: `research/synthesis/grammar.py`

Change the `_OP_TO_TEMPLATE` entries:
```python
"lif_neuron": "spiking_moe_block",        # was: spiking_residual_block
"sparse_threshold": "spiking_moe_block",   # was: spiking_residual_block
"spike_rate_code": "spiking_moe_block",    # was: spiking_residual_block
```

Keep `stdp_attention` on `spiking_residual_block` for now — it requires specific predecessor context (sparse_threshold or spike_rate_code per context_rules.py line 634-636).

Or better: create a second template `spiking_stdp_block` that chains: `lif_neuron → sparse_threshold → stdp_attention → linear_proj → residual` (the minimal viable STDP chain).

### 3. Update context rules

File: `research/synthesis/context_rules.py`

The `lif_neuron` graph validation (line 628-630) currently requires:
```python
if not _has_descendant_op(graph, nid, {"spike_rate_code", "stdp_attention"}, children):
    violations.append("lif_neuron requires spiking successor context")
```

Add `tropical_gate` as a valid successor for the new template pattern:
```python
if not _has_descendant_op(graph, nid, {"spike_rate_code", "stdp_attention", "tropical_gate"}, children):
```

### 4. Register template and weights

File: `research/synthesis/templates.py`

Add to `TEMPLATES` dict and `DEFAULT_TEMPLATE_WEIGHTS` (weight 3.0 — the winning pattern has lr=0.007).

### 5. Update motif template allowlist

File: `research/synthesis/context_rules.py`

Add `"spiking_moe_block"` to any spiking motif allowlists in `_MOTIF_TEMPLATE_ALLOWLIST`.

## Verification

```bash
# Placement
python -c "
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
for op in ['lif_neuron', 'sparse_threshold', 'spike_rate_code']:
    cfg = GrammarConfig.exploration(target_ops=frozenset({op}), model_dim=64, boost_factor=50.0)
    success = sum(1 for s in range(100)
                  if (g := _try_gen(cfg, s)) and op in {n.op_name for n in g.nodes.values()})
    print(f'{op}: {success}/100')
"

# Compile + train test
python -c "
from research.synthesis.templates import tpl_spiking_moe_block
from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_model
import torch, random
g = ComputationGraph(model_dim=64)
inp = g.add_input()
out = tpl_spiking_moe_block(g, inp, random.Random(42))
g.set_output(out)
model = compile_model([g, g], vocab_size=256, max_seq_len=128)
x = torch.randint(0, 256, (4, 64))
out = model(x)
out.sum().backward()
print(f'OK: {out.shape}')
"

# Test suite
python -m pytest research/tests/test_synthesis_integration.py research/tests/test_context_rules.py research/tests/test_math_space_rules.py -x --tb=short -q

# Run exploration
python -m research.tools.explore_under_observed --mode=forced --graphs-per-op=20 --device=cuda --record --ops lif_neuron,sparse_threshold
```
