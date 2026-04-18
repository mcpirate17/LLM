# Template Optimization Playbook

**Created**: 2026-04-15
**Context**: Systematic template optimization to beat GPT-2/Mamba 5x
**Companion files**:
- `TEMPLATE_AUDIT.md` — Full catalog of ~163 templates categorized by strength
- `tools/eval_templates.py` — Extended evaluation tool (train + binding probes)
- `research.scientist.notebook.notebook_core.record_program_result()` — Store eval results in LabNotebook DB
- `tests/test_template_optimization.py` — 34-test validation suite

## 1. The Winning Architecture Pattern

Every template that achieves >30% S1 rate follows this structure:

```
norm -> {attention_path || complementary_path} -> merge -> residual -> norm -> FFN -> residual
```

**Critical requirements:**
1. **Parallel mixing**: Attention + SSM (or other) in PARALLEL, not sequential
2. **Full-width FFN**: `swiglu_mlp` with mlp_ratio 2-3, or motif-picked FFN
3. **3 residuals**: merge(path_a, path_b), skip(input, merged), skip(mid, ffn_output)
4. **Attention op**: Must include one of: `softmax_attention`, `latent_attention_compressor`, `graph_attention`, `local_window_attn`, `linear_attention`, `diff_attention`

**What kills learning:**
- Missing FFN (no information expansion)
- Sparse tail with low density (75%+ dead gradients)
- Sequential dual-attention (bottleneck, not complementary)
- Reciprocal/log as sole nonlinearity (gradient instability)
- Over-complex routing (calibrated_branch_merge, 5+ residuals)

## 2. Top Operations by Empirical S1 Rate

| Op | S1 Rate | Induction | Role |
|----|---------|-----------|------|
| latent_attention_compressor | 28.5% | 0.004 | Best attention op |
| state_space | 28.7% | — | Best SSM complement |
| padic_expand | 34.1% | 0.004 | Hierarchical features |
| adaptive_recursion | 41.3% | — | Depth-adaptive processing |
| softmax_attention | — | 0.066 max | Best induction signal |
| cumsum | — | 0.024/0.110 | Proto-attention accumulation |
| matmul | — | 0.007/0.062 | Bilinear interaction |

## 3. Top Templates by S1 Rate

| Template | S1 Rate | Loss Ratio | PPL | Induction |
|----------|---------|------------|-----|-----------|
| latent_attn_ssm_hybrid | 47% | 0.576 | 12.8 | 0.004 |
| local_attn_ssm_hybrid | 44% | 0.594 | — | — |
| multiscale_difficulty_router_adaptive_attn_ssm | 50% | — | — | — |
| latent_attn_ffn_block | 41.5% | — | — | 0.004 |
| recursive_depth_router | 37.7% | 0.616 | 6.0 | — |
| routed_bottleneck | 32.3% | 0.584 | 9.0 | — |
| graph_attn_ffn_block | 40% | — | — | — |

## 4. Evaluation Protocol

### Quick Screening (5 min)
```bash
# Build, compile, forward pass + gradient check
python -m pytest research/tests/test_template_optimization.py -m unit -q
```

### Extended Training (30 min)
```bash
# 1000 steps on GPU with binding probes
python -m research.tools.eval_templates --steps 1000 --device cuda
```

### Full Pipeline Evaluation (2+ hrs)
```bash
# 10000 steps + all probes + wikitext
python -m research.tools.eval_templates --steps 10000 --device cuda
```

### Metrics to Track at Each Checkpoint

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| `loss_ratio` | final_loss / init_loss | < 0.60 (screening) |
| `perplexity` | exp(loss) on WikiText | < 10.0 (validation) |
| `induction_auc` | Induction head formation | > 0.004 |
| `binding_auc` | Copy-at-distance | > 0.004 |
| `ar_auc` | Associative recall | > 0.05 |
| `hellaswag_acc` | Commonsense reasoning | > 0.28 |
| `s1_passed` | Passes screening stage 1 | True |

## 5. Template Optimization Cycle

### Step 1: Identify Weak Templates
```sql
-- Query from LabNotebook
SELECT
    json_extract(graph_json, '$.metadata.templates_used[0]') AS template,
    COUNT(*) AS n,
    AVG(CASE WHEN stage1_passed THEN 1.0 ELSE 0.0 END) AS s1_rate,
    AVG(loss_ratio) AS mean_loss_ratio,
    AVG(induction_auc) AS mean_induction,
    AVG(binding_auc) AS mean_binding
FROM program_results
WHERE graph_json IS NOT NULL
GROUP BY template
HAVING n >= 10
ORDER BY s1_rate DESC;
```

### Step 2: Diagnose Root Cause
For each weak template, check:
1. **Has attention?** grep for attention ops in the template function
2. **Has FFN?** grep for swiglu_mlp, gelu_mlp, or _FFN_CLASSES pick
3. **Has residuals?** count `_residual()` calls (need >= 2)
4. **Parallel or sequential?** Are attention and other paths branched from same `normed` input?
5. **Gradient killers?** Look for reciprocal, block_sparse(density<0.5), div_safe as sole nonlinearity

### Step 3: Apply Fix Pattern
Replace the broken part with the winning pattern:

```python
def tpl_fixed_template(graph, input_id, rng, weights=None):
    """norm -> {attention || SSM} -> merge -> residual -> norm -> FFN -> residual."""
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Path A: attention (pick from available attention ops)
    pa = _add(graph, "latent_attention_compressor", [normed], context="template.attn")
    pa = _add(graph, "linear_proj", [pa], {"out_dim": D}, context="template.attn_proj")

    # Path B: SSM (use state_space directly to avoid context rule violations)
    pb = _add(graph, "state_space", [normed], context="template.ssm")
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(graph, pa, pb, context="template.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(graph, input_id, merged, context="template.mid")

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, _FFN_CLASSES, weights)
    ffned = _instantiate_motif(graph, normed2, ffn, rng) if ffn else normed2
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid, ffned, context="template.output")
```

### Step 4: Validate
```bash
# 1. Syntax check
python -m py_compile research/synthesis/_templates_XXX.py

# 2. Build + compile + forward
python -c "
from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_graph
from research.synthesis.templates import apply_template
import random, torch

g = ComputationGraph(model_dim=128)
inp = g.add_input()
out = apply_template(g, inp, random.Random(42), template_name='TEMPLATE_NAME')
g.set_output(out)
layer = compile_graph(g, use_ir=True)
x = torch.randn(2, 16, 128, requires_grad=True)
y = layer(x)
y.sum().backward()
print('OK', y.shape, x.grad.norm().item())
"

# 3. Multi-seed validation
python -c "
from research.synthesis.graph import ComputationGraph
from research.synthesis.validator import validate_graph
from research.synthesis.templates import apply_template
import random

failures = 0
for seed in range(25):
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    out = apply_template(g, inp, random.Random(seed), template_name='TEMPLATE_NAME')
    g.set_output(out)
    result = validate_graph(g)
    if not result.valid:
        failures += 1
print(f'{failures}/25 validation failures')
"

# 4. Run existing tests
python -m pytest research/tests/test_template_optimization.py research/tests/test_slot_template_wiring.py -m unit -q
```

### Step 5: Train and Measure
```bash
python -m research.tools.eval_templates --steps 1000 --templates TEMPLATE_NAME --device cuda
```

### Step 6: Update Registry
In `templates.py`:
- Set weight > 0 for fixed templates (3.0-5.0 based on expected quality)
- Set weight = 0 for retired templates
- Add to COMPONENT_GRAPH_EXEMPT_TEMPLATES if needed
- Update test assertions if template structure changed

## 6. Storing Evaluation Data in LabNotebook

### Option A: Use `record_program_result()` (full integration)
```python
from research.scientist.notebook.notebook_core import LabNotebook
import json

nb = LabNotebook()  # Opens default DB

# Create experiment first
exp_id = nb.start_experiment(
    experiment_type="template_optimization_eval",
    config={"templates": ["name1", "name2"], "steps": 1000}
)

# Record result for each template
result_id = nb.record_program_result(
    experiment_id=exp_id,
    graph_fingerprint=graph.fingerprint,  # from ComputationGraph
    graph_json=json.dumps(graph.to_dict()),
    stage0_passed=True,
    stage05_passed=True,
    stage1_passed=True,  # if loss improved
    loss_ratio=final_loss / init_loss,
    param_count=n_params,
    n_train_steps=1000,
    induction_auc=probe_results["induction_auc"],
    binding_auc=probe_results["binding_auc"],
    ar_auc=probe_results.get("ar_auc"),
    model_source="template_optimization_eval",
    bypass_quality_gate=True,  # Always store for analysis
)
nb.flush_writes()
```

### Option B: Use `store_probe_results()` (update existing entries)
```python
from research.tools.backfill import store_probe_results

store_probe_results(
    nb,
    result_id="existing_result_id",
    updates={
        "induction_auc": 0.038,
        "binding_auc": 0.004,
        "ar_auc": 0.05,
        "hellaswag_acc": 0.29,
        "wikitext_perplexity": 8.5,
    },
    provenance_context={
        "backfill_type": "template_optimization_eval",
        "timestamp": time.time(),
        "steps": 10000,
    },
)
```

### Option C: Direct SQL for template analytics
```python
# Query template performance from existing data
import sqlite3
conn = sqlite3.connect("research/aria_notebook.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT
        json_extract(graph_json, '$.metadata.templates_used[0]') AS template,
        COUNT(*) AS n,
        AVG(CASE WHEN stage1_passed THEN 1.0 ELSE 0.0 END) AS s1_rate,
        AVG(loss_ratio) AS mean_lr,
        AVG(induction_auc) AS mean_ind,
        AVG(binding_auc) AS mean_bind
    FROM program_results
    WHERE graph_json IS NOT NULL
      AND template IS NOT NULL
    GROUP BY template
    HAVING n >= 5
    ORDER BY s1_rate DESC
""").fetchall()
```

## 7. Template Structural Patterns Reference

### Pattern: Parallel Attention+SSM Hybrid (BEST)
- **Examples**: latent_attn_ssm_hybrid, local_attn_ssm_hybrid
- **S1 Rate**: 44-47%
- **Structure**: norm -> {attn || SSM} -> merge -> residual -> norm -> FFN -> residual

### Pattern: Attention+FFN Block (GOOD)
- **Examples**: latent_attn_ffn_block, graph_attn_ffn_block
- **S1 Rate**: 32-41%
- **Structure**: norm -> attn -> residual -> norm -> FFN -> residual

### Pattern: Routing+Depth (GOOD for novelty)
- **Examples**: recursive_depth_router, routed_bottleneck
- **S1 Rate**: 32-38%
- **Structure**: norm -> difficulty_scorer -> {easy || medium || hard} -> merge -> FFN

### Pattern: Sequential Exotic (RISKY)
- **Examples**: tropical_residual, cosine_scoring, decay_sequence
- **S1 Rate**: 5-15%
- **Structure**: norm -> exotic_op -> residual (no FFN, no parallel path)
- **Fix**: Add parallel attention path + FFN

### Pattern: Reference Architecture (BASELINE)
- **Examples**: gpt2_reference, mamba_reference
- **S1 Rate**: 30% (GPT-2)
- **Structure**: norm -> attn -> proj -> residual -> norm -> FFN -> residual

## 8. Common Pitfalls

1. **Using `_pick_compatible_motif(MOTIF_CLASS_SSM)` in parallel paths**: Some SSM motifs (sparse_span_builder, cumsum) have context rules that fail validation. Use `state_space` directly.

2. **Adding matmul as core transformation**: Matmul computes bilinear form but loses dimensionality control. Use it only inside FFN, not as the main mixer.

3. **Over-parameterization**: 4+ residual blocks with redundant projections waste parameters. The sweet spot is 3 residuals.

4. **Missing `_fix_dim()`**: If an op changes dimensionality, always call `_fix_dim()` before merging/residual.

5. **Sparse compression as FFN substitute**: `block_sparse_linear(density=0.25)` kills 75% of gradients. Never use as sole nonlinearity.

## 9. Automated Template Audit Query

```python
# Run this to identify all templates needing optimization
from research.synthesis.templates import TEMPLATES, DEFAULT_TEMPLATE_WEIGHTS

for name, fn in TEMPLATES.items():
    w = DEFAULT_TEMPLATE_WEIGHTS.get(name, 1.0)
    if w == 0:
        continue  # Already retired
    
    # Get source code
    import inspect
    src = inspect.getsource(fn)
    
    has_attention = any(op in src for op in [
        'softmax_attention', 'latent_attention_compressor',
        'graph_attention', 'local_window_attn', 'linear_attention',
        'diff_attention', 'MOTIF_CLASS_ATTENTION'
    ])
    has_ffn = any(op in src for op in [
        'swiglu_mlp', 'gelu_mlp', '_FFN_CLASSES', 'fused_linear_gelu'
    ])
    n_residuals = src.count('_residual(') + src.count('template_add_residual(')
    has_parallel = src.count('normed]') >= 2 or src.count('[normed,') >= 1
    
    issues = []
    if not has_attention:
        issues.append("NO_ATTENTION")
    if not has_ffn:
        issues.append("NO_FFN")
    if n_residuals < 2:
        issues.append(f"LOW_RESIDUALS({n_residuals})")
    if not has_parallel:
        issues.append("SEQUENTIAL")
    
    if issues:
        print(f"  WEAK: {name:45s} w={w:.1f}  issues={', '.join(issues)}")
```
