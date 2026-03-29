# Under-Observed Component Inventory

**Date**: 2026-03-21
**Source**: `op_success_rates` table in `research/lab_notebook.db`
**Threshold**: < 20 observations

## Summary

- **45 total under-observed components** identified
- **7 with zero observations** (never tracked in op_success_rates)
- **3 with 0% S0 pass rate** (always fail compilation/forward)
- **35 with 2-19 observations**

## Root Causes Fixed

| Root Cause | Ops Affected | Fix |
|---|---|---|
| Wrong role mapping | n_way_sparse_router | Added explicit ROUTE mapping in op_roles.py |
| UNSAFE role over-applied | cumprod_safe | Changed to REDUCE (safe within sigmoid motif) |
| C kernel shape bug | tropical_matmul | Added shape validation fallback in tropical.py |
| Non-contiguous split | split3, split2, split4 | Added .contiguous() to all split op returns |
| Motif class isolation | reduce_core, guarded_act, moe_core | Added to _FFN_CLASSES in templates.py |
| Zero-support motif deprioritization | 30+ ops | Added exploration_targets + boost_factor to GrammarConfig |

## Full Inventory

### Zero Observations (7 ops)

| Op | Root Cause | Status |
|---|---|---|
| cascade | min_depth=2, residual bypass, byte_safe=False, motif support=0 | FIXED: reachable via exploration mode |
| cumprod_safe | Was UNSAFE role (grammar excluded) | FIXED: changed to REDUCE |
| early_exit | min_depth=4, residual bypass, motif support=0 | FIXED: reachable via exploration mode |
| embedding_lookup | UNSAFE (needs token indices) — genuinely broken mid-graph | KEPT EXCLUDED |
| mod_topk | byte_safe=False, min_depth=2, residual bypass | FIXED: reachable via exploration mode |
| reciprocal | 4-step motif with low sampling probability | FIXED: compiles and forwards correctly |
| split3 | Template-only (tpl_three_way_split), dim>=24 | FIXED: contiguity bug resolved |

### Broken (0% S0 pass, 3 ops)

| Op | Observations | Root Cause | Status |
|---|---|---|---|
| n_way_sparse_router | 6 | Wrong role: fallback PROJECT instead of ROUTE | FIXED: explicit ROUTE mapping |
| sparse_threshold | 10 | Was in spiking motif with no empirical support | FIXED: now reachable and compiles |
| stdp_attention | 10 | Same spiking motif issue | FIXED: now reachable and compiles |

### Low Observations (35 ops, sorted by count)

| Op | Obs | S0% | S1% | Category | Motif Class | Fix Applied |
|---|---|---|---|---|---|---|
| max_last | 2 | 100 | 0 | REDUCTION | reduce_core | Added to _FFN_CLASSES |
| kronecker_linear | 2 | 100 | 50 | PARAMETERIZED | efficient_proj | Boost via exploration |
| mean_last | 3 | 100 | 0 | REDUCTION | reduce_core | Added to _FFN_CLASSES |
| sum_last | 4 | 100 | 0 | REDUCTION | reduce_core | Added to _FFN_CLASSES |
| padic_gate | 5 | 100 | 60 | PARAMETERIZED | math_space | Boost via exploration |
| log | 6 | 100 | 0 | ELEMENTWISE_UNARY | guarded_act | Added to _FFN_CLASSES |
| sign_ste | 6 | 100 | 0 | ELEMENTWISE_UNARY | guarded_act | Added to _FFN_CLASSES |
| chebyshev_spectral_mix | 6 | 83 | 0 | MIXING | channel_core | Boost via exploration |
| maximum | 7 | 71 | 29 | ELEMENTWISE_BINARY | template-only | Template weight boost |
| poincare_add | 7 | 100 | 71 | ELEMENTWISE_BINARY | math_space | Boost via exploration |
| geometric_product | 7 | 100 | 0 | ELEMENTWISE_BINARY | template-only | Template weight boost |
| norm_last | 8 | 88 | 0 | REDUCTION | reduce_core | Added to _FFN_CLASSES |
| cumsum | 9 | 100 | 0 | REDUCTION | reduce_core | Added to _FFN_CLASSES |
| sqrt | 9 | 100 | 0 | ELEMENTWISE_UNARY | guarded_act | Added to _FFN_CLASSES |
| tropical_matmul | 9 | 89 | 0 | ELEMENTWISE_BINARY | template-only | C kernel shape fix + template boost |
| div_safe | 10 | 80 | 70 | ELEMENTWISE_BINARY | template-only | Template weight boost |
| minimum | 11 | 82 | 0 | ELEMENTWISE_BINARY | template-only | Template weight boost |
| sub | 11 | 91 | 0 | ELEMENTWISE_BINARY | template-only | Template weight boost |
| hyp_distance | 12 | 83 | 25 | LINEAR_ALGEBRA | template-only | Template weight boost |
| shared_basis_proj | 13 | 100 | 8 | PARAMETERIZED | efficient_proj | Boost via exploration |
| tropical_moe | 14 | 100 | 50 | PARAMETERIZED | math_space | Boost via exploration |
| tropical_router | 14 | 86 | 64 | PARAMETERIZED | math_space | Boost via exploration |
| lif_neuron | 14 | 71 | 0 | ELEMENTWISE_UNARY | math_space | Boost via exploration |
| hyperbolic_norm | 15 | 93 | 47 | MIXING | math_space | Boost via exploration |
| spike_rate_code | 16 | 75 | 13 | ELEMENTWISE_UNARY | math_space | Boost via exploration |
| outer_product | 16 | 94 | 88 | LINEAR_ALGEBRA | template-only | Template weight boost |
| hyp_linear | 16 | 100 | 6 | PARAMETERIZED | math_space | Boost via exploration |
| hyp_tangent_nonlinear | 16 | 100 | 6 | ELEMENTWISE_UNARY | math_space | Boost via exploration |
| exp | 16 | 100 | 0 | ELEMENTWISE_UNARY | guarded_act | Added to _FFN_CLASSES |
| grouped_linear | 17 | 94 | 6 | PARAMETERIZED | efficient_proj | Boost via exploration |
| fixed_point_iter | 18 | 100 | 6 | PARAMETERIZED | ssm_core | Already well-supported |
| token_merge | 19 | 100 | 100 | FUNCTIONAL | ssm_core | Already well-supported |
| bottleneck_proj | 19 | 100 | 16 | PARAMETERIZED | efficient_proj | Boost via exploration |
| rotor_transform | 19 | 84 | 16 | MIXING | math_space | Boost via exploration |

## Verification Results

After fixes, 200-seed bulk test with exploration mode (boost=8x):

- **33/43 ops generated** (excluding embedding_lookup)
- **100% compile rate** for all generated ops
- **100% forward pass rate** (after contiguity + shape fixes)
- Remaining 10 ops verified individually with 500+ seeds each
