# Low-S1 Root Cause Audit

## graph_attention
- Classification: B. stale evidence from fixed bug, M. valid but weak
- Evidence: obs=96 compile_rate=99.0% s1_rate=0.0% recent_obs=96 recent_s1=0 bf16_errors=66 init_poisoned=0 top_errors={'RuntimeError': 66, 'unstable_dynamics': 11, 'causality_violation': 9, 's1_fail': 5}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus graph_attention`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: no

## local_window_attn
- Classification: B. stale evidence from fixed bug
- Evidence: obs=75 compile_rate=0.0% s1_rate=0.0% recent_obs=75 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'OutOfResources': 46, 'nan_forward': 22, 'forward_error': 6, 'cuda_fatal': 1}
- Code path: `research/synthesis/compiler_ops_attention.py::_op_local_window_attn`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus local_window_attn`
- Code fix needed: yes
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: no

## state_space
- Classification: C. vocab/gate mismatch artifact, H. numerical instability, M. valid but weak
- Evidence: obs=68 compile_rate=97.1% s1_rate=0.0% recent_obs=68 recent_s1=0 bf16_errors=2 init_poisoned=5 top_errors={'unstable_dynamics': 42, 'causality_violation': 13, 's1_fail': 8, 'RuntimeError': 2}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus state_space`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: yes

## sliding_window_mask
- Classification: B. stale evidence from fixed bug, K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=66 compile_rate=100.0% s1_rate=0.0% recent_obs=66 recent_s1=0 bf16_errors=54 init_poisoned=0 top_errors={'RuntimeError': 54, 'unstable_dynamics': 9, 'rapid_screening_error': 2, 'causality_violation': 1}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus sliding_window_mask`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## causal_mask
- Classification: B. stale evidence from fixed bug, K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=63 compile_rate=95.2% s1_rate=0.0% recent_obs=63 recent_s1=0 bf16_errors=41 init_poisoned=0 top_errors={'RuntimeError': 41, 'unstable_dynamics': 12, 'causality_violation': 5, 'cuda_fatal': 2}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus causal_mask`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## tropical_center
- Classification: C. vocab/gate mismatch artifact, D. graph-context misuse, H. numerical instability, L. niche op needing restricted placement
- Evidence: obs=62 compile_rate=96.8% s1_rate=0.0% recent_obs=62 recent_s1=0 bf16_errors=4 init_poisoned=6 top_errors={'unstable_dynamics': 21, 's1_fail': 18, 'causality_violation': 16, 'RuntimeError': 4}
- Code path: `research/synthesis/compiler_ops_mathspaces.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus tropical_center`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## early_exit
- Classification: D. graph-context misuse, L. niche op needing restricted placement
- Evidence: obs=60 compile_rate=68.3% s1_rate=0.0% recent_obs=60 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 21, 'rapid_screening_error': 20, 'forward_error': 19}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus early_exit`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## gated_delta
- Classification: M. valid but weak
- Evidence: obs=55 compile_rate=87.3% s1_rate=0.0% recent_obs=55 recent_s1=0 bf16_errors=22 init_poisoned=0 top_errors={'RuntimeError': 22, 'unstable_dynamics': 12, 'forward_error': 7, 's1_fail': 6}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus gated_delta`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## diff_attention
- Classification: M. valid but weak
- Evidence: obs=54 compile_rate=96.3% s1_rate=0.0% recent_obs=54 recent_s1=0 bf16_errors=27 init_poisoned=0 top_errors={'RuntimeError': 27, 'unstable_dynamics': 12, 'causality_violation': 8, 's1_fail': 5}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus diff_attention`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## fused_linear_gelu
- Classification: M. valid but weak
- Evidence: obs=54 compile_rate=66.7% s1_rate=0.0% recent_obs=54 recent_s1=0 bf16_errors=29 init_poisoned=0 top_errors={'RuntimeError': 23, 'forward_error': 16, 's1_fail': 11, 'rapid_screening_error': 2}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus fused_linear_gelu`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## integral_kernel
- Classification: M. valid but weak
- Evidence: obs=50 compile_rate=92.0% s1_rate=0.0% recent_obs=50 recent_s1=0 bf16_errors=23 init_poisoned=0 top_errors={'RuntimeError': 21, 'causality_violation': 10, 'rapid_screening_error': 7, 'unstable_dynamics': 6}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus integral_kernel`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## split3
- Classification: K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=50 compile_rate=62.0% s1_rate=0.0% recent_obs=50 recent_s1=0 bf16_errors=14 init_poisoned=0 top_errors={'forward_error': 19, 's1_fail': 16, 'rapid_screening_error': 15}
- Code path: `research/synthesis/motifs.py / research/synthesis/templates.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus split3`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: yes

## exp
- Classification: H. numerical instability, M. valid but weak
- Evidence: obs=47 compile_rate=91.5% s1_rate=0.0% recent_obs=47 recent_s1=0 bf16_errors=2 init_poisoned=0 top_errors={'s1_fail': 23, 'unstable_dynamics': 16, 'forward_error': 4, 'rapid_screening_error': 4}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus exp`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## cumprod_safe
- Classification: none
- Evidence: obs=40 compile_rate=97.5% s1_rate=0.0% recent_obs=40 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 33, 'rapid_screening_error': 6, 'forward_error': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus cumprod_safe`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## lif_neuron
- Classification: D. graph-context misuse, L. niche op needing restricted placement
- Evidence: obs=40 compile_rate=85.0% s1_rate=0.0% recent_obs=40 recent_s1=0 bf16_errors=1 init_poisoned=0 top_errors={'s1_fail': 17, 'causality_violation': 10, 'rapid_screening_error': 7, 'cuda_fatal': 4}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus lif_neuron`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## sparse_threshold
- Classification: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Evidence: obs=36 compile_rate=63.9% s1_rate=0.0% recent_obs=36 recent_s1=0 bf16_errors=3 init_poisoned=0 top_errors={'rapid_screening_error': 14, 'nan_forward': 9, 's1_fail': 9, 'forward_error': 3}
- Code path: `research/synthesis/compiler_ops_mathspaces.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus sparse_threshold`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## stdp_attention
- Classification: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Evidence: obs=36 compile_rate=63.9% s1_rate=0.0% recent_obs=36 recent_s1=0 bf16_errors=3 init_poisoned=0 top_errors={'rapid_screening_error': 14, 'nan_forward': 9, 's1_fail': 9, 'forward_error': 3}
- Code path: `research/synthesis/compiler_ops_mathspaces.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus stdp_attention`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## cumsum
- Classification: M. valid but weak
- Evidence: obs=31 compile_rate=80.6% s1_rate=0.0% recent_obs=31 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 9, 'rapid_screening_error': 8, 'unstable_dynamics': 7, 'forward_error': 6}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus cumsum`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## sub
- Classification: M. valid but weak
- Evidence: obs=31 compile_rate=96.8% s1_rate=0.0% recent_obs=31 recent_s1=0 bf16_errors=10 init_poisoned=0 top_errors={'rapid_screening_error': 19, 'RuntimeError': 10, 'cuda_fatal': 1, 's1_fail': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus sub`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## chebyshev_spectral_mix
- Classification: M. valid but weak
- Evidence: obs=30 compile_rate=93.3% s1_rate=0.0% recent_obs=30 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 21, 'rapid_screening_error': 5, 'causality_violation': 2, 'cuda_fatal': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus chebyshev_spectral_mix`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## minimum
- Classification: M. valid but weak
- Evidence: obs=24 compile_rate=91.7% s1_rate=0.0% recent_obs=24 recent_s1=0 bf16_errors=9 init_poisoned=0 top_errors={'s1_fail': 13, 'RuntimeError': 9, 'cuda_fatal': 2}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus minimum`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## embedding_lookup
- Classification: none
- Evidence: obs=23 compile_rate=78.3% s1_rate=0.0% recent_obs=23 recent_s1=0 bf16_errors=4 init_poisoned=0 top_errors={'s1_fail': 15, 'forward_error': 5, 'rapid_screening_error': 3}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus embedding_lookup`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## geometric_product
- Classification: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Evidence: obs=23 compile_rate=95.7% s1_rate=0.0% recent_obs=23 recent_s1=0 bf16_errors=7 init_poisoned=0 top_errors={'rapid_screening_error': 8, 'RuntimeError': 7, 's1_fail': 7, 'forward_error': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus geometric_product`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## n_way_sparse_router
- Classification: D. graph-context misuse, L. niche op needing restricted placement
- Evidence: obs=22 compile_rate=54.5% s1_rate=0.0% recent_obs=22 recent_s1=0 bf16_errors=2 init_poisoned=0 top_errors={'rapid_screening_error': 7, 's1_fail': 5, 'RuntimeError': 4, 'forward_error': 4}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus n_way_sparse_router`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## sign_ste
- Classification: H. numerical instability, M. valid but weak
- Evidence: obs=22 compile_rate=86.4% s1_rate=0.0% recent_obs=22 recent_s1=0 bf16_errors=5 init_poisoned=0 top_errors={'s1_fail': 11, 'RuntimeError': 4, 'forward_error': 3, 'rapid_screening_error': 3}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus sign_ste`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## tropical_matmul
- Classification: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Evidence: obs=22 compile_rate=95.5% s1_rate=0.0% recent_obs=22 recent_s1=0 bf16_errors=8 init_poisoned=0 top_errors={'rapid_screening_error': 12, 'RuntimeError': 8, 'cuda_fatal': 1, 's1_fail': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus tropical_matmul`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: no

## norm_last
- Classification: K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=21 compile_rate=76.2% s1_rate=0.0% recent_obs=21 recent_s1=0 bf16_errors=4 init_poisoned=0 top_errors={'unstable_dynamics': 7, 's1_fail': 6, 'forward_error': 4, 'rapid_screening_error': 3}
- Code path: `research/synthesis/motifs.py / research/synthesis/templates.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus norm_last`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: yes

## sqrt
- Classification: H. numerical instability, M. valid but weak
- Evidence: obs=20 compile_rate=100.0% s1_rate=0.0% recent_obs=20 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 10, 'unstable_dynamics': 9, 'rapid_screening_error': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus sqrt`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## log
- Classification: H. numerical instability, M. valid but weak
- Evidence: obs=19 compile_rate=100.0% s1_rate=0.0% recent_obs=19 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 10, 'unstable_dynamics': 6, 'rapid_screening_error': 3}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus log`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## mod_topk
- Classification: none
- Evidence: obs=19 compile_rate=89.5% s1_rate=0.0% recent_obs=19 recent_s1=0 bf16_errors=1 init_poisoned=0 top_errors={'rapid_screening_error': 11, 's1_fail': 6, 'forward_error': 2}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus mod_topk`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## sum_last
- Classification: K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=17 compile_rate=94.1% s1_rate=0.0% recent_obs=17 recent_s1=0 bf16_errors=1 init_poisoned=0 top_errors={'s1_fail': 7, 'rapid_screening_error': 5, 'unstable_dynamics': 4, 'forward_error': 1}
- Code path: `research/synthesis/motifs.py / research/synthesis/templates.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus sum_last`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: yes

## max_last
- Classification: K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=15 compile_rate=93.3% s1_rate=0.0% recent_obs=15 recent_s1=0 bf16_errors=1 init_poisoned=0 top_errors={'s1_fail': 11, 'RuntimeError': 1, 'unstable_dynamics': 1, 'rapid_screening_error': 1}
- Code path: `research/synthesis/motifs.py / research/synthesis/templates.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus max_last`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: yes

## mean_last
- Classification: K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=15 compile_rate=80.0% s1_rate=0.0% recent_obs=15 recent_s1=0 bf16_errors=5 init_poisoned=0 top_errors={'rapid_screening_error': 5, 's1_fail': 4, 'RuntimeError': 3, 'forward_error': 3}
- Code path: `research/synthesis/motifs.py / research/synthesis/templates.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus mean_last`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: no
- Screening interpretation must change: yes

## reciprocal
- Classification: none
- Evidence: obs=10 compile_rate=100.0% s1_rate=0.0% recent_obs=10 recent_s1=0 bf16_errors=0 init_poisoned=0 top_errors={'s1_fail': 9, 'rapid_screening_error': 1}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus reciprocal`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## linear_attention
- Classification: B. stale evidence from fixed bug, M. valid but weak
- Evidence: obs=158 compile_rate=98.1% s1_rate=0.6% recent_obs=157 recent_s1=0 bf16_errors=115 init_poisoned=0 top_errors={'RuntimeError': 115, 'unstable_dynamics': 22, 'causality_violation': 11, 'rapid_screening_error': 5}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus linear_attention`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: no

## identity
- Classification: C. vocab/gate mismatch artifact, K. structural/non-learnable op, L. niche op needing restricted placement, N. telemetry/reporting ambiguity
- Evidence: obs=306 compile_rate=94.8% s1_rate=0.7% recent_obs=306 recent_s1=2 bf16_errors=16 init_poisoned=20 top_errors={'causality_violation': 224, 's1_fail': 29, 'unstable_dynamics': 16, 'RuntimeError': 16}
- Code path: `research/synthesis/motifs.py / research/synthesis/templates.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus identity`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## progressive_compression_gate
- Classification: B. stale evidence from fixed bug, D. graph-context misuse, L. niche op needing restricted placement
- Evidence: obs=149 compile_rate=97.3% s1_rate=0.7% recent_obs=148 recent_s1=0 bf16_errors=112 init_poisoned=0 top_errors={'RuntimeError': 111, 'unstable_dynamics': 31, 'cuda_fatal': 3, 'rapid_screening_error': 2}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus progressive_compression_gate`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: no

## softmax_attention
- Classification: B. stale evidence from fixed bug, M. valid but weak
- Evidence: obs=240 compile_rate=97.1% s1_rate=0.8% recent_obs=238 recent_s1=0 bf16_errors=142 init_poisoned=0 top_errors={'RuntimeError': 141, 'unstable_dynamics': 57, 'causality_violation': 21, 's1_fail': 6}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus softmax_attention`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: no

## linear_proj_down
- Classification: B. stale evidence from fixed bug, C. vocab/gate mismatch artifact
- Evidence: obs=979 compile_rate=94.6% s1_rate=1.0% recent_obs=970 recent_s1=1 bf16_errors=745 init_poisoned=5 top_errors={'RuntimeError': 732, 'unstable_dynamics': 113, 's1_fail': 39, 'forward_error': 26}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus linear_proj_down`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: yes

## linear_proj_up
- Classification: none
- Evidence: obs=990 compile_rate=95.8% s1_rate=1.3% recent_obs=981 recent_s1=4 bf16_errors=690 init_poisoned=5 top_errors={'RuntimeError': 679, 'unstable_dynamics': 133, 's1_fail': 76, 'causality_violation': 23}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus linear_proj_up`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## concat
- Classification: B. stale evidence from fixed bug, C. vocab/gate mismatch artifact, K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=489 compile_rate=82.8% s1_rate=1.4% recent_obs=485 recent_s1=3 bf16_errors=194 init_poisoned=30 top_errors={'RuntimeError': 162, 's1_fail': 81, 'unstable_dynamics': 79, 'forward_error': 46}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus concat`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## hyp_linear
- Classification: C. vocab/gate mismatch artifact, D. graph-context misuse, G. shape/contract mismatch, I. gradient explosion, L. niche op needing restricted placement
- Evidence: obs=64 compile_rate=95.3% s1_rate=1.6% recent_obs=63 recent_s1=1 bf16_errors=1 init_poisoned=18 top_errors={'s1_fail': 38, 'rapid_screening_error': 11, 'failed_convergence': 5, 'inflight_grad_explosion': 4}
- Code path: `research/synthesis/compiler_ops_mathspaces.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus hyp_linear`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## hyp_tangent_nonlinear
- Classification: C. vocab/gate mismatch artifact, D. graph-context misuse, G. shape/contract mismatch, I. gradient explosion, L. niche op needing restricted placement
- Evidence: obs=64 compile_rate=95.3% s1_rate=1.6% recent_obs=63 recent_s1=1 bf16_errors=1 init_poisoned=18 top_errors={'s1_fail': 38, 'rapid_screening_error': 11, 'failed_convergence': 5, 'inflight_grad_explosion': 4}
- Code path: `research/synthesis/compiler_ops_mathspaces.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus hyp_tangent_nonlinear`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## rwkv_time_mixing
- Classification: B. stale evidence from fixed bug, M. valid but weak
- Evidence: obs=63 compile_rate=98.4% s1_rate=1.6% recent_obs=62 recent_s1=0 bf16_errors=50 init_poisoned=0 top_errors={'RuntimeError': 50, 'unstable_dynamics': 6, 'causality_violation': 5, 'forward_error': 1}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus rwkv_time_mixing`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: no

## rope_rotate
- Classification: B. stale evidence from fixed bug
- Evidence: obs=57 compile_rate=98.2% s1_rate=1.8% recent_obs=56 recent_s1=0 bf16_errors=32 init_poisoned=0 top_errors={'RuntimeError': 32, 'unstable_dynamics': 15, 'rapid_screening_error': 4, 'causality_violation': 2}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus rope_rotate`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: no

## split2
- Classification: B. stale evidence from fixed bug, C. vocab/gate mismatch artifact, K. structural/non-learnable op, L. niche op needing restricted placement
- Evidence: obs=389 compile_rate=88.2% s1_rate=1.8% recent_obs=385 recent_s1=3 bf16_errors=166 init_poisoned=30 top_errors={'RuntimeError': 162, 'unstable_dynamics': 79, 's1_fail': 49, 'nan_forward': 23}
- Code path: `research/synthesis/compiler.py::_cast_params_to`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus split2`
- Code fix needed: no
- Rules fix needed: yes
- Rerun needed: yes
- Screening interpretation must change: yes

## fixed_point_iter
- Classification: C. vocab/gate mismatch artifact, M. valid but weak
- Evidence: obs=55 compile_rate=100.0% s1_rate=1.8% recent_obs=55 recent_s1=1 bf16_errors=10 init_poisoned=14 top_errors={'s1_fail': 30, 'rapid_screening_error': 12, 'RuntimeError': 10, 'unstable_dynamics': 1}
- Code path: `research/synthesis/compiler_ops_mathspaces.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus fixed_point_iter`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: yes

## transpose_sd
- Classification: none
- Evidence: obs=55 compile_rate=94.5% s1_rate=1.8% recent_obs=55 recent_s1=1 bf16_errors=8 init_poisoned=21 top_errors={'s1_fail': 22, 'causality_violation': 11, 'rapid_screening_error': 8, 'RuntimeError': 7}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus transpose_sd`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: no
- Screening interpretation must change: no

## grouped_linear
- Classification: C. vocab/gate mismatch artifact, I. gradient explosion, M. valid but weak
- Evidence: obs=52 compile_rate=96.2% s1_rate=1.9% recent_obs=50 recent_s1=0 bf16_errors=9 init_poisoned=21 top_errors={'s1_fail': 32, 'RuntimeError': 8, 'unstable_dynamics': 3, 'rapid_screening_error': 3}
- Code path: `research/synthesis/compiler.py / compiler_ops_math.py / compiler_ops_attention.py`
- Reproduction: `python -m research.tools.audit_low_s1_components --focus grouped_linear`
- Code fix needed: no
- Rules fix needed: no
- Rerun needed: yes
- Screening interpretation must change: yes
