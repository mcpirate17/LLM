# Reference Architecture Baselines & Robustness Plan

**Created:** 2026-02-24
**Coordination file for:** claude-opus, gemini, codex
**Status:** ACTIVE

---

## Goal

Add known-good reference architectures (GPT-2, Mamba, Retrieval-Augmented LLM, RWKV) to the eval pipeline, pin them on the Discoveries leaderboard, and fix the broken robustness metrics so we can objectively compare Aria's novel architectures against established baselines.

---

## Phase 0: Fix Broken Infrastructure (BLOCKING — do first)

### P0.1 — Fix robustness metrics pipeline [gemini]
- [x] Debug `promote_to_tier()` in `scientist/notebook.py` (line ~3716-3780) — column whitelist isn't propagating robustness values
- [x] Trace runner.py `program_metrics` → `upsert_leaderboard()` path for: `quant_int8_retention`, `robustness_long_ctx_score`, `robustness_noise_score`, `init_sensitivity_std`
- [x] Current DB state: 378 leaderboard entries, only 1 has quant/noise/init data, 0 have long_ctx. The eval code runs but values are lost.
- [x] Write integration test: run a known-good program through eval, assert robustness columns are non-null in leaderboard after promotion
- Files: `scientist/runner.py` (~lines 7272, 7305, 11708, 11741), `scientist/notebook.py` (upsert_leaderboard, promote_to_tier)

### P0.2 — Fix tier promotion bottleneck [gemini]
- [x] 375/378 entries stuck at "screening" tier — investigate why screening→investigation promotion isn't triggering
- [x] Top program (loss ratio 2.23e-06) is only at investigation — trace why it hasn't promoted to validation
- [x] Check promotion gate thresholds in runner.py — may be too strict or not being triggered
- Files: `scientist/runner.py` (promotion logic), `scientist/notebook.py` (promote_to_tier)

### P0.3 — Add "pinned" / "reference" support to leaderboard [claude-opus 2026-02-24]
- [x] Add `is_reference` BOOLEAN column to leaderboard table (default FALSE)
- [x] Add `reference_name` TEXT column (e.g. "GPT-2 Small", "Mamba-130M")
- [x] Update `upsert_leaderboard()` and `get_leaderboard()` to support reference fields
- [x] Reference entries always appear in Discoveries regardless of tier (dedup excludes refs)
- [x] Add `model_source` value "reference" for these entries (column already exists)
- [x] Add notebook methods: `pin_reference(entry_id, name)` and `get_references()`
- Files: `scientist/notebook.py`

### P0.4 — Dashboard: show pinned references [codex]
- [x] Discoveries.js: Add "Reference Baselines" section at top, always visible
- [x] Leaderboard.js: Pin icon + reference label for pinned entries, always sort to top
- [x] Add visual comparison: show Aria candidates' metrics as % of reference (e.g. "92% of GPT-2 loss")
- [x] Add filter toggle: "Show references" / "Hide references"
- Files: `dashboard/src/components/Discoveries.js`, `dashboard/src/components/Leaderboard.js`

---

## Phase 1: Missing Kernels for Reference Architectures

### Coverage Summary
| Architecture       | Current | Blocking Ops                                    |
|--------------------|---------|-------------------------------------------------|
| GPT-2              | 95%     | embedding_lookup, rope_rotate                   |
| Mamba              | 85%     | gated_linear, channel_expansion                 |
| Retrieval-Aug      | 70%     | cosine_similarity, gather_topk, embedding_lookup|
| RWKV               | 90%     | rwkv_time_mixing (have rwkv_channel already)    |

### P1.1 — Core missing C kernels [claude-opus 2026-02-24]
All in `aria-designer/runtime/src/kernels.c` + `kernels.h`:
- [x] `aria_embedding_lookup_f32()` + backward — token ID → dense vector
- [x] `aria_rope_rotate_f32()` — true rotary position embedding (SIMD)
- [x] `aria_gated_linear_f32()` + backward — fused (x @ W) * sigmoid(x @ W_gate)
- [x] `aria_cosine_similarity_f32()` — normalized dot product
- [x] `aria_gather_topk_f32()` — top-k vector selection by score
- [x] `aria_rwkv_time_mixing_f32()` — RWKV WKV linear attention kernel
- [ ] Add FP16 variants for the above where applicable
- Build: `cd aria-designer/runtime && make clean && make build`

### P1.2 — Register new primitives + compiler handlers [claude-opus 2026-02-24]
- [x] Added 7 new ops to PRIMITIVE_REGISTRY (83 total): layernorm, embedding_lookup, rope_rotate, gated_linear, cosine_similarity, gather_topk, rwkv_time_mixing
- [x] Added compiler execution handlers (@register_op) for all 7 new ops
- [x] Fixed IRExecutor shape tracking (was always using model_dim, now tracks per-node output dims)
- [ ] Add to `component_mapping.yaml` in aria-designer
- Files: `synthesis/primitives.py`, `synthesis/compiler.py`, `synthesis/ir_executor.py`

### P1.3 — Cython bridge wrappers [claude-opus 2026-02-24]
- [x] Add dispatch entries in `runtime/native/cython/aria_bridge.pyx` for all P1.1 ops
- [x] Add to `_NATIVE_C_KERNEL_OPS` in `scientist/native_runner.py`
- [ ] Update `runtime/native/include/kernel_abi.h` if ABI changes needed
- Build: `cd research/runtime/native/cython && python setup.py build_ext --inplace`

### P1.4 — Rust scheduler support [codex]
- [x] Register new op types in `runtime/native/rust/aria-scheduler/src/ffi.rs`
- [x] Add dispatch arms for new kernel calls
- Build: `cd research/runtime/native/rust/aria-scheduler && maturin develop --release`
- [x] Build/install validated in venv via `maturin develop --release` (2026-02-24)

### P1.5 — Designer component manifests [gemini]
- [x] Create manifest YAML for each new component in `aria-designer/components/`:
  - `embedding_lookup.yaml`, `rope_rotate.yaml`, `gated_linear.yaml`
  - `cosine_similarity.yaml`, `gather_topk.yaml`
  - `rwkv_time_mixing.yaml`
- [x] Each needs: name, category, ports (with dtype contract), default params, description
- [x] Validate: `python aria-designer/tools/validate_manifests.py`
- Files: `aria-designer/components/`

---

## Phase 2: Build Reference Architecture Graphs

### P2.1 — GPT-2 Small reference [claude-opus 2026-02-24]
- [x] Built `build_gpt2_layer()` in `synthesis/reference_architectures.py`
  - Architecture: LN → softmax_attention → residual → LN → linear(D→4D) → GELU → linear(4D→D) → residual
  - compile_model stacks N layers with embedding/output projection
- [ ] Build aria-designer workflow JSON in `aria-designer/workflows/reference_gpt2.json`

### P2.2 — Mamba reference [claude-opus 2026-02-24]
- [x] Built `build_mamba_layer()` in `synthesis/reference_architectures.py`
  - Architecture: LN → conv1d → SiLU → selective_scan → gated_linear → residual
- [ ] Build aria-designer workflow JSON in `aria-designer/workflows/reference_mamba.json`

### P2.3 — Retrieval-Augmented LLM reference [gemini]
- [x] `build_retrieval_augmented(d_model=256, n_layers=8, n_retrieval_layers=2, top_k=4)`
  - Architecture: embedding → alternating(self_attn_block, cross_attn_retrieval_block) → output_proj
  - Retrieval block: query_proj → cosine_sim(query, memory_bank) → gather_topk → cross_attention → residual
  - Memory bank: fixed external embedding matrix (simulated)
- [x] Also build as aria-designer workflow JSON

### P2.4 — RWKV reference [gemini]
- [x] `build_rwkv(d_model=256, n_layers=12)`
  - Architecture: embedding → 12x(LN → time_mixing → residual → LN → channel_mixing → residual) → LN → output_proj
  - Time mixing: RWKV linear attention (WKV kernel) with learned decay
  - Channel mixing: RWKV channel mix (key/value/receptance gating)
  - Already have `rwkv_channel` in Cython bridge — reuse it
- [x] Also build as aria-designer workflow JSON

### P2.5 — Reference architecture runner [claude-opus 2026-02-24]
- [x] Created `tools/register_references.py`:
  - Builds all 4 reference architectures via compile_model (N layers)
  - Runs safe_eval → micro-train → baseline comparison → upsert_leaderboard → pin_reference
  - CLI: `python -m research.tools.register_references --arch all --device cpu`
- [x] Created `tests/test_reference_architectures.py` (39 tests, all passing)
- [x] Fixed IRExecutor shape tracking bug (was ignoring out_dim config)
- [ ] Add to `__main__.py` as `--mode=register-references`

---

## Phase 3: Dashboard Robustness & Comparison UI

### P3.1 — Robustness profile in ProgramDetail [codex]
- [x] Add "Robustness Profile" card to ProgramDetail.js showing:
  - Noise sensitivity score (0-1 gauge, lower = more robust)
  - Long-context score (0-1 gauge, higher = better scaling)
  - Init sensitivity std (bar, lower = less init-dependent)
  - Quantization INT8 retention % and quality-per-byte
  - Spectral norm from fingerprint
- Files: `dashboard/src/components/ProgramDetail.js`

### P3.2 — Leaderboard robustness columns [codex]
- [x] Add optional columns: noise_score, quant_retention, long_ctx_score
- [x] Add sort-by dropdown for robustness metrics
- [x] Add filter: "Only show robust" (noise_score < 0.3 AND quant_retention > 80%)
- Files: `dashboard/src/components/Leaderboard.js`

### P3.3 — Reference comparison overlay [gemini]
- [ ] In ProgramDetail: "Compare to Reference" dropdown
  - Select GPT-2 / Mamba / RAG / Diffusion
  - Show side-by-side metrics radar chart
  - Highlight where candidate beats/loses to reference
- [ ] In Discoveries: "vs Reference" column showing % improvement over best matching reference
- Files: `dashboard/src/components/ProgramDetail.js`, `Discoveries.js`

### P3.4 — Scoring engine update [gemini]
- [ ] Incorporate robustness metrics into `scoringEngine.js` composite score
- [ ] Add "reference_delta" bonus: candidates that beat a reference architecture get a tier bonus
- Files: `dashboard/src/scoringEngine.js` (or equivalent)

---

## Phase 4: Integration Tests & Validation

### P4.1 — Reference architecture tests [claude-opus]
- [ ] `tests/test_reference_architectures.py`:
  - Each reference builds without error
  - Each compiles to a valid torch.nn.Module
  - Each produces non-NaN output on random input
  - Each can train for 10 steps without gradient explosion
  - Each gets correct novelty_score (GPT-2 ≈ 0.0, novel archs > 0.3)
- [ ] Test that references appear on leaderboard after registration
- [ ] Test that pinned references survive leaderboard queries

### P4.2 — Kernel tests for new ops [claude-opus]
- [ ] C kernel unit tests for all P1.1 ops (forward + backward)
- [ ] Cython dispatch tests
- [ ] Numerical parity tests: C kernel output ≈ PyTorch reference (atol=1e-5)

### P4.3 — End-to-end reference pipeline test [codex]
- [x] Added and validated endpoint-level pinned-reference e2e (`research/tests/test_reference_registration_e2e.py`) in venv (2026-02-24)
- [ ] Build GPT-2 reference → eval → leaderboard → pin → dashboard API returns it
- [ ] Build Mamba reference → eval → leaderboard → pin → compare against GPT-2
- [ ] Verify Aria-generated program shows meaningful novelty_score vs references

---

## Agent Assignment Summary

| Agent       | Primary Responsibilities                                     |
|-------------|--------------------------------------------------------------|
| claude-opus | P1.1-P1.3 (C/Cython kernels), P2.1-P2.2 (GPT-2/Mamba graphs), P4.1-P4.2 (tests) |
| gemini      | P0.1-P0.2 (robustness fix, tier fix), P1.5 (manifests), P2.3-P2.4 (RAG/Diffusion graphs), P3.3-P3.4 (comparison UI) |
| codex       | P0.3-P0.4 (pin support, dashboard), P1.4 (Rust), P2.5 (registration tool), P3.1-P3.2 (robustness UI), P4.3 (e2e test) |

---

## Dependency Order

```
P0.1 (fix robustness) ──┐
P0.2 (fix tiers)  ──────┤
P0.3 (pin support) ─────┼──→ P2.5 (register tool) ──→ P4.3 (e2e test)
P0.4 (dashboard pin) ───┘         ↑
                                  │
P1.1 (C kernels) ──→ P1.2 (primitives) ──→ P1.3 (Cython) ──→ P2.1-P2.4 (build graphs)
P1.4 (Rust) ────────────────────────────────────────┘              │
P1.5 (manifests) ──────────────────────────────────────────────────┘
                                                                   │
P3.1-P3.4 (dashboard) ← requires P0.1 + P0.3                     │
P4.1-P4.2 (tests) ← requires P1.1 + P2.1-P2.4 ──────────────────┘
```

---

## How to Use This File

1. Read before starting work
2. Claim a task: change `[ ]` to `[agent_name YYYY-MM-DD]`
3. When done: change to `[x]`
4. If blocked: add a note under the task explaining why
5. Re-read before editing to avoid conflicts

---

*Last updated: 2026-02-24 19:20 by claude-opus — All 4 references registered & pinned. Fixed selective_scan/rwkv_time_mixing gradient flow. P1.3 Cython wrappers done.*
