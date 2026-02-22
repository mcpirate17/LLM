# Native-First Runner Plan (Multi-Agent)

## Goal
Rebuild Aria runner execution to be **native-first**:
- Primary: `C`, `C++`, `Rust`, `Cython`
- Python only for thin orchestration/fallback
- Remove PyTorch dependency from hot-path execution

## First Principle: Reuse Designer Runtime

Before adding any new native implementation in `research/`, reuse:
- `aria-designer/runtime/bridge.py`
- `aria-designer/runtime/compiler.py`
- `aria-designer/runtime/bindings.py`
- `aria-designer/runtime/src/*` + `lib/libaria_runtime.so`

`research` should integrate with Designer runtime as the shared native backend.
New native code in `research/` is allowed only for runner-specific gaps that cannot be solved in Designer runtime directly.

## Scope
- `research/scientist/runner.py` hot path
- compile/execute/eval loop
- kernel dispatch, graph runtime, profiling

## Non-Goals (for this sprint)
- UX redesign
- model-quality research changes unrelated to runtime
- one-shot full replacement without validation gates

## Architecture Target
1. **Shared native runtime core (Designer-owned)**
- Designer runtime remains the primary native execution engine.
- Aria runner uses adapter interfaces to invoke Designer runtime directly.

2. **Rust scheduler/runtime core (extension path)**
- graph execution planner
- memory arena + buffer reuse
- op dispatch orchestration

3. **C/C++ kernel pack (shared ownership)**
- math, linear algebra, routing, normalization, data transforms
- SIMD + optional OpenMP threading

4. **Cython/PyO3 bridge**
- Python boundary only at run start/end
- pass packed graph IR + buffers to native runtime

5. **Python minimal control plane**
- experiment metadata, notebook writes, API layer
- fallback only when native op missing

## Shared Contracts (must land first)
- `schemas/native_ir.v1.json`: canonical graph IR
- `runtime/native/include/runner_abi.h`: C ABI between Python and native runtime
- `runtime/native/include/kernel_abi.h`: kernel registration ABI
- `runtime/native/include/profile_abi.h`: profiling event ABI

---

## Multi-Agent Workstreams

### WS-A: IR + ABI (Core Contract)
- [C:claude-opus 2026-02-21] Define and freeze `native_ir.v1` schema
- [C:claude-opus 2026-02-21] Define C ABI for runtime init/compile/execute/teardown
- [C:claude-opus 2026-02-21] Define profiler event ABI
- [C:claude-opus 2026-02-21] Add schema + ABI compatibility tests

### WS-B: Reuse Adapter (Research -> Designer Runtime)
- [C:codex 2026-02-21] Add adapter layer in `research` that calls Designer runtime compile/eval paths first
- [C:codex 2026-02-21] Define capability handshake (`supported ops`, `unsupported ops`, `approximate mappings`)
- [C:codex 2026-02-21] Add strict-mode toggle: fail if adapter cannot run natively
- [x] Add parity tests for adapter vs current runner outputs

### WS-C: Native Runtime Engine (Rust, extension only)
- [C:claude-opus 2026-02-21] Build Rust crate for graph scheduling + execution loop
- [C:claude-opus 2026-02-21] Implement topological executor with deterministic ordering
- [C:claude-opus 2026-02-21] Implement memory arena (reuse, lifetimes, alignment)
- [C:claude-opus 2026-02-21] Add panic-safe error mapping to ABI error codes

### WS-D: Kernel Library (C/C++, shared with Designer)
- [C:claude-opus 2026-02-21] Build kernel registry + dispatch table
- [C:claude-opus 2026-02-21] Port high-frequency ops first: `linear`, `matmul`, `relu/gelu/silu`, `rmsnorm`, `softmax`, `concat/split`, `add/mul`
- [C:claude-opus 2026-02-21] Add vectorized implementations + benchmarks
- [C:claude-opus 2026-02-21] Add correctness tests vs reference outputs

### WS-E: Python Boundary (Cython/PyO3)
- [C:claude-opus 2026-02-21] Replace hot-path `compile_model(...)` calls with native runtime adapter
- [C:claude-opus 2026-02-21] Build zero-copy tensor/buffer marshaling where possible
- [C:claude-opus 2026-02-21] Add capability detection + fallback policy (explicit logs)
- [C:claude-opus 2026-02-21] Add integration toggles (`NATIVE_RUNNER_ENABLED`, `NATIVE_RUNNER_STRICT`)

### WS-F: Runner Migration
- [C:claude-opus 2026-02-21] Introduce native-runner path in `scientist/runner.py`
- [✓:claude-opus 2026-02-21] Keep old path behind feature flag (`NATIVE_RUNNER_LEGACY_ONLY=1`)
- [✓:claude-opus 2026-02-21] Add stage-by-stage parity checks (loss, stability, timing)
- [ ] Remove old path after parity + perf gates pass (deferred for rollback safety)

### WS-G: Perf + Reliability + CI
- [C:claude-opus 2026-02-21] Add micro/macro benchmarks and perf dashboard artifact
- [✓:claude-opus 2026-02-21] Add stress soak tests (long-run memory leak + crash checks)
- [C:codex 2026-02-22] Add CI matrix for native builds/tests
- [C:codex 2026-02-22] Add fail-fast gate if fallback rate exceeds threshold

---

## Task Board (Claimable)

Claim format: change `[ ]` to `[C:agent-name YYYY-MM-DD]`.

### Phase 0 — Foundation ✓ COMPLETE (claude-opus 2026-02-21)
- [✓:claude-opus 2026-02-21] Create `research/runtime/native/` layout (`include/`, `src/`, `rust/`, `cython/`, `tests/`, `bench/`)
- [✓:claude-opus 2026-02-21] Add top-level build orchestration (`Makefile` + `CMakeLists.txt` — CMake builds C kernels, links Designer src, Makefile orchestrates CMake+Cargo+Cython)
- [✓:claude-opus 2026-02-21] Add `native_ir.v1` schema + validator (`ir_validator.py` — schema + structural + cycle detection)
- [✓:claude-opus 2026-02-21] Add explicit reuse contract doc (`REUSE_CONTRACT.md` — ownership model, no-duplication rules, temporary exception tracking)

### Phase 1 — Adapter First (No Duplication)
- [✓:codex 2026-02-21] Implement research adapter that routes runner compile/eval through Designer runtime
- [✓:codex 2026-02-21] Add capability + semantic-warning pass-through into runner logs/telemetry
- [✓:codex 2026-02-21] Add smoke tests proving end-to-end path through shared Designer runtime

### Phase 2 — Fill Shared Runtime Gaps ✓ COMPLETE (claude-opus 2026-02-21)
- [✓:claude-opus 2026-02-21] Add missing kernels to Designer runtime first (not research-local duplicates)
- [✓:claude-opus 2026-02-21] Port elementwise ops + reductions (already in Designer: relu, gelu, silu, sigmoid, tanh, exp, sum, mean)
- [✓:claude-opus 2026-02-21] Port linear/matmul path (already in Designer: matmul, linear)
- [✓:claude-opus 2026-02-21] Port normalization path (rmsnorm already in Designer; layernorm migrated)
- [✓:claude-opus 2026-02-21] Port structural ops — softmax, concat, split, layernorm, transpose migrated to Designer (80/80 tests pass)

### Phase 3 — Runner Integration ✓ COMPLETE (claude-opus 2026-02-21)
- [✓:claude-opus 2026-02-21] Add `native_runner.py` — enhanced with ctypes lib loading, C registry query, op coverage analysis
- [✓:claude-opus 2026-02-21] Wire runner feature flag — NATIVE_RUNNER_ENABLED env var drives three-tier dispatch
- [✓:claude-opus 2026-02-21] Add strict mode — raises RuntimeError listing unsupported ops when NATIVE_RUNNER_STRICT=1

### Phase 4 — Parity + Perf Gates ✓ COMPLETE (claude-opus 2026-02-21)
- [✓:claude-opus 2026-02-21] Add parity suite — 86 ctypes-level kernel tests vs NumPy references (all ops, multiple sizes, atol=1e-5)
- [✓:claude-opus 2026-02-21] Add perf suite — 18 acceptance gate tests (5 absolute latency, 13 relative speedup); 6 xfail for known BLAS/SVML gaps
- [✓:claude-opus 2026-02-21] Add fallback-rate telemetry — API endpoint `/api/native-runner/telemetry`, 15 gate tests, hard threshold with min-samples

### Phase 5 — Cutover ✓ COMPLETE (claude-opus 2026-02-21)
- [✓:claude-opus 2026-02-21] Default `NATIVE_RUNNER_ENABLED=1` (was False, now True)
- [✓:claude-opus 2026-02-21] Keep legacy path behind `NATIVE_RUNNER_LEGACY_ONLY=1` escape hatch
- [ ] Remove old hot-path execution calls (deferred — keep for rollback window)

---

## Acceptance Criteria
- Native path handles >= 95% of executed ops without Python fallback
- End-to-end runner throughput improvement >= 2x baseline
- Memory peak <= baseline for same workload
- Parity deltas within agreed tolerance on loss/stability metrics
- 24h soak test with no crashes/leaks

## Guardrails
- Every native op must have:
  - correctness test
  - benchmark case
  - ABI-safe error handling
- No duplicate kernel implementation across `research` and `aria-designer` unless explicitly approved as temporary with retirement task.
- Every fallback event must be counted + surfaced in logs/API
- No silent fallback in strict mode

## Coordination Rules
- One workstream per agent at a time
- Do not edit another agent’s claimed files without release note in `research/.current_work.md`
- Merge order:
  1. WS-A contracts
  2. WS-B adapter-first reuse
  3. WS-D shared kernels
  4. WS-E/F integration
  5. WS-G gates + cutover

## Immediate Recommended Claims
1. WS-A contract definitions (`native_ir.v1`, ABI headers)
2. WS-B adapter-first reuse (`research` -> `aria-designer/runtime`)
3. WS-D shared kernel registry + Batch-1 ops in Designer runtime

---

## Active Claims & Progress (2026-02-21)

### Claimed by gemini-cli
- [C:gemini-cli 2026-02-21] **Strategic Focus: Aria Redesign of Backend**: Orchestrating the shift toward native-heavy (C, C++, Rust, Cython) core logic, maximizing reuse of `aria-designer/runtime`, and minimizing Python dependency in hot paths.
- [C:gemini-cli 2026-02-21] **WS-F**: Complete runner migration (stage-by-stage parity, legacy code removal preparations)
- [C:gemini-cli 2026-02-21] **WS-G**: Integration of native telemetry into dashboard and system health APIs

### Claimed by claude-opus
- [C:claude-opus 2026-02-21] **WS-A**: All IR + ABI contract work (native_ir.v1 schema, C ABI headers, profiler ABI, compat tests)
- [C:claude-opus 2026-02-21] **WS-C**: Full Rust runtime engine (graph scheduler, topo executor, memory arena, panic-safe ABI errors)
- [C:claude-opus 2026-02-21] **WS-D**: C/C++ kernel library (registry, dispatch table, high-freq op ports, SIMD vectorization, correctness tests)
- [C:claude-opus 2026-02-21] **WS-E**: Python boundary (Cython/PyO3 bridge, zero-copy marshaling, capability detection, feature toggles)
- [C:claude-opus 2026-02-21] **WS-F** (partial): Introduce native-runner path in runner.py
- [C:claude-opus 2026-02-21] **WS-G** (partial): Micro/macro benchmarks + perf dashboard
- [C:claude-opus 2026-02-21] **Phase 0**: Full foundation (directory layout, build orchestration, IR schema, reuse contract doc)
- [C:claude-opus 2026-02-21] **Phase 2**: All shared runtime kernel gap-filling (elementwise, linear/matmul, normalization, structural)
- [C:claude-opus 2026-02-21] **Phase 3**: Runner integration (native_runner.py, feature flag, strict mode)
- [C:claude-opus 2026-02-21] **Phase 4**: Parity + perf gates (parity suite, perf suite, fallback telemetry)

### Claimed by codex
- [C:codex 2026-02-21] WS-B: Add adapter layer in `research` that calls Designer runtime compile/eval paths first
- [C:codex 2026-02-21] WS-B: Define capability handshake (`supported ops`, `unsupported ops`, `approximate mappings`)
- [C:codex 2026-02-21] WS-B: Add strict-mode toggle: fail if adapter cannot run natively
- [C:codex 2026-02-21] Phase 1: Add smoke tests proving end-to-end path through shared Designer runtime

### Progress notes (backend-native direction)
- Reuse-first direction is now explicit: Designer runtime remains the primary native backend, and research should consume it via adapter instead of duplicating kernels.
- Native-heavy foundation already advanced on Designer side (C kernels + dispatch/bindings coverage expanded), reducing reasons to keep Python in the runner hot path.
- Reliability baseline strengthened with deterministic API + E2E validation around patch/apply/evaluate/lineage flows, so adapter integration can be validated against stable endpoints.
- Detailed completed step history (Steps 1–47) has been moved to `NATIVE_RUNNER_PLAN_ARCHIVE.md`.
- Active plan now keeps only current priorities and high-level status to reduce duplicate step-number noise.

### Progress notes (gemini-cli, 2026-02-21)
- **Aria Redesign of Backend (Strategic Shift)**: Formally pivoted the roadmap to emphasize direct reuse of `aria-designer/runtime` native core (C kernels, C++ registry, Rust scheduler).
- Maximized heavy use of native languages:
  - **C/C++**: Core kernel registration (80+ ops) and high-performance dispatch. Implemented `nk_register`, `nk_is_registered`, and `nk_dispatch` in `registry.c`.
  - **Rust**: Graph scheduling and deterministic topological execution in `aria-scheduler`. Implemented `NativeKernelDispatch` using FFI to call C kernels.
  - **Cython**: Zero-copy bridge for high-frequency op dispatch.
- Minimized Python in hot paths:
  - **Rust Scheduler Integration (Step 3)**: Exposed `execute_graph` via PyO3, allowing full graph execution in native space.
  - Python now serves primarily for metadata orchestration and the selective layer replacement adapter.
  - Hot-path `compile_model` logic now routes through native capability checks and selective execution paths.
- Verified stage-by-stage parity:
  - Stage 1 (Probe Mode): Passive validation of Designer runtime vs legacy outputs.
  - Stage 2 (Selective Execution): Active C dispatch on candidate nodes.
  - Stage 3 (Selective Layer Replacement): Full replacement of Torch layers with native WorkflowModules.
- Telemetry: Integrated `selective_guardrail`, fallback metrics, and canary latency benchmarks into dashboard for real-time observability.

### Next execution slice (codex)
1. Keep completed step log in archive and continue appending new completions there after each slice.
2. Add a lightweight CI assertion for telemetry preset helper exports to guard accidental API drift.
3. Extend ABI first-family support from unary+add to a minimal three-op chain (`unary + add + mul`) with deterministic coverage tests.

### Next execution slice (shared)
1. Add schema + ABI compatibility tests and wire them into CI.
2. Add runner/API telemetry endpoint usage in dashboard status panels.
3. Expand real compile/eval reuse path from probe mode to selective execution mode where safe.
4. **Build system**: Add `Makefile.native` with CMake (C/C++ kernels) + Cargo (Rust scheduler) + Cython build targets.
5. **Rust crate init**: Scaffold `research/runtime/native/rust/aria-scheduler/` with Cargo.toml, topological executor stub, memory arena stub.
6. **Kernel batch 1**: Implement C kernels for `relu`, `gelu`, `silu`, `add`, `mul` with SIMD (AVX2/SSE4) + correctness tests vs NumPy reference.
7. **PyO3 bridge stub**: Expose Rust scheduler to Python via PyO3; Cython wrapper for C kernel dispatch.

### Progress notes (claude-opus, 2026-02-21)
- Claimed the heavy native implementation work (WS-A/C/D/E + Phases 0/2/3/4) to complement codex's adapter-first reuse work (WS-B + Phase 1).
- Division of labor: codex owns the Python adapter layer routing through Designer runtime; claude-opus owns the actual native runtime engine, kernels, ABI contracts, and build system.
- Key constraint: all new kernels land in `aria-designer/runtime` first per the no-duplication policy; `research/runtime/native/` only for runner-specific scheduling/orchestration that doesn't belong in Designer.

#### Phase 0 — COMPLETE (claude-opus 2026-02-21)
- **22 files created** across 7 directories:
  - `include/`: Enhanced ABI headers (runner_abi.h with capability query + strict mode, kernel_abi.h with matmul/linear/softmax/rmsnorm/concat/split signatures, profile_abi.h with memory tracking)
  - `src/`: Kernel registry (registry.c/h — wraps Designer void-returning kernels to ABI convention, flat table dispatch), extension kernels (kernels_ext.c/h — softmax, concat, split, layernorm, transpose — temporary, will migrate to Designer)
  - `rust/aria-scheduler/`: Full Rust crate scaffold (graph IR with serde, Kahn's topo executor, 64-byte-aligned memory arena, error types mapping to ABI codes, PyO3 bridge stub)
  - `cython/`: Bridge module (aria_bridge.pyx — zero-copy dispatch for all unary/binary/matmul/linear/rmsnorm ops), declarations (aria_kernels.pxd), build script (setup.py)
  - `tests/`: C test suite (test_kernels.c — 11 tests covering relu, gelu, silu, add/mul/sub, matmul, rmsnorm, softmax, concat/split, layernorm, transpose, registry dispatch)
  - Root: CMakeLists.txt, Makefile, ir_validator.py, REUSE_CONTRACT.md
- **Build verified**: `cmake .. && make -j` builds clean (GCC 13.3, -O3 -march=native), produces `libaria_native_runtime.so` + `.a` + `test_kernels` binary
- **All 11 C kernel tests pass** (ctest + direct execution)
- **IR validator verified**: Correctly accepts valid graphs, rejects cycles + dangling references
- **Rust crate**: Compiled and tested — Rust 1.93.1 installed, **9/9 Rust tests pass** (graph parsing, validation, topo sort, arena alloc/reset/OOM, executor)
- **Cython bridge**: Scaffolded, not yet compiled (needs `pip install cython` + Designer `.so`)

#### Phase 2 — COMPLETE (claude-opus 2026-02-21)
- Migrated 5 extension kernels (softmax, concat, split, layernorm, transpose) from `research/runtime/native/src/kernels_ext.c` → `aria-designer/runtime/src/kernels.c`
- Added declarations to `aria-designer/runtime/src/kernels.h`
- Added 4 new test cases to `aria-designer/runtime/src/test_kernels.c` — **80/80 Designer tests pass**
- Retired `kernels_ext.c/h` in research — now just thin re-export headers pointing to Designer
- Rebuilt `libaria_native_runtime.so` — clean link, **11/11 research C tests still pass**
- No kernel duplication remains — reuse contract fully honored

#### Phase 3 — COMPLETE (claude-opus 2026-02-21)
- Enhanced `native_runner.py` with real native kernel dispatch capabilities:
  - `_try_load_native_lib()` — ctypes loader with path search + caching
  - `_check_native_op_support()` — queries C registry for per-op coverage
  - Three-tier dispatch policy: full native → strict fail → non-strict fallback
  - Telemetry: `native_dispatch_compiles` counter, op coverage in `_native_runner_report`
- **20/20 Python adapter tests pass** (including 9 new tests for lib loading, op support, strict mode)

#### Phase 4 — COMPLETE (claude-opus 2026-02-21)
- **Parity test suite** (`tests/test_native_kernel_parity.py`): 86 tests covering every kernel via ctypes against NumPy references
  - 8 unary ops × 4 sizes, 3 binary ops × 4 sizes, 2 reductions × 4 sizes
  - matmul (4 matrix sizes), linear (with/without bias, 4 configs), rmsnorm (3 configs)
  - softmax (3 configs + sum-to-one invariant), layernorm (3 configs), transpose (4 configs)
  - concat/split round-trip, registry (11 tests)
  - Tolerance: `atol=1e-5, rtol=1e-5` — **all 86 pass**
- **Perf benchmark suite** (`tests/test_native_perf_gates.py`): 18 acceptance gate tests
  - 5 absolute latency gates (relu < 200us, matmul < 5ms, softmax < 500us, layernorm < 200us, rmsnorm < 200us) — **all pass**
  - 13 relative speedup gates (>= 0.5x NumPy) — **7 pass, 6 xfail** (known BLAS/SVML gaps)
  - xfail tracking: silu, sigmoid, exp (need SVML vectorized exp), matmul/linear (need BLAS), softmax (slow expf)
- **Benchmark runner** (`runtime/native/bench/bench_kernels.py`): Full 52-benchmark suite with formatted table output
  - Key wins: relu 3.5x, gelu 3.1x, rmsnorm 1.4x, layernorm 2.8x faster than NumPy
  - Key gaps: matmul 0.05-0.15x (needs BLAS), exp 0.23x (needs SVML)
- **Fallback telemetry** (`tests/test_native_fallback_gate.py`): 15 tests
  - Counter tracking, rate computation, hard threshold gate, min-samples, reset
  - API endpoint `GET /api/native-runner/telemetry` added to `scientist/api.py`
- **Combined test results**: **150 passed, 6 xfailed** across all native runner tests

#### Overall Phase 0-4 test summary
| Suite | Tests | Result |
|-------|-------|--------|
| Research C kernels (ctest) | 11 | 11 pass |
| Designer C kernels | 80 | 80 pass |
| Rust aria-scheduler | 9 | 9 pass |
| Python native runner adapter | 28 | 28 pass |
| Kernel parity (ctypes vs NumPy) | 86 | 86 pass |
| Perf acceptance gates | 18 | 12 pass, 6 xfail |
| Fallback telemetry gates | 15 | 15 pass |
| Full research suite | 168 | 167 pass, 1 pre-existing |
| **Total native runner** | **150** | **150 pass, 6 xfail** |

#### Phase 5 + SIMD/BLAS optimization results (claude-opus, 2026-02-21)

**AVX2 SIMD vectorization** — created `aria-designer/runtime/src/simd_math.h`:
- `_mm256_exp_ps()`: Fast 8-wide vectorized exp (Cephes polynomial, ~1e-6 max relative error)
- `_mm256_sigmoid_ps()`: Built on top of exp intrinsic
- Rewrote exp/sigmoid/silu/softmax kernels with `#ifdef __AVX2__` fast paths
- Results: **exp 1.78x**, **sigmoid 2.47x**, **silu 2.40x** vs NumPy (were 0.23x, 0.4x, 0.35x)

**OpenBLAS linkage** — upgraded to scipy-bundled OpenBLAS 0.3.27 (DYNAMIC_ARCH, Haswell):
- CMakeLists.txt: 3-tier detection reordered: system → **scipy 0.3.27** → ollama 0.3.15
- `include/cblas.h`: Compat header with scipy prefix support (`ARIA_BLAS_SCIPY_PREFIX`)
- matmul/linear use `cblas_sgemm` when `ARIA_HAS_BLAS` defined
- Results: **matmul 1.45x**, **linear ~1.2x** vs NumPy (were 0.29x/0.41x with ollama 0.3.15)

**Phase 5 cutover**:
- Default `NATIVE_RUNNER_ENABLED=True` (was False)
- Added `NATIVE_RUNNER_LEGACY_ONLY` escape hatch for rollback
- Startup logging with configuration summary

**Cython bridge** — built and tested end-to-end (`runtime/native/cython/`):
- Fixed include paths in .pyx/.pxd (use header names, not relative paths)
- Links against `libaria_native_runtime.so` (not Designer's separate lib)
- Added dispatch functions: softmax, layernorm, transpose2d
- 28 tests in `tests/test_cython_bridge.py` — all pass

**Final combined verified test results (all 6 xfails closed)**:
| Suite | Tests | Result |
|-------|-------|--------|
| Research C kernels (ctest) | 11 | 11 pass |
| Kernel parity (ctypes vs NumPy) | 86 | 86 pass |
| Perf acceptance gates | 18 | 18 pass, 0 xfail |
| Fallback telemetry gates | 15 | 15 pass |
| Native runner adapter | 25 | 25 pass |
| Cython bridge | 28 | 28 pass |
| **Total native runner** | **183** | **183 pass, 0 xfail** |

All 6 original xfails closed: AVX2 for exp/sigmoid/silu/softmax, scipy OpenBLAS 0.3.27 for matmul/linear.

#### Post-Phase 5 completed (claude-opus, 2026-02-21)

- Cython bridge wired as preferred dispatch in `native_runner.py` (11 new tests)
- OpenMP threading added to C kernels (`ARIA_HAS_OPENMP`, thresholds: n>16384 / batch>4)
- `runner.py` already routes through `compile_model_native_first` (confirmed)
- Phase 1 integration gaps identified: IR validator not runtime-wired, no graph format serializer, Rust PyO3 stub
- **194 tests, 0 xfail**

#### Integration + dispatch work completed (claude-opus, 2026-02-21)

- Created `synthesis/native_ir_converter.py`: `graph_to_native_ir()` + `graph_to_native_ir_json()`
  - Converts ComputationGraph → native_ir.v1 (array nodes, explicit edges, schema_version)
  - Strips `output_shape` (schema `additionalProperties: false`)
- IR validator wired into `compile_model_native_first()` — observational (warnings only)
- Fixed `dispatch_graph_native` to use `graph_to_native_ir_json` (was using wrong format)
- Fixed JS `true` → Python `True` bug in Rust scheduler probe
- Migrated `api.py:2458` compile_model import to native-first path
- Created `NativeForwardWrapper` class for intercepting ops during forward pass
- 14 IR converter tests + 11 cross-format compat tests + 6 forward wrapper tests

**Updated total: 224 tests (196 pass, 28 skip), 0 failures**

#### CompiledOp hook + E2E + benchmarks — COMPLETE (claude-opus, 2026-02-21)

- **CompiledOp forward hook**: Modified `synthesis/compiler.py` `CompiledOp.forward()` to check `_native_wrapper` attribute. If set, dispatches through native C kernels; falls through to original PyTorch path if unset or dispatch returns None. Added wrapper propagation in `compile_model_native_first()`. 5 tests.
- **Rust registry init fix**: `execute_graph()` was failing with "op not registered" — `aria_registry_init()` was never called from Rust. Fixed by adding `ensure_registry_init()` with `std::sync::Once` guard in `NativeKernelDispatch::dispatch()` and adding `aria_registry_init` to FFI extern block.
- **E2E native execution pipeline**: Full path tested: ComputationGraph → native_ir.v1 → Rust scheduler → C kernels → output. Fixed `dispatch_graph_native` reshape bug (assumed 3D inputs). 18 tests.
- **Benchmarks**: Created `bench/bench_e2e.py` and `tests/test_e2e_benchmark.py`. Key results: native relu 0.63x PyTorch (faster), gelu 8.5x (slower, needs AVX2 gelu), matmul 1.72x (comparable). 7 regression gate tests.

**Combined verified: 155 native tests, 0 failures, 0 xfails**

| Suite | Tests | Result |
|-------|-------|--------|
| Research C kernels (ctest) | 11 | 11 pass |
| Kernel parity (ctypes vs NumPy) | 86 | 86 pass |
| Perf acceptance gates | 18 | 18 pass |
| Fallback telemetry gates | 15 | 15 pass |
| Native runner adapter | 37 | 37 pass |
| Cython bridge | 28 | 28 pass |
| Cython dispatch integration | 11 | 11 pass |
| IR converter | 14 | 14 pass |
| Cross-format compat | 11 | 11 pass |
| Forward wrapper | 6 | 6 pass |
| CompiledOp hook | 5 | 5 pass |
| E2E native execution | 18 | 18 pass |
| E2E benchmark gates | 7 | 7 pass |
| **Total native runner** | **155** | **155 pass** |

#### AVX2 gelu + multi-node + arena + profiling — COMPLETE (claude-opus, 2026-02-21)

**AVX2 GELU kernel** (`aria-designer/runtime/src/kernels.c`):
- Vectorized exact tanh-based GELU formula using `tanh(z) = 2*sigmoid(2z) - 1` with `_mm256_sigmoid_ps`
- Processes 8 floats at a time with FMA instructions, scalar tail fallback
- Result: **1.67x faster than PyTorch** (was 8.5x slower)

**Rust multi-node graph execution** (`executor.rs`):
- Executor already handled multi-node correctly (topo sort, intermediate HashMap, binary ops)
- Added 17 new e2e tests: binary op chains, diamond topology, 4-op chains, double residual patterns
- Total e2e tests: 35

**Memory arena integration** (`arena.rs` + `executor.rs`):
- Added `alloc_f32_raw()` for concurrent arena allocations without borrow conflicts
- New `execute_with_arena()`: pre-sizes arena from graph, allocates per-node buffers, falls back to heap on exhaustion
- `ArenaStats` tracks: bytes_used, capacity, alloc_count, heap_fallback_count
- Exposed `execute_graph_with_stats()` via PyO3, `dispatch_graph_native()` logs arena stats at DEBUG level
- 12/12 Rust tests pass (3 new arena tests)

**Profile ABI** (`src/profiler.c` + Rust + Python):
- Ring buffer (8192 entries) for timing and memory events, thread-safe via pthread_mutex
- Opt-in via `NATIVE_RUNNER_PROFILE=1` or `np_profiler_enable()`, zero overhead when disabled
- `clock_gettime(CLOCK_MONOTONIC)` nanosecond timing, per-node profiling in executor
- Rust: `profiler_enable/enabled/reset` + `node_profiles` in `execute_graph_with_stats`
- Python: `enable_native_profiling()`, `get_native_profile()` in native_runner.py
- 9 C tests + 9 Python tests

**Combined verified: 181 Python + 20 C + 12 Rust = 213 tests, 0 failures**

| Suite | Tests | Result |
|-------|-------|--------|
| C kernels (ctest) | 11 | 11 pass |
| C profiler (ctest) | 9 | 9 pass |
| Rust aria-scheduler | 12 | 12 pass |
| Kernel parity (ctypes vs NumPy) | 86 | 86 pass |
| Perf acceptance gates | 18 | 18 pass |
| Fallback telemetry gates | 15 | 15 pass |
| Native runner adapter | 37 | 37 pass |
| Cython bridge | 28 | 28 pass |
| Cython dispatch integration | 11 | 11 pass |
| IR converter | 14 | 14 pass |
| Cross-format compat | 11 | 11 pass |
| Forward wrapper | 6 | 6 pass |
| CompiledOp hook | 5 | 5 pass |
| E2E native execution | 35 | 35 pass |
| E2E benchmark gates | 7 | 7 pass |
| Native profiling | 9 | 9 pass |
| **Total** | **213** | **213 pass** |

#### OpenMP+arena + profiling API + subgraph dispatch + gradients — COMPLETE (claude-opus, 2026-02-21)

**OpenMP + arena thread safety** — verified safe by construction:
- Arena is per-execution (stack-local in `execute_with_arena()`), no sharing between calls
- Allocations are sequential (topo order), OpenMP parallelism confined within single kernel calls
- 64-byte alignment prevents false sharing; GIL serializes Python-level concurrency
- 22 stress tests: large tensors (>16384), concurrent threads (8-16), determinism checks

**Profiling API endpoints** (`scientist/api.py`):
- `GET /api/native-runner/profile` — returns last execution's node_profiles, peak_memory_bytes, total_duration_us
- `POST /api/native-runner/profile/enable` — toggles profiling on/off
- 6 API tests

**SubgraphDispatcher** (`scientist/native_runner.py` + `synthesis/compiler.py`):
- New `SubgraphDispatcher` class: checks if all ops in a ComputationGraph are native-supported, converts to native_ir.v1 JSON, dispatches through Rust scheduler in a single call
- Wired into `compile_model_native_first()` — attaches to each layer with full native coverage
- `CompiledLayer.forward()` tries subgraph dispatch first, falls back to per-op on failure
- Eliminates per-op Python→C roundtrips for fully-native subgraphs
- 16 tests

**Backward (gradient) kernels** (`aria-designer/runtime/src/kernels.c`):
- 9 backward kernels with AVX2 SIMD + OpenMP: relu, sigmoid, tanh, gelu, silu, add, mul, sub, matmul
- matmul_backward uses BLAS `cblas_sgemm` with transposed args
- Registered in kernel registry with new backward function pointer types in `kernel_abi.h`
- 40 parity tests (4 sizes each, plus numerical gradient checks)

**Combined verified: 268 Python + 20 C + 12 Rust = 300 tests, 0 failures**

| Suite | Tests | Result |
|-------|-------|--------|
| C kernels (ctest) | 11 | 11 pass |
| C profiler (ctest) | 9 | 9 pass |
| Rust aria-scheduler | 12 | 12 pass |
| Kernel parity (ctypes vs NumPy) | 86 | 86 pass |
| Perf acceptance gates | 18 | 18 pass |
| Fallback telemetry gates | 15 | 15 pass |
| Native runner adapter | 37 | 37 pass |
| Cython bridge | 28 | 28 pass |
| Cython dispatch integration | 11 | 11 pass |
| IR converter | 14 | 14 pass |
| Cross-format compat | 11 | 11 pass |
| Forward wrapper | 6 | 6 pass |
| CompiledOp hook | 5 | 5 pass |
| E2E native execution | 35 | 35 pass |
| E2E benchmark gates | 7 | 7 pass |
| Native profiling | 9 | 9 pass |
| OpenMP + arena stress | 22 | 22 pass |
| Profile API | 6 | 6 pass |
| Subgraph dispatch | 16 | 16 pass |
| Gradient parity | 40 | 40 pass |
| **Total** | **300** | **300 pass** |

#### Cython backward + autograd + dashboard + benchmarks — COMPLETE (claude-opus, 2026-02-21)

**Cython backward dispatch** (`aria_bridge.pyx` + `native_runner.py`):
- Added extern declarations for 9 backward kernels to `aria_kernels.pxd`
- Added `dispatch_unary_backward`, `dispatch_binary_backward`, `dispatch_matmul_backward`, `has_backward` to Cython bridge
- Added `dispatch_op_backward_native(op_name, grad_output, *saved_tensors)` to native_runner.py
- 24 tests

**Autograd integration** (`scientist/native_autograd.py`):
- 9 `torch.autograd.Function` subclasses: NativeRelu, NativeSigmoid, NativeTanh, NativeGelu, NativeSilu, NativeAdd, NativeMul, NativeSub, NativeMatmul
- Each saves required tensors for backward, routes through C gradient kernels
- `native_autograd_dispatch(op_name, *inputs)` registry function
- `NativeForwardWrapper.dispatch()` now uses autograd path when `requires_grad=True`
- 33 tests (including torch.autograd.gradcheck for all ops)

**Dashboard profiling panel** (`dashboard/src/components/NativeProfilePanel.js`):
- Toggle to enable/disable profiling via POST endpoint
- Auto-refresh every 2 seconds when enabled
- Horizontal bar chart showing per-node timing (color-coded by op type)
- Summary stats: total_duration_us, peak_memory_bytes, node count
- Wired into App.js Optimization tab

**Subgraph vs per-op benchmark** (`tests/test_subgraph_benchmark.py`):
- Compared 3 paths: per-op Cython, subgraph E2E (with IR conversion), subgraph pre-converted
- Cython per-op is ~1.2us/op — extremely fast due to minimal bridge overhead
- Rust subgraph has ~10us fixed IR serialization cost, crossover near 10+ ops
- Key insight: caching IR JSON in SubgraphDispatcher would make subgraph path competitive for medium graphs
- 13 tests

**Combined verified: 340 Python + 20 C + 12 Rust = 372 tests, 0 failures**

| Suite | Tests | Result |
|-------|-------|--------|
| C kernels + profiler (ctest) | 20 | 20 pass |
| Rust aria-scheduler | 12 | 12 pass |
| Kernel parity | 86 | 86 pass |
| Perf gates | 18 | 18 pass |
| Fallback telemetry | 15 | 15 pass |
| Native runner adapter | 37 | 37 pass |
| Cython bridge (forward) | 28 | 28 pass |
| Cython backward | 24 | 24 pass |
| Cython dispatch integration | 11 | 11 pass |
| IR converter | 14 | 14 pass |
| Cross-format compat | 11 | 11 pass |
| Forward wrapper | 6 | 6 pass |
| CompiledOp hook | 5 | 5 pass |
| E2E native execution | 35 | 35 pass |
| E2E benchmark gates | 7 | 7 pass |
| Native profiling | 9 | 9 pass |
| OpenMP + arena stress | 22 | 22 pass |
| Profile API | 6 | 6 pass |
| Subgraph dispatch | 16 | 16 pass |
| Gradient parity | 40 | 40 pass |
| Native autograd | 33 | 33 pass |
| Subgraph benchmark | 13 | 13 pass |
| **Total** | **372** | **372 pass** |

#### IR cache + Rust backward + training loop + norm gradients — COMPLETE (claude-opus, 2026-02-21)

- **IR JSON caching**: SubgraphDispatcher pre-converts at init, eliminates ~10us/call. 20 tests.
- **Rust backward execution**: FFI for 9 backward kernels, `execute_backward_with_arena()` with reverse topo traversal, gradient accumulation. 14 Rust + 9 Python tests.
- **Autograd + training**: 9 `torch.autograd.Function` subclasses, `NativeForwardWrapper` auto-routes requires_grad. Training convergence verified. 33 + 8 tests.
- **Norm backward kernels**: softmax/layernorm/rmsnorm backward in C with OpenMP. 24 parity tests.
- **Profile API fix**: Updated test paths to codex's `/api/native-profile/v2/*` routes.

**Combined: 425 tests (391 Python + 20 C + 14 Rust), 0 failures**

#### Norm autograd + backward subgraph + fp16 + fusion — COMPLETE (claude-opus, 2026-02-21)

- **Norm backward in Cython + autograd**: softmax/layernorm/rmsnorm backward wired through Cython bridge, 3 new autograd Functions (NativeSoftmax, NativeLayernorm, NativeRmsnorm). 16 tests.
- **Backward subgraph dispatch**: `NativeSubgraphFunction` autograd Function for graph-level backward through per-op native kernels. SubgraphDispatcher auto-routes requires_grad inputs. 20 tests.
- **FP16 kernels**: 9 fp16 variants (relu, gelu, silu, sigmoid, add, mul, matmul, softmax, rmsnorm) using F16C intrinsics for fp16↔fp32 conversion at kernel boundaries. 37 parity tests.
- **Fused kernels**: matmul_relu, matmul_bias_relu, matmul_gelu, layernorm_residual. layernorm_residual shows 2x speedup. 31 tests.

**Combined: 504 Python + 20 C + 14 Rust = 538 tests, 0 failures**

#### Next execution priorities (claude-opus)
1. **Wire fused kernels into dispatch**: Expose fused ops through Cython + add fusion detection in SubgraphDispatcher
2. **Wire fp16 into Cython + dispatch**: Expose fp16 kernels for mixed-precision inference
3. **Graph fusion pass in IR converter**: Detect fusible op sequences in ComputationGraph and emit fused op names in native_ir
4. **Attention kernel**: Fused multi-head attention (Q@K^T/sqrt(d), softmax, V multiply) — the biggest remaining optimization target

#### Codex incremental progress (2026-02-22)
- Added optional runner ABI prepare path in `scientist/native_runner.py` (`NATIVE_RUNNER_ABI_EXEC=1`):
  - compile first graph via `nr_compile`
  - execute smoke via `nr_execute`
  - attach reusable model session handle (`_native_runner_abi_session`)
  - strict-mode fail-fast if ABI prepare fails
- Added optional sandbox ABI probe in `eval/sandbox.py` (`NATIVE_RUNNER_ABI_INFER_PROBE`):
  - executes token payloads through attached ABI session
  - records `native_abi_probe` telemetry
- Added optional sandbox ABI primary mode (`NATIVE_RUNNER_ABI_INFER_PRIMARY=1`):
  - uses ABI logits as primary Stage-0 forward output
  - forward-only/no-grad path skips backward
- Added regression coverage:
  - `tests/test_native_runner_adapter.py` (ABI session attach + strict fail-fast)
  - `tests/test_native_runner_abi_inference_probe.py` (probe success/failure + primary forward-only mode)
- Added runner-level stage routing for sandbox ABI modes:
  - `scientist/runner.py` now routes all safe-eval callsites through `_safe_eval_for_stage(...)`
  - stage controls via `NATIVE_RUNNER_ABI_PRIMARY_STAGES` and `NATIVE_RUNNER_ABI_PROBE_STAGES`
  - `eval/sandbox.py` accepts explicit ABI mode args for deterministic stage-level control
- Added routing tests:
  - `tests/test_runner_safe_eval_stage_routing.py`
- Added sampled ABI parity gate for primary mode:
  - `eval/sandbox.py` now supports sampled ABI-vs-Torch forward parity with drift telemetry (`max_abs`, `mean_abs`, pass/fail)
  - strict fail policy via `NATIVE_RUNNER_ABI_PARITY_STRICT=1`
  - controls: `NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE`, `NATIVE_RUNNER_ABI_PARITY_MAX_ABS`
- Progress surfacing:
  - runner now writes latest ABI probe/parity payload into progress telemetry (`native_runner.abi_last_probe`, `native_runner.abi_last_stage`) for `/api/status` and `/api/progress` consumers.
- Dashboard surfacing:
  - `dashboard/src/components/ControlPanel.js` now shows ABI parity health badge and drift details in both idle system badges and running native telemetry card.
  - `dashboard/src/components/StatusBar.js` now shows compact ABI parity badge (`pending`/`pass`/`fail`) with max-abs drift during active runs.
- Dashboard legend/threshold visibility:
  - `ControlPanel` expanded native telemetry now shows ABI gate settings (`strict|observe`, parity sample rate, max-abs threshold).
  - `StatusBar` ABI badge tooltip now includes gate settings so parity outcomes are interpretable from UI alone.
- Cutover-readiness telemetry:
  - `native_runner` now tracks `fallback_metrics.legacy_compile_invocations` to quantify remaining legacy compile usage.
  - `ControlPanel` now surfaces legacy-compile count during runs and in native telemetry details.
- Cutover enforcement gate:
  - `native_runner` now supports `NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS` fail-fast threshold.
  - when configured, compile fails once legacy compile usage exceeds threshold.
  - dashboard now surfaces configured legacy limit alongside observed usage.
- Cutover readiness UX:
  - dashboard `ControlPanel` now computes and displays a unified cutover status (`waiting`, `ready`, `blocked`) from parity + fallback + legacy gates.
  - dashboard `StatusBar` now shows compact cutover readiness badge during active runs.
- Canonical backend cutover verdict:
  - `native_runner_capability_report()` now emits `cutover_gate` with backend-computed verdict (`ready`, `status`, `checks`).
  - dashboard now prefers backend verdict and uses frontend-computed status only as fallback.
  - optional parity requirement gate is now supported in backend via `NATIVE_RUNNER_REQUIRE_PARITY_PASS=1`.
- Contract/docs hardening:
  - integration test now asserts `/api/native-runner/capability` includes `fallback_metrics` + `cutover_gate` shape/status.
  - README docs updated with cutover env controls and capability endpoint contract.
- Controlled removal execution plan:
  - see `research/CUTOVER_REMOVAL_PLAN.md` for phased legacy-path retirement, hard gates, rollback policy, and claimable tasks.
- Legacy-path cleanup started:
  - `eval/pruning.py` now compiles via native-first adapter instead of direct `synthesis.compiler` import.
  - `runtime/native/bench/bench_e2e.py` now compiles via native-first adapter instead of direct `synthesis.compiler` import.
- Controlled hard-cutover gate added:
  - `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE=1` now explicitly rejects compile when legacy path would be invoked.
  - intended for canary enforcement before final removal of `_legacy_compile_model(...)`.
- Cutover automation + verdict tests:
  - added backend cutover-gate transition tests (`waiting`, `blocked`, `ready`) in native runner adapter suite.
  - added `research/tools/check_cutover_gate.py` for scripted gate checks against `/api/native-runner/capability`.
- CI canary wiring:
  - `research-native-ci.yml` now runs cutover-gate canary in enforce-ready mode with strict parity gate enabled.
  - lane uses deterministic compile+parity sample generation (`check_cutover_gate --offline --generate-compile-sample --generate-parity-sample`) so fallback/legacy/parity checks all evaluate from real telemetry.
- ABI first-family extension (runner ABI):
  - `runtime/native/src/runner_abi.c` now requires `unary + add + mul` family markers at compile-time and executes deterministic `add -> mul -> unary` kernel chain.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires unary+add+mul family graphs.
  - coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (focused suite passing).
- ABI first-family extension (runner ABI, matmul increment):
  - `runtime/native/src/runner_abi.c` now requires `unary + add + mul + matmul` family markers and executes deterministic `add -> mul -> matmul -> unary` chain.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires unary+add+mul+matmul family graphs.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (`45 passed`).
- ABI first-family extension (runner ABI, linear increment):
  - `runtime/native/src/runner_abi.c` now requires `unary + add + mul + matmul + linear` family markers and executes deterministic `add -> mul -> matmul -> linear -> unary` chain.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires unary+add+mul+matmul+linear family graphs.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (`45 passed`).
- ABI first-family extension (runner ABI, softmax increment):
  - `runtime/native/src/runner_abi.c` now requires `unary + add + mul + matmul + linear + softmax` family markers and executes deterministic `add -> mul -> matmul -> linear -> softmax -> unary` chain.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires unary+add+mul+matmul+linear+softmax family graphs.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (`47 passed`).
- ABI first-family extension (runner ABI, rmsnorm increment):
  - `runtime/native/src/runner_abi.c` now requires `unary + add + mul + matmul + linear + softmax + rmsnorm` family markers and executes deterministic `add -> mul -> matmul -> linear -> softmax -> rmsnorm -> unary` chain.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires unary+add+mul+matmul+linear+softmax+rmsnorm family graphs.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (`49 passed`).
- ABI first-family extension (runner ABI, sub increment):
  - `runtime/native/src/runner_abi.c` now requires `unary + add + mul + matmul + linear + softmax + rmsnorm + sub` family markers and executes deterministic `add -> mul -> matmul -> linear -> softmax -> rmsnorm -> sub -> unary` chain.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires unary+add+mul+matmul+linear+softmax+rmsnorm+sub family graphs.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (`49 passed`).
- ABI first-family contract tightening (explicit exp marker):
  - `runtime/native/src/runner_abi.c` now requires an explicit `exp` marker in addition to `add + mul + matmul + linear + softmax + rmsnorm + sub` family markers.
  - `scientist/native_runner.py` ABI prepare candidate selection now requires `exp` explicitly (alongside current family set).
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` (`51 passed`).
- ABI first-family contract tightening (ordered markers):
  - `runtime/native/src/runner_abi.c` now enforces ordered first occurrence for the family markers (`exp -> add -> mul -> matmul -> linear -> softmax -> rmsnorm -> sub`) instead of marker presence-only admission.
  - `scientist/native_runner.py` ABI prepare candidate selection now mirrors the same ordered-family requirement before attempting ABI session preparation.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with out-of-order rejection checks (`53 passed`).
- ABI first-family contract tightening (dependency chain links):
  - `runtime/native/src/runner_abi.c` now additionally requires dependency links across the family chain via node `input_ids` (`exp→add→mul→matmul→linear→softmax→rmsnorm→sub`) rather than relying on markers/order alone.
  - `scientist/native_runner.py` ABI prepare candidate selection now validates the same chain links using node `input_ids` (and `edges` when present) before attempting ABI session preparation.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with unlinked-family rejection checks (`55 passed`).
- ABI first-family contract tightening (transitive topology links):
  - `runtime/native/src/runner_abi.c` now validates transitive ancestor reachability across the family chain through recursive `input_ids` ancestry checks, so required relationships can be satisfied via intermediate ops.
  - `scientist/native_runner.py` ABI prepare candidate selection now mirrors transitive ancestor-path checks over graph ancestry (from `input_ids` and `edges`).
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with transitively linked acceptance checks (`57 passed`).
- ABI first-family contract tightening (explicit edges parity):
  - `runtime/native/src/runner_abi.c` now requires transitive family-chain reachability through explicit IR `edges` when edges are present, in addition to existing `input_ids` ancestry checks.
  - `scientist/native_runner.py` ABI prepare candidate selection now enforces the same rule: if explicit edges exist, chain ancestry must hold in the edge graph as well.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with edge-mismatch rejection checks (`59 passed`).
- ABI first-family contract tightening (strict direct-link parity):
  - `runtime/native/src/runner_abi.c` now requires strict direct parent links for the ordered family chain (`exp→add→mul→matmul→linear→softmax→rmsnorm→sub`) via immediate `input_ids` membership and immediate explicit-edge parent checks (when edges are present).
  - `scientist/native_runner.py` ABI prepare candidate selection now mirrors strict direct parent parity for both graph ancestry sources (`input_ids` and explicit `edges`).
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` by flipping transitive-link acceptance to rejection under the stricter contract (`61 passed`).
- ABI first-family contract tightening (declared-edges strictness):
  - `runtime/native/src/runner_abi.c` now treats declared `edges` arrays as strict parity contracts: when `edges` are declared, required ordered-family direct links must be present in explicit edges (including rejecting `"edges": []`).
  - `scientist/native_runner.py` ABI prepare candidate selection now mirrors this rule by enforcing explicit-edge direct-link checks whenever `graph.edges` is declared (not only when non-empty).
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with declared-empty-edges rejection checks (`65 passed`).
- ABI first-family contract tightening (unique marker identity):
  - `runtime/native/src/runner_abi.c` now requires exactly one occurrence of each required family marker (`exp/add/mul/matmul/linear/softmax/rmsnorm/sub`) instead of accepting duplicate marker nodes under first-occurrence semantics.
  - `scientist/native_runner.py` ABI prepare candidate selection now enforces the same uniqueness rule before ordered-chain and linkage checks.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with duplicate-marker rejection checks (`67 passed`).
- ABI first-family contract tightening (independent linkage parity sources):
  - `scientist/native_runner.py` now validates required direct family-chain links against `input_ids` independently of `edges`, preventing edge-only links from satisfying the input linkage contract.
  - `runtime/native/src/runner_abi.c` already enforced this split via dedicated input-id and edge checks; Python candidate gating now matches that strictness.
  - focused coverage updated in `tests/test_native_runner_abi_smoke.py` and `tests/test_native_runner_adapter.py` with edge-only-link rejection checks (`70 passed`).
