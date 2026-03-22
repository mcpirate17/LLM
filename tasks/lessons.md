# Lessons Learned

Track corrections and recurring mistakes to prevent them from happening again.
Update this file after ANY correction from the user.

---

## 2026-03-16 — Audit bootstrap
- This file was missing despite being required by CLAUDE.md since project inception.
- All agents must review this file at session start and update it after corrections.

## 2026-03-16 — Full workspace audit lessons

### Packaging
- `research/` pytest.ini requires marker-based test selection (`-m "unit or api"`). Running `pytest tests/` without a marker will exit with an error. The Makefile `test-research` target must include `-m "unit or api"`.
- 45+ `sys.path.insert` calls across the workspace. New scripts should use `python -m research.tools.X` pattern instead of sys.path surgery.

### Dead code
- DRY consolidation was incomplete: `shared_utils.py` had 3 exports (`safe_json_loads`, `ensure_metadata_dict`, `canonicalize_text`) that were created during dedup but never called. Always grep for callers after moving code.
- `_safe_float` had 3 copies despite a dedup pass. After consolidation, grep the whole repo for the old function name to catch stragglers.
- `aria_core/aria_core/gpu/` was 5,631 lines of dead code from the HYDRA era. Ported code that isn't wired into imports is invisible dead weight.

### Testing
- Dashboard consistency tests (`TestDashboardConsistency`) grep React source files for specific strings. These drift frequently and produce false negatives. They need a maintenance pass or a less brittle approach.
- `test_norm_autograd.py` fails when run from `research/` cwd because it imports `scientist.X` (bare module) which then uses `from ...defaults` relative imports that can't resolve. This is a sys.path packaging issue.

### Multi-agent coordination
- File-based coordination (claims files, TEST_LOCK) works for 3-4 agents. The conflicts.md pattern caught the dead C++ file uncertainty.
- ARIA_CORE finishing first is correct — it has no dependencies and unblocks build verification for downstream agents.

## 2026-03-21 — Under-observed component audit

### C kernel parity bugs
- `aria_core.tropical_matmul_batched_f32` returns (B, S, D) instead of expected (B, S, S) when both inputs have the same shape. The Python fallback returns the correct shape. Always validate C kernel output shapes and fall through to Python if mismatched.
- General rule: any C kernel that replaces a Python op must have its output shape validated by the caller. A single wrong shape from a C kernel can cascade into silent shape mismatches throughout the model.

### Split op contiguity
- `split2`, `split3`, `split4` compiler handlers return tensor slices that are non-contiguous views. Downstream ops (especially C kernels) crash with "x must be contiguous". Always call `.contiguous()` on slice returns.
- General rule: any compiler op handler that returns a tensor view must ensure contiguity.

### Op role mapping gaps
- `n_way_sparse_router` was NOT in `_OP_ROLE_MAP` and fell back to category default (PARAMETERIZED → PROJECT instead of ROUTE). This caused role validation mismatches.
- General rule: every new op must have an explicit entry in `_OP_ROLE_MAP`. Category fallbacks are too coarse.

### Motif class isolation
- `reduce_core`, `guarded_act`, and `moe_core` motif classes were only in `_ALL_CLASSES`, not in `_FFN_CLASSES` or `_MIXER_CLASSES`. Templates mostly use `_FFN_CLASSES` and `_MIXER_CLASSES`, so these classes were rarely sampled.
- General rule: when adding a new motif class, ensure it's in at least one of the frequently-used class groups (`_FFN_CLASSES` or `_MIXER_CLASSES`).
