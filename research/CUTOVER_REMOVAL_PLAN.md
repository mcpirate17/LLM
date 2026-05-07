# Native Runner Legacy Removal Plan

This file is the source of truth for removing the remaining legacy compile and execution paths from `research/`.

## Exit Criteria

- `NATIVE_RUNNER_ABI_MODEL_ONLY=1` works for the supported compile/run matrix.
- `python -m research.tools.check_no_legacy_compile` passes.
- `python -m research.tools.check_no_legacy_execution_paths` passes.
- Fallback telemetry stays within the configured cutover gate thresholds.
- Dashboard capability payload remains accurate after legacy-path removal.

## Rollout Stages

1. Observe
   - Keep native runner enabled in permissive mode.
   - Track fallback rate and legacy compile invocations.
   - Fix unsupported ops or execution gaps surfaced by telemetry.

2. Canary
   - Run strict no-legacy checks with:
     - `NATIVE_RUNNER_ENABLED=1`
     - `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1`
     - `NATIVE_RUNNER_ABI_MODEL_ONLY=1`
   - Gate merges on the no-legacy verification tools for the canary lane.

3. Cutover
   - Remove emergency use of `NATIVE_RUNNER_LEGACY_ONLY`.
   - Reject legacy compile whenever native-runner mode is enabled.
   - Keep parity and telemetry checks active during the first cutover window.

4. Removal
   - Delete dead legacy-only code paths and deprecated telemetry aliases.
   - Remove rollout flags that only exist for the compatibility window.
   - Update README and operator docs in the same change.

## Maintenance Rule

- Any README reference to cutover policy must point to this file.
- Any future legacy-path deletion must include capability-report regression coverage.
