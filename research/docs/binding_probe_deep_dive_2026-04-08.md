# Binding Probe Deep Dive

## Summary

The current `binding_auc` implementation is not obviously broken as code, but it
is weak as a governance signal.

Main findings:

- `binding_auc` is a zero-shot copy-at-distance metric.
- `induction_auc` and `ar_auc` are train-to-task probes.
- This mismatch makes direct threshold comparison fragile.
- Live frontier fingerprints with strong loss often remain near
  `binding_auc ~= 0.003-0.005`.
- The threshold commentary currently expects much larger values
  (`>0.1` or `>0.3`), which does not match the live regime.
- The repository lacked an actual calibration test asserting that the binding
  method separates attention from local-only architectures under the live
  protocol.
- Reference calibration now shows the old zero-shot metric does **not**
  separate attention from local or recurrent references at all.
- The first adapted metric (`binding_auc_adapted`) is also too weakly trained
  to separate those references under its original 300-step regime.
- A new curriculum-trained metric does separate them.

## What Was Not Changed

- The meaning of the existing `binding_auc` column was preserved.
- Existing scoring and notebook logic that read `binding_auc` were not silently
  repointed to a new metric.

## What Was Added

### New Metric

Added an adapted binding metric in
[binding_adapted.py](/home/tim/Projects/LLM/research/eval/binding_adapted.py):

- `binding_auc_adapted`
- `binding_distance_accuracies_adapted_json`
- `binding_probe_adapted_steps`
- `binding_probe_adapted_elapsed_ms`
- `binding_probe_adapted_protocol_version`

This metric:

- deep-copies the model
- briefly trains it on copy-at-distance batches
- then runs the existing binding range probe

Interpretation:

- `binding_auc`: zero-shot copy transfer from language micro-train
- `binding_auc_adapted`: architecture ability after direct copy-task adaptation

Added a stronger curriculum metric in
[binding_curriculum.py](/home/tim/Projects/LLM/research/eval/binding_curriculum.py):

- `binding_auc_curriculum`
- `binding_distance_accuracies_curriculum_json`
- `binding_probe_curriculum_steps`
- `binding_probe_curriculum_elapsed_ms`
- `binding_probe_curriculum_protocol_version`

This metric:

- deep-copies the model
- trains on exact copy targets only, not the full LM objective
- cycles across multiple copy distances
- evaluates exact copy accuracy by distance after training

### Persistence

The new fields were added to notebook migration metadata in
[_shared.py](/home/tim/Projects/LLM/research/scientist/notebook/_shared.py).

### Runtime Wiring

Investigation and validation now compute and persist the adapted metric in
[_helpers.py](/home/tim/Projects/LLM/research/scientist/runner/_helpers.py).

Investigation and validation now also compute and persist the curriculum metric
in
[_helpers.py](/home/tim/Projects/LLM/research/scientist/runner/_helpers.py).

### Calibration Test

Added
[test_binding_adapted_probe.py](/home/tim/Projects/LLM/research/tests/test_binding_adapted_probe.py),
which checks that the adapted probe gives a higher score to a minimal attention
graph than to a local-only conv graph.

Added
[test_binding_curriculum_probe.py](/home/tim/Projects/LLM/research/tests/test_binding_curriculum_probe.py),
which checks the stronger curriculum probe separates minimal attention from a
local-only conv graph.

## Reference Calibration

Calibration output:

- [binding_reference_calibration_2026-04-08.md](/home/tim/Projects/LLM/research/docs/binding_reference_calibration_2026-04-08.md)

Headline result:

| Model | `binding_auc` | `binding_auc_adapted` | `binding_auc_curriculum` |
| --- | ---: | ---: | ---: |
| `local_conv` | 0.0029 | 0.0042 | 0.0028 |
| `gpt2` | 0.0029 | 0.0032 | 0.0803 |
| `mamba` | 0.0029 | 0.0029 | 0.0049 |
| `rwkv` | 0.0029 | 0.0030 | 0.0044 |

Interpretation:

- The old zero-shot metric is flat and not discriminative on the live
  reference panel.
- The first adapted metric is still too weakly configured to be useful as a
  governance signal.
- The curriculum metric is the first binding-family signal in this repo that
  clearly separates GPT-style attention from local and recurrent references.

## Practical Recommendation

For now:

- keep `binding_auc` for backward compatibility
- do not treat current `binding_auc` thresholds as strongly calibrated
- do not use `binding_auc_adapted` for promotion logic yet
- use `binding_auc_curriculum` as the better forward-looking diagnostic for
  whether an architecture can acquire copy-at-distance behavior

Next recommended step:

- backfill `binding_auc_curriculum` on a representative frontier slice
- derive fresh thresholds from that column instead of reusing the old
  `binding_auc` bands
- keep any future scoring change isolated to the new column name
