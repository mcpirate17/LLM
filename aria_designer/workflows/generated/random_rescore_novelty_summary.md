# Random Rescore Novelty Summary

Sampled successful candidates: `12`
Average old buggy bridge novelty: `0.217`
Average fixed bridge novelty: `0.290`
Average uplift: `0.073`
Median uplift: `0.075`
Uplift range: `0.024` to `0.118`

## Observation

In this random successful sample, every candidate gained novelty after behavioral fingerprinting was threaded into bridge scoring. The effect is systematic rather than isolated.

## Recommended Rescore Policy

1. Rescore all non-failing candidates that were scored through the bridge while the novelty bug was present.
2. Highest priority: candidates near ranking / promotion cutoffs, because a typical uplift of ~0.07 is large enough to reorder them materially.
3. Next priority: routed or otherwise advanced graphs, because the old structural-only path disproportionately penalized them.
4. Lowest priority: candidates that already fail sandbox or do not produce a valid bridge evaluation, because they do not have a comparable novelty score yet.

## Largest Uplifts In Sample

- `imported_f32e22ad-5c4` (`f32e22ad-5c4`): `0.170` -> `0.288` (`+0.118`)
- `imported_ref_rwkv_61754c8e` (`ref_rwkv_61754c8e`): `0.170` -> `0.263` (`+0.093`)
- `imported_563e60ca-cff` (`563e60ca-cff`): `0.190` -> `0.283` (`+0.093`)
- `imported_1df90e93-4b0` (`1df90e93-4b0`): `0.214` -> `0.297` (`+0.083`)
- `imported_5bc26a03-ed2` (`5bc26a03-ed2`): `0.180` -> `0.263` (`+0.083`)
